# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Helper wrapper for a Tensorflow optimizer."""

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from typing import List, Union

from . import autosummary
from . import tfutil
from .. import util

from .tfutil import TfExpression, TfExpressionEx

try:
    # TensorFlow 1.13
    from tensorflow.python.ops import nccl_ops
except:
    # Older TensorFlow versions
    import tensorflow.contrib.nccl as nccl_ops

class Optimizer:
    """A Wrapper for tf.train.Optimizer.

    Automatically takes care of:
    - Gradient averaging for multi-GPU training.
    - Gradient accumulation for arbitrarily large minibatches.
    - Dynamic loss scaling and typecasts for FP16 training.
    - Ignoring corrupted gradients that contain NaNs/Infs.
    - Reporting statistics.
    - Well-chosen default settings.
    """

    def __init__(self,
        name:                   str             = "Train",                  # Name string that will appear in TensorFlow graph.
        tf_optimizer:           str             = "tf.train.AdamOptimizer", # Underlying optimizer class.
        learning_rate:          TfExpressionEx  = 0.001,                    # Learning rate. Can vary over time.
        minibatch_multiplier:   TfExpressionEx  = None,                     # Treat N consecutive minibatches as one by accumulating gradients.
        share:                  "Optimizer"     = None,                     # Share internal state with a previously created optimizer?
        use_loss_scaling:       bool            = False,                    # Enable dynamic loss scaling for robust mixed-precision training?
        loss_scaling_init:      float           = 64.0,                     # Log2 of initial loss scaling factor.
        loss_scaling_inc:       float           = 0.0005,                   # Log2 of per-minibatch loss scaling increment when there is no overflow.
        loss_scaling_dec:       float           = 1.0,                      # Log2 of per-minibatch loss scaling decrement when there is an overflow.
        report_mem_usage:       bool            = False,                    # Report fine-grained memory usage statistics in TensorBoard?
        **kwargs):

        # Public fields.
        self.name                   = name
        self.learning_rate          = learning_rate
        self.minibatch_multiplier   = minibatch_multiplier
        self.id                     = self.name.replace("/", ".")
        self.scope                  = tf.get_default_graph().unique_name(self.id)
        self.optimizer_class        = util.get_obj_by_name(tf_optimizer)
        self.optimizer_kwargs       = dict(kwargs)
        self.use_loss_scaling       = use_loss_scaling
        self.loss_scaling_init      = loss_scaling_init
        self.loss_scaling_inc       = loss_scaling_inc
        self.loss_scaling_dec       = loss_scaling_dec

        # Private fields.
        self._updates_applied       = False
        self._devices               = OrderedDict() # device_name => EasyDict()
        self._shared_optimizers     = OrderedDict() # device_name => optimizer_class
        self._gradient_shapes       = None          # [shape, ...]
        self._report_mem_usage      = report_mem_usage

        # Validate arguments.
        assert callable(self.optimizer_class)

        # Share internal state if requested.
        if share is not None:
            assert isinstance(share, Optimizer)
            assert self.optimizer_class is share.optimizer_class
            assert self.learning_rate is share.learning_rate
            assert self.optimizer_kwargs == share.optimizer_kwargs
            self._shared_optimizers = share._shared_optimizers # pylint: disable=protected-access

    def _get_device(self, device_name: str):
        """Get internal state for the given TensorFlow device."""
        tfutil.assert_tf_initialized()
        if device_name in self._devices:
            return self._devices[device_name]

        # Initialize fields.
        device = util.EasyDict()
        device.name             = device_name
        device.optimizer        = None          # Underlying optimizer:     optimizer_class
        device.loss_scaling_var = None          # Log2 of loss scaling:     tf.Variable
        device.grad_raw         = OrderedDict() # Raw gradients:            var => [grad, ...]
        device.grad_clean       = OrderedDict() # Clean gradients:          var => grad
        device.grad_acc_vars    = OrderedDict() # Accumulation sums:        var => tf.Variable
        device.grad_acc_count   = None          # Accumulation counter:     tf.Variable
        device.grad_acc         = OrderedDict() # Accumulated gradients:    var => grad

        # Setup TensorFlow objects.
        with tfutil.absolute_name_scope(self.scope + "/Devices"), tf.device(device_name), tf.control_dependencies(None):
            if device_name not in self._shared_optimizers:
                optimizer_name = self.scope.replace("/", "_") + "_opt%d" % len(self._shared_optimizers)
                self._shared_optimizers[device_name] = self.optimizer_class(name=optimizer_name, learning_rate=self.learning_rate, **self.optimizer_kwargs)
            device.optimizer = self._shared_optimizers[device_name]
            if self.use_loss_scaling:
                device.loss_scaling_var = tf.Variable(np.float32(self.loss_scaling_init), trainable=False, name="loss_scaling_var")

        # Register device.
        self._devices[device_name] = device
        return device

    def register_gradients(self, loss: TfExpression, trainable_vars: Union[List, dict]) -> None:
        """Register the gradients of the given loss function with respect to the given variables.
        Intended to be called once per GPU."""
        tfutil.assert_tf_initialized()
        assert not self._updates_applied
        device = self._get_device(loss.device)

        # Validate trainables.
        if isinstance(trainable_vars, dict):
            trainable_vars = list(trainable_vars.values())  # allow passing in Network.trainables as vars
        assert isinstance(trainable_vars, list) and len(trainable_vars) >= 1
        assert all(tfutil.is_tf_expression(expr) for expr in trainable_vars + [loss])
        assert all(var.device == device.name for var in trainable_vars)

        # Validate shapes.
        if self._gradient_shapes is None:
            self._gradient_shapes = [var.shape.as_list() for var in trainable_vars]
        assert len(trainable_vars) == len(self._gradient_shapes)
        assert all(var.shape.as_list() == var_shape for var, var_shape in zip(trainable_vars, self._gradient_shapes))

        # Report memory usage if requested.
        deps = []
        if self._report_mem_usage:
            self._report_mem_usage = False
            try:
                with tf.name_scope(self.id + '_mem'), tf.device(device.name), tf.control_dependencies([loss]):
                    deps.append(autosummary.autosummary(self.id + "/mem_usage_gb", tf.contrib.memory_stats.BytesInUse() / 2**30))
            except tf.errors.NotFoundError:
                pass

        # Compute gradients.
        with tf.name_scope(self.id + "_grad"), tf.device(device.name), tf.control_dependencies(deps):
            loss = self.apply_loss_scaling(tf.cast(loss, tf.float32))
            gate = tf.train.Optimizer.GATE_NONE  # disable gating to reduce memory usage
            grad_list = device.optimizer.compute_gradients(loss=loss, var_list=trainable_vars, gate_gradients=gate)

        # Register gradients.
        for grad, var in grad_list:
            if var not in device.grad_raw:
                device.grad_raw[var] = []
            device.grad_raw[var].append(grad)

    def apply_updates(self, allow_no_op: bool = False) -> tf.Operation:
        """Construct training op to update the registered variables based on their gradients."""
        tfutil.assert_tf_initialized()
        assert not self._updates_applied
        self._updates_applied = True
        all_ops = []

        # Check for no-op.
        if allow_no_op and len(self._devices) == 0:
            with tfutil.absolute_name_scope(self.scope):
                return tf.no_op(name='TrainingOp')

        # Clean up gradients.
        for device_idx, device in enumerate(self._devices.values()):
            with tfutil.absolute_name_scope(self.scope + "/Clean%d" % device_idx), tf.device(device.name):
                for var, grad in device.grad_raw.items():

                    # Filter out disconnected gradients and convert to float32.
                    grad = [g for g in grad if g is not None]
                    grad = [tf.cast(g, tf.float32) for g in grad]

                    # Sum within the device.
                    if len(grad) == 0:
                        grad = tf.zeros(var.shape)  # No gradients => zero.
                    elif len(grad) == 1:
                        grad = grad[0]              # Single gradient => use as is.
                    else:
                        grad = tf.add_n(grad)       # Multiple gradients => sum.

                    # Scale as needed.
                    scale = 1.0 / len(device.grad_raw[var]) / len(self._devices)
                    scale = tf.constant(scale, dtype=tf.float32, name="scale")
                    if self.minibatch_multiplier is not None:
                        scale /= tf.cast(self.minibatch_multiplier, tf.float32)
                    scale = self.undo_loss_scaling(scale)
                    device.grad_clean[var] = grad * scale

        # Sum gradients across devices.
        if len(self._devices) > 1:
            with tfutil.absolute_name_scope(self.scope + "/Broadcast"), tf.device(None):
                for all_vars in zip(*[device.grad_clean.keys() for device in self._devices.values()]):
                    if len(all_vars) > 0 and all(dim > 0 for dim in all_vars[0].shape.as_list()): # NCCL does not support zero-sized tensors.
                        all_grads = [device.grad_clean[var] for device, var in zip(self._devices.values(), all_vars)]
                        all_grads = nccl_ops.all_sum(all_grads)
                        for device, var, grad in zip(self._devices.values(), all_vars, all_grads):
                            device.grad_clean[var] = grad

        # Apply updates separately on each device.
        for device_idx, device in enumerate(self._devices.values()):
            with tfutil.absolute_name_scope(self.scope + "/Apply%d" % device_idx), tf.device(device.name):
                # pylint: disable=cell-var-from-loop

                # Accumulate gradients over time.
                if self.minibatch_multiplier is None:
                    acc_ok = tf.constant(True, name='acc_ok')
                    device.grad_acc = OrderedDict(device.grad_clean)
                else:
                    # Create variables.
                    with tf.control_dependencies(None):
                        for var in device.grad_clean.keys():
                            device.grad_acc_vars[var] = tf.Variable(tf.zeros(var.shape), trainable=False, name="grad_acc_var")
                        device.grad_acc_count = tf.Variable(tf.zeros([]), trainable=False, name="grad_acc_count")

                    # Track counter.
                    count_cur = device.grad_acc_count + 1.0
                    count_inc_op = lambda: tf.assign(device.grad_acc_count, count_cur)
                    count_reset_op = lambda: tf.assign(device.grad_acc_count, tf.zeros([]))
                    acc_ok = (count_cur >= tf.cast(self.minibatch_multiplier, tf.float32))
                    all_ops.append(tf.cond(acc_ok, count_reset_op, count_inc_op))

                    # Track gradients.
                    for var, grad in device.grad_clean.items():
                        acc_var = device.grad_acc_vars[var]
                        acc_cur = acc_var + grad
                        device.grad_acc[var] = acc_cur
                        with tf.control_dependencies([acc_cur]):
                            acc_inc_op = lambda: tf.assign(acc_var, acc_cur)
                            acc_reset_op = lambda: tf.assign(acc_var, tf.zeros(var.shape))
                            all_ops.append(tf.cond(acc_ok, acc_reset_op, acc_inc_op))

                # No overflow => apply gradients.
                all_ok = tf.reduce_all(tf.stack([acc_ok] + [tf.reduce_all(tf.is_finite(g)) for g in device.grad_acc.values()]))
                apply_op = lambda: device.optimizer.apply_gradients([(tf.cast(grad, var.dtype), var) for var, grad in device.grad_acc.items()])
                all_ops.append(tf.cond(all_ok, apply_op, tf.no_op))

                # Adjust loss scaling.
                if self.use_loss_scaling:
                    ls_inc_op = lambda: tf.assign_add(device.loss_scaling_var, self.loss_scaling_inc)
                    ls_dec_op = lambda: tf.assign_sub(device.loss_scaling_var, self.loss_scaling_dec)
                    ls_update_op = lambda: tf.group(tf.cond(all_ok, ls_inc_op, ls_dec_op))
                    all_ops.append(tf.cond(acc_ok, ls_update_op, tf.no_op))

                # Last device => report statistics.
                if device_idx == len(self._devices) - 1:
                    all_ops.append(autosummary.autosummary(self.id + "/learning_rate", self.learning_rate))
                    all_ops.append(autosummary.autosummary(self.id + "/overflow_frequency", tf.where(all_ok, 0, 1), condition=acc_ok))
                    if self.use_loss_scaling:
                        all_ops.append(autosummary.autosummary(self.id + "/loss_scaling_log2", device.loss_scaling_var))

        # Initialize variables.
        self.reset_optimizer_state()
        if self.use_loss_scaling:
            tfutil.init_uninitialized_vars([device.loss_scaling_var for device in self._devices.values()])
        if self.minibatch_multiplier is not None:
            tfutil.run([var.initializer for device in self._devices.values() for var in list(device.grad_acc_vars.values()) + [device.grad_acc_count]])

        # Group everything into a single op.
        with tfutil.absolute_name_scope(self.scope):
            return tf.group(*all_ops, name="TrainingOp")

    def reset_optimizer_state(self) -> None:
        """Reset internal state of the underlying optimizer."""
        tfutil.assert_tf_initialized()
        tfutil.run([var.initializer for device in self._devices.values() for var in device.optimizer.variables()])

    def get_loss_scaling_var(self, device: str) -> Union[tf.Variable, None]:
        """Get or create variable representing log2 of the current dynamic loss scaling factor."""
        return self._get_device(device).loss_scaling_var

    def apply_loss_scaling(self, value: TfExpression) -> TfExpression:
        """Apply dynamic loss scaling for the given expression."""
        assert tfutil.is_tf_expression(value)
        if not self.use_loss_scaling:
            return value
        return value * tfutil.exp2(self.get_loss_scaling_var(value.device))

    def undo_loss_scaling(self, value: TfExpression) -> TfExpression:
        """Undo the effect of dynamic loss scaling for the given expression."""
        assert tfutil.is_tf_expression(value)
        if not self.use_loss_scaling:
            return value
        return value * tfutil.exp2(-self.get_loss_scaling_var(value.device)) # pylint: disable=invalid-unary-operand-type


class SimpleAdam:
    """Simplified version of tf.train.AdamOptimizer that behaves identically when used with dnnlib.tflib.Optimizer."""

    def __init__(self, name="Adam", learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.name = name
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.all_state_vars = []

    def variables(self):
        return self.all_state_vars

    def compute_gradients(self, loss, var_list, gate_gradients=tf.train.Optimizer.GATE_NONE):
        assert gate_gradients == tf.train.Optimizer.GATE_NONE
        return list(zip(tf.gradients(loss, var_list), var_list))

    def apply_gradients(self, grads_and_vars):
        with tf.name_scope(self.name):
            state_vars = []
            update_ops = []

            # Adjust learning rate to deal with startup bias.
            with tf.control_dependencies(None):
                b1pow_var = tf.Variable(dtype=tf.float32, initial_value=1, trainable=False)
                b2pow_var = tf.Variable(dtype=tf.float32, initial_value=1, trainable=False)
                state_vars += [b1pow_var, b2pow_var]
            b1pow_new = b1pow_var * self.beta1
            b2pow_new = b2pow_var * self.beta2
            update_ops += [tf.assign(b1pow_var, b1pow_new), tf.assign(b2pow_var, b2pow_new)]
            lr_new = self.learning_rate * tf.sqrt(1 - b2pow_new) / (1 - b1pow_new)

            # Construct ops to update each variable.
            for grad, var in grads_and_vars:
                with tf.control_dependencies(None):
                    m_var = tf.Variable(dtype=tf.float32, initial_value=tf.zeros_like(var), trainable=False)
                    v_var = tf.Variable(dtype=tf.float32, initial_value=tf.zeros_like(var), trainable=False)
                    state_vars += [m_var, v_var]
                m_new = self.beta1 * m_var + (1 - self.beta1) * grad
                v_new = self.beta2 * v_var + (1 - self.beta2) * tf.square(grad)
                var_delta = lr_new * m_new / (tf.sqrt(v_new) + self.epsilon)
                update_ops += [tf.assign(m_var, m_new), tf.assign(v_var, v_new), tf.assign_sub(var, var_delta)]

            # Group everything together.
            self.all_state_vars += state_vars
            return tf.group(*update_ops)
