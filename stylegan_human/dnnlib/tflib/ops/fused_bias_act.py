# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Custom TensorFlow ops for efficient bias and activation."""

import os
import numpy as np
import tensorflow as tf
from .. import custom_ops
from ...util import EasyDict

def _get_plugin():
    return custom_ops.get_plugin(os.path.splitext(__file__)[0] + '.cu')

#----------------------------------------------------------------------------

activation_funcs = {
    'linear':   EasyDict(func=lambda x, **_:        x,                          def_alpha=None, def_gain=1.0,           cuda_idx=1, ref='y', zero_2nd_grad=True),
    'relu':     EasyDict(func=lambda x, **_:        tf.nn.relu(x),              def_alpha=None, def_gain=np.sqrt(2),    cuda_idx=2, ref='y', zero_2nd_grad=True),
    'lrelu':    EasyDict(func=lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha), def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', zero_2nd_grad=True),
    'tanh':     EasyDict(func=lambda x, **_:        tf.nn.tanh(x),              def_alpha=None, def_gain=1.0,           cuda_idx=4, ref='y', zero_2nd_grad=False),
    'sigmoid':  EasyDict(func=lambda x, **_:        tf.nn.sigmoid(x),           def_alpha=None, def_gain=1.0,           cuda_idx=5, ref='y', zero_2nd_grad=False),
    'elu':      EasyDict(func=lambda x, **_:        tf.nn.elu(x),               def_alpha=None, def_gain=1.0,           cuda_idx=6, ref='y', zero_2nd_grad=False),
    'selu':     EasyDict(func=lambda x, **_:        tf.nn.selu(x),              def_alpha=None, def_gain=1.0,           cuda_idx=7, ref='y', zero_2nd_grad=False),
    'softplus': EasyDict(func=lambda x, **_:        tf.nn.softplus(x),          def_alpha=None, def_gain=1.0,           cuda_idx=8, ref='y', zero_2nd_grad=False),
    'swish':    EasyDict(func=lambda x, **_:        tf.nn.sigmoid(x) * x,       def_alpha=None, def_gain=np.sqrt(2),    cuda_idx=9, ref='x', zero_2nd_grad=False),
}

#----------------------------------------------------------------------------

def fused_bias_act(x, b=None, axis=1, act='linear', alpha=None, gain=None, impl='cuda'):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can have any shape, but if `b` is defined, the
                dimension corresponding to `axis`, as well as the rank, must be known.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `axis`.
        axis:   The dimension in `x` corresponding to the elements of `b`.
                The value of `axis` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying `1.0`.
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """

    impl_dict = {
        'ref':  _fused_bias_act_ref,
        'cuda': _fused_bias_act_cuda,
    }
    return impl_dict[impl](x=x, b=b, axis=axis, act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------

def _fused_bias_act_ref(x, b, axis, act, alpha, gain):
    """Slow reference implementation of `fused_bias_act()` using standard TensorFlow ops."""

    # Validate arguments.
    x = tf.convert_to_tensor(x)
    b = tf.convert_to_tensor(b) if b is not None else tf.constant([], dtype=x.dtype)
    act_spec = activation_funcs[act]
    assert b.shape.rank == 1 and (b.shape[0] == 0 or b.shape[0] == x.shape[axis])
    assert b.shape[0] == 0 or 0 <= axis < x.shape.rank
    if alpha is None:
        alpha = act_spec.def_alpha
    if gain is None:
        gain = act_spec.def_gain

    # Add bias.
    if b.shape[0] != 0:
        x += tf.reshape(b, [-1 if i == axis else 1 for i in range(x.shape.rank)])

    # Evaluate activation function.
    x = act_spec.func(x, alpha=alpha)

    # Scale by gain.
    if gain != 1:
        x *= gain
    return x

#----------------------------------------------------------------------------

def _fused_bias_act_cuda(x, b, axis, act, alpha, gain):
    """Fast CUDA implementation of `fused_bias_act()` using custom ops."""

    # Validate arguments.
    x = tf.convert_to_tensor(x)
    empty_tensor = tf.constant([], dtype=x.dtype)
    b = tf.convert_to_tensor(b) if b is not None else empty_tensor
    act_spec = activation_funcs[act]
    assert b.shape.rank == 1 and (b.shape[0] == 0 or b.shape[0] == x.shape[axis])
    assert b.shape[0] == 0 or 0 <= axis < x.shape.rank
    if alpha is None:
        alpha = act_spec.def_alpha
    if gain is None:
        gain = act_spec.def_gain

    # Special cases.
    if act == 'linear' and b is None and gain == 1.0:
        return x
    if act_spec.cuda_idx is None:
        return _fused_bias_act_ref(x=x, b=b, axis=axis, act=act, alpha=alpha, gain=gain)

    # CUDA kernel.
    cuda_kernel = _get_plugin().fused_bias_act
    cuda_kwargs = dict(axis=axis, act=act_spec.cuda_idx, alpha=alpha, gain=gain)

    # Forward pass: y = func(x, b).
    def func_y(x, b):
        y = cuda_kernel(x=x, b=b, ref=empty_tensor, grad=0, **cuda_kwargs)
        y.set_shape(x.shape)
        return y

    # Backward pass: dx, db = grad(dy, x, y)
    def grad_dx(dy, x, y):
        ref = {'x': x, 'y': y}[act_spec.ref]
        dx = cuda_kernel(x=dy, b=empty_tensor, ref=ref, grad=1, **cuda_kwargs)
        dx.set_shape(x.shape)
        return dx
    def grad_db(dx):
        if b.shape[0] == 0:
            return empty_tensor
        db = dx
        if axis < x.shape.rank - 1:
            db = tf.reduce_sum(db, list(range(axis + 1, x.shape.rank)))
        if axis > 0:
            db = tf.reduce_sum(db, list(range(axis)))
        db.set_shape(b.shape)
        return db

    # Second order gradients: d_dy, d_x = grad2(d_dx, d_db, x, y)
    def grad2_d_dy(d_dx, d_db, x, y):
        ref = {'x': x, 'y': y}[act_spec.ref]
        d_dy = cuda_kernel(x=d_dx, b=d_db, ref=ref, grad=1, **cuda_kwargs)
        d_dy.set_shape(x.shape)
        return d_dy
    def grad2_d_x(d_dx, d_db, x, y):
        ref = {'x': x, 'y': y}[act_spec.ref]
        d_x = cuda_kernel(x=d_dx, b=d_db, ref=ref, grad=2, **cuda_kwargs)
        d_x.set_shape(x.shape)
        return d_x

    # Fast version for piecewise-linear activation funcs.
    @tf.custom_gradient
    def func_zero_2nd_grad(x, b):
        y = func_y(x, b)
        @tf.custom_gradient
        def grad(dy):
            dx = grad_dx(dy, x, y)
            db = grad_db(dx)
            def grad2(d_dx, d_db):
                d_dy = grad2_d_dy(d_dx, d_db, x, y)
                return d_dy
            return (dx, db), grad2
        return y, grad

    # Slow version for general activation funcs.
    @tf.custom_gradient
    def func_nonzero_2nd_grad(x, b):
        y = func_y(x, b)
        def grad_wrap(dy):
            @tf.custom_gradient
            def grad_impl(dy, x):
                dx = grad_dx(dy, x, y)
                db = grad_db(dx)
                def grad2(d_dx, d_db):
                    d_dy = grad2_d_dy(d_dx, d_db, x, y)
                    d_x = grad2_d_x(d_dx, d_db, x, y)
                    return d_dy, d_x
                return (dx, db), grad2
            return grad_impl(dy, x)
        return y, grad_wrap

    # Which version to use?
    if act_spec.zero_2nd_grad:
        return func_zero_2nd_grad(x, b)
    return func_nonzero_2nd_grad(x, b)

#----------------------------------------------------------------------------
