# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Miscellaneous helper utils for Tensorflow."""

import os
import numpy as np
import tensorflow as tf

# Silence deprecation warnings from TensorFlow 1.13 onwards
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow.contrib   # requires TensorFlow 1.x!
tf.contrib = tensorflow.contrib

from typing import Any, Iterable, List, Union

TfExpression = Union[tf.Tensor, tf.Variable, tf.Operation]
"""A type that represents a valid Tensorflow expression."""

TfExpressionEx = Union[TfExpression, int, float, np.ndarray]
"""A type that can be converted to a valid Tensorflow expression."""


def run(*args, **kwargs) -> Any:
    """Run the specified ops in the default session."""
    assert_tf_initialized()
    return tf.get_default_session().run(*args, **kwargs)


def is_tf_expression(x: Any) -> bool:
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))


def shape_to_list(shape: Iterable[tf.Dimension]) -> List[Union[int, None]]:
    """Convert a Tensorflow shape to a list of ints. Retained for backwards compatibility -- use TensorShape.as_list() in new code."""
    return [dim.value for dim in shape]


def flatten(x: TfExpressionEx) -> TfExpression:
    """Shortcut function for flattening a tensor."""
    with tf.name_scope("Flatten"):
        return tf.reshape(x, [-1])


def log2(x: TfExpressionEx) -> TfExpression:
    """Logarithm in base 2."""
    with tf.name_scope("Log2"):
        return tf.log(x) * np.float32(1.0 / np.log(2.0))


def exp2(x: TfExpressionEx) -> TfExpression:
    """Exponent in base 2."""
    with tf.name_scope("Exp2"):
        return tf.exp(x * np.float32(np.log(2.0)))


def lerp(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpressionEx:
    """Linear interpolation."""
    with tf.name_scope("Lerp"):
        return a + (b - a) * t


def lerp_clip(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpression:
    """Linear interpolation with clip."""
    with tf.name_scope("LerpClip"):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def absolute_name_scope(scope: str) -> tf.name_scope:
    """Forcefully enter the specified name scope, ignoring any surrounding scopes."""
    return tf.name_scope(scope + "/")


def absolute_variable_scope(scope: str, **kwargs) -> tf.variable_scope:
    """Forcefully enter the specified variable scope, ignoring any surrounding scopes."""
    return tf.variable_scope(tf.VariableScope(name=scope, **kwargs), auxiliary_name_scope=False)


def _sanitize_tf_config(config_dict: dict = None) -> dict:
    # Defaults.
    cfg = dict()
    cfg["rnd.np_random_seed"]               = None      # Random seed for NumPy. None = keep as is.
    cfg["rnd.tf_random_seed"]               = "auto"    # Random seed for TensorFlow. 'auto' = derive from NumPy random state. None = keep as is.
    cfg["env.TF_CPP_MIN_LOG_LEVEL"]         = "1"       # 0 = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.
    cfg["graph_options.place_pruned_graph"] = True      # False = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
    cfg["gpu_options.allow_growth"]         = True      # False = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.

    # Remove defaults for environment variables that are already set.
    for key in list(cfg):
        fields = key.split(".")
        if fields[0] == "env":
            assert len(fields) == 2
            if fields[1] in os.environ:
                del cfg[key]

    # User overrides.
    if config_dict is not None:
        cfg.update(config_dict)
    return cfg


def init_tf(config_dict: dict = None) -> None:
    """Initialize TensorFlow session using good default settings."""
    # Skip if already initialized.
    if tf.get_default_session() is not None:
        return

    # Setup config dict and random seeds.
    cfg = _sanitize_tf_config(config_dict)
    np_random_seed = cfg["rnd.np_random_seed"]
    if np_random_seed is not None:
        np.random.seed(np_random_seed)
    tf_random_seed = cfg["rnd.tf_random_seed"]
    if tf_random_seed == "auto":
        tf_random_seed = np.random.randint(1 << 31)
    if tf_random_seed is not None:
        tf.set_random_seed(tf_random_seed)

    # Setup environment variables.
    for key, value in cfg.items():
        fields = key.split(".")
        if fields[0] == "env":
            assert len(fields) == 2
            os.environ[fields[1]] = str(value)

    # Create default TensorFlow session.
    create_session(cfg, force_as_default=True)


def assert_tf_initialized():
    """Check that TensorFlow session has been initialized."""
    if tf.get_default_session() is None:
        raise RuntimeError("No default TensorFlow session found. Please call dnnlib.tflib.init_tf().")


def create_session(config_dict: dict = None, force_as_default: bool = False) -> tf.Session:
    """Create tf.Session based on config dict."""
    # Setup TensorFlow config proto.
    cfg = _sanitize_tf_config(config_dict)
    config_proto = tf.ConfigProto()
    for key, value in cfg.items():
        fields = key.split(".")
        if fields[0] not in ["rnd", "env"]:
            obj = config_proto
            for field in fields[:-1]:
                obj = getattr(obj, field)
            setattr(obj, fields[-1], value)

    # Create session.
    session = tf.Session(config=config_proto)
    if force_as_default:
        # pylint: disable=protected-access
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()
    return session


def init_uninitialized_vars(target_vars: List[tf.Variable] = None) -> None:
    """Initialize all tf.Variables that have not already been initialized.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tf.variables_initializer(tf.report_uninitialized_variables()).run()
    """
    assert_tf_initialized()
    if target_vars is None:
        target_vars = tf.global_variables()

    test_vars = []
    test_ops = []

    with tf.control_dependencies(None):  # ignore surrounding control_dependencies
        for var in target_vars:
            assert is_tf_expression(var)

            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/IsVariableInitialized:0"))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)

                with absolute_name_scope(var.name.split(":")[0]):
                    test_ops.append(tf.is_variable_initialized(var))

    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars])


def set_vars(var_to_value_dict: dict) -> None:
    """Set the values of given tf.Variables.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tflib.run([tf.assign(var, value) for var, value in var_to_value_dict.items()]
    """
    assert_tf_initialized()
    ops = []
    feed_dict = {}

    for var, value in var_to_value_dict.items():
        assert is_tf_expression(var)

        try:
            setter = tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/setter:0"))  # look for existing op
        except KeyError:
            with absolute_name_scope(var.name.split(":")[0]):
                with tf.control_dependencies(None):  # ignore surrounding control_dependencies
                    setter = tf.assign(var, tf.placeholder(var.dtype, var.shape, "new_value"), name="setter")  # create new setter

        ops.append(setter)
        feed_dict[setter.op.inputs[1]] = value

    run(ops, feed_dict)


def create_var_with_large_initial_value(initial_value: np.ndarray, *args, **kwargs):
    """Create tf.Variable with large initial value without bloating the tf graph."""
    assert_tf_initialized()
    assert isinstance(initial_value, np.ndarray)
    zeros = tf.zeros(initial_value.shape, initial_value.dtype)
    var = tf.Variable(zeros, *args, **kwargs)
    set_vars({var: initial_value})
    return var


def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if nhwc_to_nchw:
        images = tf.transpose(images, [0, 3, 1, 2])
    return images * ((drange[1] - drange[0]) / 255) + drange[0]


def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False, shrink=1):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        ksize = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return tf.saturate_cast(images, tf.uint8)
