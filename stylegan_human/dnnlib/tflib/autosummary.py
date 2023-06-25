# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Helper for adding automatically tracked values to Tensorboard.

Autosummary creates an identity op that internally keeps track of the input
values and automatically shows up in TensorBoard. The reported value
represents an average over input components. The average is accumulated
constantly over time and flushed when save_summaries() is called.

Notes:
- The output tensor must be used as an input for something else in the
  graph. Otherwise, the autosummary op will not get executed, and the average
  value will not get accumulated.
- It is perfectly fine to include autosummaries with the same name in
  several places throughout the graph, even if they are executed concurrently.
- It is ok to also pass in a python scalar or numpy array. In this case, it
  is added to the average immediately.
"""

from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2

from . import tfutil
from .tfutil import TfExpression
from .tfutil import TfExpressionEx

# Enable "Custom scalars" tab in TensorBoard for advanced formatting.
# Disabled by default to reduce tfevents file size.
enable_custom_scalars = False

_dtype = tf.float64
_vars = OrderedDict()  # name => [var, ...]
_immediate = OrderedDict()  # name => update_op, update_value
_finalized = False
_merge_op = None


def _create_var(name: str, value_expr: TfExpression) -> TfExpression:
    """Internal helper for creating autosummary accumulators."""
    assert not _finalized
    name_id = name.replace("/", "_")
    v = tf.cast(value_expr, _dtype)

    if v.shape.is_fully_defined():
        size = np.prod(v.shape.as_list())
        size_expr = tf.constant(size, dtype=_dtype)
    else:
        size = None
        size_expr = tf.reduce_prod(tf.cast(tf.shape(v), _dtype))

    if size == 1:
        if v.shape.ndims != 0:
            v = tf.reshape(v, [])
        v = [size_expr, v, tf.square(v)]
    else:
        v = [size_expr, tf.reduce_sum(v), tf.reduce_sum(tf.square(v))]
    v = tf.cond(tf.is_finite(v[1]), lambda: tf.stack(v), lambda: tf.zeros(3, dtype=_dtype))

    with tfutil.absolute_name_scope("Autosummary/" + name_id), tf.control_dependencies(None):
        var = tf.Variable(tf.zeros(3, dtype=_dtype), trainable=False)  # [sum(1), sum(x), sum(x**2)]
    update_op = tf.cond(tf.is_variable_initialized(var), lambda: tf.assign_add(var, v), lambda: tf.assign(var, v))

    if name in _vars:
        _vars[name].append(var)
    else:
        _vars[name] = [var]
    return update_op


def autosummary(name: str, value: TfExpressionEx, passthru: TfExpressionEx = None, condition: TfExpressionEx = True) -> TfExpressionEx:
    """Create a new autosummary.

    Args:
        name:     Name to use in TensorBoard
        value:    TensorFlow expression or python value to track
        passthru: Optionally return this TF node without modifications but tack an autosummary update side-effect to this node.

    Example use of the passthru mechanism:

    n = autosummary('l2loss', loss, passthru=n)

    This is a shorthand for the following code:

    with tf.control_dependencies([autosummary('l2loss', loss)]):
        n = tf.identity(n)
    """
    tfutil.assert_tf_initialized()
    name_id = name.replace("/", "_")

    if tfutil.is_tf_expression(value):
        with tf.name_scope("summary_" + name_id), tf.device(value.device):
            condition = tf.convert_to_tensor(condition, name='condition')
            update_op = tf.cond(condition, lambda: tf.group(_create_var(name, value)), tf.no_op)
            with tf.control_dependencies([update_op]):
                return tf.identity(value if passthru is None else passthru)

    else:  # python scalar or numpy array
        assert not tfutil.is_tf_expression(passthru)
        assert not tfutil.is_tf_expression(condition)
        if condition:
            if name not in _immediate:
                with tfutil.absolute_name_scope("Autosummary/" + name_id), tf.device(None), tf.control_dependencies(None):
                    update_value = tf.placeholder(_dtype)
                    update_op = _create_var(name, update_value)
                    _immediate[name] = update_op, update_value
            update_op, update_value = _immediate[name]
            tfutil.run(update_op, {update_value: value})
        return value if passthru is None else passthru


def finalize_autosummaries() -> None:
    """Create the necessary ops to include autosummaries in TensorBoard report.
    Note: This should be done only once per graph.
    """
    global _finalized
    tfutil.assert_tf_initialized()

    if _finalized:
        return None

    _finalized = True
    tfutil.init_uninitialized_vars([var for vars_list in _vars.values() for var in vars_list])

    # Create summary ops.
    with tf.device(None), tf.control_dependencies(None):
        for name, vars_list in _vars.items():
            name_id = name.replace("/", "_")
            with tfutil.absolute_name_scope("Autosummary/" + name_id):
                moments = tf.add_n(vars_list)
                moments /= moments[0]
                with tf.control_dependencies([moments]):  # read before resetting
                    reset_ops = [tf.assign(var, tf.zeros(3, dtype=_dtype)) for var in vars_list]
                    with tf.name_scope(None), tf.control_dependencies(reset_ops):  # reset before reporting
                        mean = moments[1]
                        std = tf.sqrt(moments[2] - tf.square(moments[1]))
                        tf.summary.scalar(name, mean)
                        if enable_custom_scalars:
                            tf.summary.scalar("xCustomScalars/" + name + "/margin_lo", mean - std)
                            tf.summary.scalar("xCustomScalars/" + name + "/margin_hi", mean + std)

    # Setup layout for custom scalars.
    layout = None
    if enable_custom_scalars:
        cat_dict = OrderedDict()
        for series_name in sorted(_vars.keys()):
            p = series_name.split("/")
            cat = p[0] if len(p) >= 2 else ""
            chart = "/".join(p[1:-1]) if len(p) >= 3 else p[-1]
            if cat not in cat_dict:
                cat_dict[cat] = OrderedDict()
            if chart not in cat_dict[cat]:
                cat_dict[cat][chart] = []
            cat_dict[cat][chart].append(series_name)
        categories = []
        for cat_name, chart_dict in cat_dict.items():
            charts = []
            for chart_name, series_names in chart_dict.items():
                series = []
                for series_name in series_names:
                    series.append(layout_pb2.MarginChartContent.Series(
                        value=series_name,
                        lower="xCustomScalars/" + series_name + "/margin_lo",
                        upper="xCustomScalars/" + series_name + "/margin_hi"))
                margin = layout_pb2.MarginChartContent(series=series)
                charts.append(layout_pb2.Chart(title=chart_name, margin=margin))
            categories.append(layout_pb2.Category(title=cat_name, chart=charts))
        layout = summary_lib.custom_scalar_pb(layout_pb2.Layout(category=categories))
    return layout

def save_summaries(file_writer, global_step=None):
    """Call FileWriter.add_summary() with all summaries in the default graph,
    automatically finalizing and merging them on the first call.
    """
    global _merge_op
    tfutil.assert_tf_initialized()

    if _merge_op is None:
        layout = finalize_autosummaries()
        if layout is not None:
            file_writer.add_summary(layout)
        with tf.device(None), tf.control_dependencies(None):
            _merge_op = tf.summary.merge_all()

    file_writer.add_summary(_merge_op.eval(), global_step)
