# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Helper for managing networks."""

import types
import inspect
import re
import uuid
import sys
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from typing import Any, List, Tuple, Union

from . import tfutil
from .. import util

from .tfutil import TfExpression, TfExpressionEx

_import_handlers = []  # Custom import handlers for dealing with legacy data in pickle import.
_import_module_src = dict()  # Source code for temporary modules created during pickle import.


def import_handler(handler_func):
    """Function decorator for declaring custom import handlers."""
    _import_handlers.append(handler_func)
    return handler_func


class Network:
    """Generic network abstraction.

    Acts as a convenience wrapper for a parameterized network construction
    function, providing several utility methods and convenient access to
    the inputs/outputs/weights.

    Network objects can be safely pickled and unpickled for long-term
    archival purposes. The pickling works reliably as long as the underlying
    network construction function is defined in a standalone Python module
    that has no side effects or application-specific imports.

    Args:
        name: Network name. Used to select TensorFlow name and variable scopes.
        func_name: Fully qualified name of the underlying network construction function, or a top-level function object.
        static_kwargs: Keyword arguments to be passed in to the network construction function.

    Attributes:
        name: User-specified name, defaults to build func name if None.
        scope: Unique TensorFlow scope containing template graph and variables, derived from the user-specified name.
        static_kwargs: Arguments passed to the user-supplied build func.
        components: Container for sub-networks. Passed to the build func, and retained between calls.
        num_inputs: Number of input tensors.
        num_outputs: Number of output tensors.
        input_shapes: Input tensor shapes (NC or NCHW), including minibatch dimension.
        output_shapes: Output tensor shapes (NC or NCHW), including minibatch dimension.
        input_shape: Short-hand for input_shapes[0].
        output_shape: Short-hand for output_shapes[0].
        input_templates: Input placeholders in the template graph.
        output_templates: Output tensors in the template graph.
        input_names: Name string for each input.
        output_names: Name string for each output.
        own_vars: Variables defined by this network (local_name => var), excluding sub-networks.
        vars: All variables (local_name => var).
        trainables: All trainable variables (local_name => var).
        var_global_to_local: Mapping from variable global names to local names.
    """

    def __init__(self, name: str = None, func_name: Any = None, **static_kwargs):
        tfutil.assert_tf_initialized()
        assert isinstance(name, str) or name is None
        assert func_name is not None
        assert isinstance(func_name, str) or util.is_top_level_function(func_name)
        assert util.is_pickleable(static_kwargs)

        self._init_fields()
        self.name = name
        self.static_kwargs = util.EasyDict(static_kwargs)

        # Locate the user-specified network build function.
        if util.is_top_level_function(func_name):
            func_name = util.get_top_level_function_name(func_name)
        module, self._build_func_name = util.get_module_from_obj_name(func_name)
        self._build_func = util.get_obj_from_module(module, self._build_func_name)
        assert callable(self._build_func)

        # Dig up source code for the module containing the build function.
        self._build_module_src = _import_module_src.get(module, None)
        if self._build_module_src is None:
            self._build_module_src = inspect.getsource(module)

        # Init TensorFlow graph.
        self._init_graph()
        self.reset_own_vars()

    def _init_fields(self) -> None:
        self.name = None
        self.scope = None
        self.static_kwargs = util.EasyDict()
        self.components = util.EasyDict()
        self.num_inputs = 0
        self.num_outputs = 0
        self.input_shapes = [[]]
        self.output_shapes = [[]]
        self.input_shape = []
        self.output_shape = []
        self.input_templates = []
        self.output_templates = []
        self.input_names = []
        self.output_names = []
        self.own_vars = OrderedDict()
        self.vars = OrderedDict()
        self.trainables = OrderedDict()
        self.var_global_to_local = OrderedDict()

        self._build_func = None  # User-supplied build function that constructs the network.
        self._build_func_name = None  # Name of the build function.
        self._build_module_src = None  # Full source code of the module containing the build function.
        self._run_cache = dict()  # Cached graph data for Network.run().

    def _init_graph(self) -> None:
        # Collect inputs.
        self.input_names = []

        for param in inspect.signature(self._build_func).parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
                self.input_names.append(param.name)

        self.num_inputs = len(self.input_names)
        assert self.num_inputs >= 1

        # Choose name and scope.
        if self.name is None:
            self.name = self._build_func_name
        assert re.match("^[A-Za-z0-9_.\\-]*$", self.name)
        with tf.name_scope(None):
            self.scope = tf.get_default_graph().unique_name(self.name, mark_as_used=True)

        # Finalize build func kwargs.
        build_kwargs = dict(self.static_kwargs)
        build_kwargs["is_template_graph"] = True
        build_kwargs["components"] = self.components

        # Build template graph.
        with tfutil.absolute_variable_scope(self.scope, reuse=False), tfutil.absolute_name_scope(self.scope):  # ignore surrounding scopes
            assert tf.get_variable_scope().name == self.scope
            assert tf.get_default_graph().get_name_scope() == self.scope
            with tf.control_dependencies(None):  # ignore surrounding control dependencies
                self.input_templates = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                out_expr = self._build_func(*self.input_templates, **build_kwargs)

        # Collect outputs.
        assert tfutil.is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        self.output_templates = [out_expr] if tfutil.is_tf_expression(out_expr) else list(out_expr)
        self.num_outputs = len(self.output_templates)
        assert self.num_outputs >= 1
        assert all(tfutil.is_tf_expression(t) for t in self.output_templates)

        # Perform sanity checks.
        if any(t.shape.ndims is None for t in self.input_templates):
            raise ValueError("Network input shapes not defined. Please call x.set_shape() for each input.")
        if any(t.shape.ndims is None for t in self.output_templates):
            raise ValueError("Network output shapes not defined. Please call x.set_shape() where applicable.")
        if any(not isinstance(comp, Network) for comp in self.components.values()):
            raise ValueError("Components of a Network must be Networks themselves.")
        if len(self.components) != len(set(comp.name for comp in self.components.values())):
            raise ValueError("Components of a Network must have unique names.")

        # List inputs and outputs.
        self.input_shapes = [t.shape.as_list() for t in self.input_templates]
        self.output_shapes = [t.shape.as_list() for t in self.output_templates]
        self.input_shape = self.input_shapes[0]
        self.output_shape = self.output_shapes[0]
        self.output_names = [t.name.split("/")[-1].split(":")[0] for t in self.output_templates]

        # List variables.
        self.own_vars = OrderedDict((var.name[len(self.scope) + 1:].split(":")[0], var) for var in tf.global_variables(self.scope + "/"))
        self.vars = OrderedDict(self.own_vars)
        self.vars.update((comp.name + "/" + name, var) for comp in self.components.values() for name, var in comp.vars.items())
        self.trainables = OrderedDict((name, var) for name, var in self.vars.items() if var.trainable)
        self.var_global_to_local = OrderedDict((var.name.split(":")[0], name) for name, var in self.vars.items())

    def reset_own_vars(self) -> None:
        """Re-initialize all variables of this network, excluding sub-networks."""
        tfutil.run([var.initializer for var in self.own_vars.values()])

    def reset_vars(self) -> None:
        """Re-initialize all variables of this network, including sub-networks."""
        tfutil.run([var.initializer for var in self.vars.values()])

    def reset_trainables(self) -> None:
        """Re-initialize all trainable variables of this network, including sub-networks."""
        tfutil.run([var.initializer for var in self.trainables.values()])

    def get_output_for(self, *in_expr: TfExpression, return_as_list: bool = False, **dynamic_kwargs) -> Union[TfExpression, List[TfExpression]]:
        """Construct TensorFlow expression(s) for the output(s) of this network, given the input expression(s)."""
        assert len(in_expr) == self.num_inputs
        assert not all(expr is None for expr in in_expr)

        # Finalize build func kwargs.
        build_kwargs = dict(self.static_kwargs)
        build_kwargs.update(dynamic_kwargs)
        build_kwargs["is_template_graph"] = False
        build_kwargs["components"] = self.components

        # Build TensorFlow graph to evaluate the network.
        with tfutil.absolute_variable_scope(self.scope, reuse=True), tf.name_scope(self.name):
            assert tf.get_variable_scope().name == self.scope
            valid_inputs = [expr for expr in in_expr if expr is not None]
            final_inputs = []
            for expr, name, shape in zip(in_expr, self.input_names, self.input_shapes):
                if expr is not None:
                    expr = tf.identity(expr, name=name)
                else:
                    expr = tf.zeros([tf.shape(valid_inputs[0])[0]] + shape[1:], name=name)
                final_inputs.append(expr)
            out_expr = self._build_func(*final_inputs, **build_kwargs)

        # Propagate input shapes back to the user-specified expressions.
        for expr, final in zip(in_expr, final_inputs):
            if isinstance(expr, tf.Tensor):
                expr.set_shape(final.shape)

        # Express outputs in the desired format.
        assert tfutil.is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        if return_as_list:
            out_expr = [out_expr] if tfutil.is_tf_expression(out_expr) else list(out_expr)
        return out_expr

    def get_var_local_name(self, var_or_global_name: Union[TfExpression, str]) -> str:
        """Get the local name of a given variable, without any surrounding name scopes."""
        assert tfutil.is_tf_expression(var_or_global_name) or isinstance(var_or_global_name, str)
        global_name = var_or_global_name if isinstance(var_or_global_name, str) else var_or_global_name.name
        return self.var_global_to_local[global_name]

    def find_var(self, var_or_local_name: Union[TfExpression, str]) -> TfExpression:
        """Find variable by local or global name."""
        assert tfutil.is_tf_expression(var_or_local_name) or isinstance(var_or_local_name, str)
        return self.vars[var_or_local_name] if isinstance(var_or_local_name, str) else var_or_local_name

    def get_var(self, var_or_local_name: Union[TfExpression, str]) -> np.ndarray:
        """Get the value of a given variable as NumPy array.
        Note: This method is very inefficient -- prefer to use tflib.run(list_of_vars) whenever possible."""
        return self.find_var(var_or_local_name).eval()

    def set_var(self, var_or_local_name: Union[TfExpression, str], new_value: Union[int, float, np.ndarray]) -> None:
        """Set the value of a given variable based on the given NumPy array.
        Note: This method is very inefficient -- prefer to use tflib.set_vars() whenever possible."""
        tfutil.set_vars({self.find_var(var_or_local_name): new_value})

    def __getstate__(self) -> dict:
        """Pickle export."""
        state = dict()
        state["version"]            = 4
        state["name"]               = self.name
        state["static_kwargs"]      = dict(self.static_kwargs)
        state["components"]         = dict(self.components)
        state["build_module_src"]   = self._build_module_src
        state["build_func_name"]    = self._build_func_name
        state["variables"]          = list(zip(self.own_vars.keys(), tfutil.run(list(self.own_vars.values()))))
        return state

    def __setstate__(self, state: dict) -> None:
        """Pickle import."""
        # pylint: disable=attribute-defined-outside-init
        tfutil.assert_tf_initialized()
        self._init_fields()

        # Execute custom import handlers.
        for handler in _import_handlers:
            state = handler(state)

        # Set basic fields.
        assert state["version"] in [2, 3, 4]
        self.name = state["name"]
        self.static_kwargs = util.EasyDict(state["static_kwargs"])
        self.components = util.EasyDict(state.get("components", {}))
        self._build_module_src = state["build_module_src"]
        self._build_func_name = state["build_func_name"]

        # Create temporary module from the imported source code.
        module_name = "_tflib_network_import_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _import_module_src[module] = self._build_module_src
        exec(self._build_module_src, module.__dict__) # pylint: disable=exec-used

        # Locate network build function in the temporary module.
        self._build_func = util.get_obj_from_module(module, self._build_func_name)
        assert callable(self._build_func)

        # Init TensorFlow graph.
        self._init_graph()
        self.reset_own_vars()
        tfutil.set_vars({self.find_var(name): value for name, value in state["variables"]})

    def clone(self, name: str = None, **new_static_kwargs) -> "Network":
        """Create a clone of this network with its own copy of the variables."""
        # pylint: disable=protected-access
        net = object.__new__(Network)
        net._init_fields()
        net.name = name if name is not None else self.name
        net.static_kwargs = util.EasyDict(self.static_kwargs)
        net.static_kwargs.update(new_static_kwargs)
        net._build_module_src = self._build_module_src
        net._build_func_name = self._build_func_name
        net._build_func = self._build_func
        net._init_graph()
        net.copy_vars_from(self)
        return net

    def copy_own_vars_from(self, src_net: "Network") -> None:
        """Copy the values of all variables from the given network, excluding sub-networks."""
        names = [name for name in self.own_vars.keys() if name in src_net.own_vars]
        tfutil.set_vars(tfutil.run({self.vars[name]: src_net.vars[name] for name in names}))

    def copy_vars_from(self, src_net: "Network") -> None:
        """Copy the values of all variables from the given network, including sub-networks."""
        names = [name for name in self.vars.keys() if name in src_net.vars]
        tfutil.set_vars(tfutil.run({self.vars[name]: src_net.vars[name] for name in names}))

    def copy_trainables_from(self, src_net: "Network") -> None:
        """Copy the values of all trainable variables from the given network, including sub-networks."""
        names = [name for name in self.trainables.keys() if name in src_net.trainables]
        tfutil.set_vars(tfutil.run({self.vars[name]: src_net.vars[name] for name in names}))

    def convert(self, new_func_name: str, new_name: str = None, **new_static_kwargs) -> "Network":
        """Create new network with the given parameters, and copy all variables from this network."""
        if new_name is None:
            new_name = self.name
        static_kwargs = dict(self.static_kwargs)
        static_kwargs.update(new_static_kwargs)
        net = Network(name=new_name, func_name=new_func_name, **static_kwargs)
        net.copy_vars_from(self)
        return net

    def setup_as_moving_average_of(self, src_net: "Network", beta: TfExpressionEx = 0.99, beta_nontrainable: TfExpressionEx = 0.0) -> tf.Operation:
        """Construct a TensorFlow op that updates the variables of this network
        to be slightly closer to those of the given network."""
        with tfutil.absolute_name_scope(self.scope + "/_MovingAvg"):
            ops = []
            for name, var in self.vars.items():
                if name in src_net.vars:
                    cur_beta = beta if name in self.trainables else beta_nontrainable
                    new_value = tfutil.lerp(src_net.vars[name], var, cur_beta)
                    ops.append(var.assign(new_value))
            return tf.group(*ops)

    def run(self,
            *in_arrays: Tuple[Union[np.ndarray, None], ...],
            input_transform: dict = None,
            output_transform: dict = None,
            return_as_list: bool = False,
            print_progress: bool = False,
            minibatch_size: int = None,
            num_gpus: int = 1,
            assume_frozen: bool = False,
            **dynamic_kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]:
        """Run this network for the given NumPy array(s), and return the output(s) as NumPy array(s).

        Args:
            input_transform:    A dict specifying a custom transformation to be applied to the input tensor(s) before evaluating the network.
                                The dict must contain a 'func' field that points to a top-level function. The function is called with the input
                                TensorFlow expression(s) as positional arguments. Any remaining fields of the dict will be passed in as kwargs.
            output_transform:   A dict specifying a custom transformation to be applied to the output tensor(s) after evaluating the network.
                                The dict must contain a 'func' field that points to a top-level function. The function is called with the output
                                TensorFlow expression(s) as positional arguments. Any remaining fields of the dict will be passed in as kwargs.
            return_as_list:     True = return a list of NumPy arrays, False = return a single NumPy array, or a tuple if there are multiple outputs.
            print_progress:     Print progress to the console? Useful for very large input arrays.
            minibatch_size:     Maximum minibatch size to use, None = disable batching.
            num_gpus:           Number of GPUs to use.
            assume_frozen:      Improve multi-GPU performance by assuming that the trainable parameters will remain changed between calls.
            dynamic_kwargs:     Additional keyword arguments to be passed into the network build function.
        """
        assert len(in_arrays) == self.num_inputs
        assert not all(arr is None for arr in in_arrays)
        assert input_transform is None or util.is_top_level_function(input_transform["func"])
        assert output_transform is None or util.is_top_level_function(output_transform["func"])
        output_transform, dynamic_kwargs = _handle_legacy_output_transforms(output_transform, dynamic_kwargs)
        num_items = in_arrays[0].shape[0]
        if minibatch_size is None:
            minibatch_size = num_items

        # Construct unique hash key from all arguments that affect the TensorFlow graph.
        key = dict(input_transform=input_transform, output_transform=output_transform, num_gpus=num_gpus, assume_frozen=assume_frozen, dynamic_kwargs=dynamic_kwargs)
        def unwind_key(obj):
            if isinstance(obj, dict):
                return [(key, unwind_key(value)) for key, value in sorted(obj.items())]
            if callable(obj):
                return util.get_top_level_function_name(obj)
            return obj
        key = repr(unwind_key(key))

        # Build graph.
        if key not in self._run_cache:
            with tfutil.absolute_name_scope(self.scope + "/_Run"), tf.control_dependencies(None):
                with tf.device("/cpu:0"):
                    in_expr = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                    in_split = list(zip(*[tf.split(x, num_gpus) for x in in_expr]))

                out_split = []
                for gpu in range(num_gpus):
                    with tf.device("/gpu:%d" % gpu):
                        net_gpu = self.clone() if assume_frozen else self
                        in_gpu = in_split[gpu]

                        if input_transform is not None:
                            in_kwargs = dict(input_transform)
                            in_gpu = in_kwargs.pop("func")(*in_gpu, **in_kwargs)
                            in_gpu = [in_gpu] if tfutil.is_tf_expression(in_gpu) else list(in_gpu)

                        assert len(in_gpu) == self.num_inputs
                        out_gpu = net_gpu.get_output_for(*in_gpu, return_as_list=True, **dynamic_kwargs)

                        if output_transform is not None:
                            out_kwargs = dict(output_transform)
                            out_gpu = out_kwargs.pop("func")(*out_gpu, **out_kwargs)
                            out_gpu = [out_gpu] if tfutil.is_tf_expression(out_gpu) else list(out_gpu)

                        assert len(out_gpu) == self.num_outputs
                        out_split.append(out_gpu)

                with tf.device("/cpu:0"):
                    out_expr = [tf.concat(outputs, axis=0) for outputs in zip(*out_split)]
                    self._run_cache[key] = in_expr, out_expr

        # Run minibatches.
        in_expr, out_expr = self._run_cache[key]
        out_arrays = [np.empty([num_items] + expr.shape.as_list()[1:], expr.dtype.name) for expr in out_expr]

        for mb_begin in range(0, num_items, minibatch_size):
            if print_progress:
                print("\r%d / %d" % (mb_begin, num_items), end="")

            mb_end = min(mb_begin + minibatch_size, num_items)
            mb_num = mb_end - mb_begin
            mb_in = [src[mb_begin : mb_end] if src is not None else np.zeros([mb_num] + shape[1:]) for src, shape in zip(in_arrays, self.input_shapes)]
            mb_out = tf.get_default_session().run(out_expr, dict(zip(in_expr, mb_in)))

            for dst, src in zip(out_arrays, mb_out):
                dst[mb_begin: mb_end] = src

        # Done.
        if print_progress:
            print("\r%d / %d" % (num_items, num_items))

        if not return_as_list:
            out_arrays = out_arrays[0] if len(out_arrays) == 1 else tuple(out_arrays)
        return out_arrays

    def list_ops(self) -> List[TfExpression]:
        include_prefix = self.scope + "/"
        exclude_prefix = include_prefix + "_"
        ops = tf.get_default_graph().get_operations()
        ops = [op for op in ops if op.name.startswith(include_prefix)]
        ops = [op for op in ops if not op.name.startswith(exclude_prefix)]
        return ops

    def list_layers(self) -> List[Tuple[str, TfExpression, List[TfExpression]]]:
        """Returns a list of (layer_name, output_expr, trainable_vars) tuples corresponding to
        individual layers of the network. Mainly intended to be used for reporting."""
        layers = []

        def recurse(scope, parent_ops, parent_vars, level):
            # Ignore specific patterns.
            if any(p in scope for p in ["/Shape", "/strided_slice", "/Cast", "/concat", "/Assign"]):
                return

            # Filter ops and vars by scope.
            global_prefix = scope + "/"
            local_prefix = global_prefix[len(self.scope) + 1:]
            cur_ops = [op for op in parent_ops if op.name.startswith(global_prefix) or op.name == global_prefix[:-1]]
            cur_vars = [(name, var) for name, var in parent_vars if name.startswith(local_prefix) or name == local_prefix[:-1]]
            if not cur_ops and not cur_vars:
                return

            # Filter out all ops related to variables.
            for var in [op for op in cur_ops if op.type.startswith("Variable")]:
                var_prefix = var.name + "/"
                cur_ops = [op for op in cur_ops if not op.name.startswith(var_prefix)]

            # Scope does not contain ops as immediate children => recurse deeper.
            contains_direct_ops = any("/" not in op.name[len(global_prefix):] and op.type not in ["Identity", "Cast", "Transpose"] for op in cur_ops)
            if (level == 0 or not contains_direct_ops) and (len(cur_ops) + len(cur_vars)) > 1:
                visited = set()
                for rel_name in [op.name[len(global_prefix):] for op in cur_ops] + [name[len(local_prefix):] for name, _var in cur_vars]:
                    token = rel_name.split("/")[0]
                    if token not in visited:
                        recurse(global_prefix + token, cur_ops, cur_vars, level + 1)
                        visited.add(token)
                return

            # Report layer.
            layer_name = scope[len(self.scope) + 1:]
            layer_output = cur_ops[-1].outputs[0] if cur_ops else cur_vars[-1][1]
            layer_trainables = [var for _name, var in cur_vars if var.trainable]
            layers.append((layer_name, layer_output, layer_trainables))

        recurse(self.scope, self.list_ops(), list(self.vars.items()), 0)
        return layers

    def print_layers(self, title: str = None, hide_layers_with_no_params: bool = False) -> None:
        """Print a summary table of the network structure."""
        rows = [[title if title is not None else self.name, "Params", "OutputShape", "WeightShape"]]
        rows += [["---"] * 4]
        total_params = 0

        for layer_name, layer_output, layer_trainables in self.list_layers():
            num_params = sum(int(np.prod(var.shape.as_list())) for var in layer_trainables)
            weights = [var for var in layer_trainables if var.name.endswith("/weight:0")]
            weights.sort(key=lambda x: len(x.name))
            if len(weights) == 0 and len(layer_trainables) == 1:
                weights = layer_trainables
            total_params += num_params

            if not hide_layers_with_no_params or num_params != 0:
                num_params_str = str(num_params) if num_params > 0 else "-"
                output_shape_str = str(layer_output.shape)
                weight_shape_str = str(weights[0].shape) if len(weights) >= 1 else "-"
                rows += [[layer_name, num_params_str, output_shape_str, weight_shape_str]]

        rows += [["---"] * 4]
        rows += [["Total", str(total_params), "", ""]]

        widths = [max(len(cell) for cell in column) for column in zip(*rows)]
        print()
        for row in rows:
            print("  ".join(cell + " " * (width - len(cell)) for cell, width in zip(row, widths)))
        print()

    def setup_weight_histograms(self, title: str = None) -> None:
        """Construct summary ops to include histograms of all trainable parameters in TensorBoard."""
        if title is None:
            title = self.name

        with tf.name_scope(None), tf.device(None), tf.control_dependencies(None):
            for local_name, var in self.trainables.items():
                if "/" in local_name:
                    p = local_name.split("/")
                    name = title + "_" + p[-1] + "/" + "_".join(p[:-1])
                else:
                    name = title + "_toplevel/" + local_name

                tf.summary.histogram(name, var)

#----------------------------------------------------------------------------
# Backwards-compatible emulation of legacy output transformation in Network.run().

_print_legacy_warning = True

def _handle_legacy_output_transforms(output_transform, dynamic_kwargs):
    global _print_legacy_warning
    legacy_kwargs = ["out_mul", "out_add", "out_shrink", "out_dtype"]
    if not any(kwarg in dynamic_kwargs for kwarg in legacy_kwargs):
        return output_transform, dynamic_kwargs

    if _print_legacy_warning:
        _print_legacy_warning = False
        print()
        print("WARNING: Old-style output transformations in Network.run() are deprecated.")
        print("Consider using 'output_transform=dict(func=tflib.convert_images_to_uint8)'")
        print("instead of 'out_mul=127.5, out_add=127.5, out_dtype=np.uint8'.")
        print()
    assert output_transform is None

    new_kwargs = dict(dynamic_kwargs)
    new_transform = {kwarg: new_kwargs.pop(kwarg) for kwarg in legacy_kwargs if kwarg in dynamic_kwargs}
    new_transform["func"] = _legacy_output_transform_func
    return new_transform, new_kwargs

def _legacy_output_transform_func(*expr, out_mul=1.0, out_add=0.0, out_shrink=1, out_dtype=None):
    if out_mul != 1.0:
        expr = [x * out_mul for x in expr]

    if out_add != 0.0:
        expr = [x + out_add for x in expr]

    if out_shrink > 1:
        ksize = [1, 1, out_shrink, out_shrink]
        expr = [tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW") for x in expr]

    if out_dtype is not None:
        if tf.as_dtype(out_dtype).is_integer:
            expr = [tf.round(x) for x in expr]
        expr = [tf.saturate_cast(x, out_dtype) for x in expr]
    return expr
