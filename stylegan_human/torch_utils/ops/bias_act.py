# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient bias and activation."""

import os
import warnings
import numpy as np
import torch
import dnnlib
import traceback

from .. import custom_ops
from .. import misc

#----------------------------------------------------------------------------

activation_funcs = {
    'linear':   dnnlib.EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
    'relu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.relu(x),                def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=2, ref='y', has_2nd_grad=False),
    'lrelu':    dnnlib.EasyDict(func=lambda x, alpha, **_:  torch.nn.functional.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
    'tanh':     dnnlib.EasyDict(func=lambda x, **_:         torch.tanh(x),                              def_alpha=0,    def_gain=1,             cuda_idx=4, ref='y', has_2nd_grad=True),
    'sigmoid':  dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x),                           def_alpha=0,    def_gain=1,             cuda_idx=5, ref='y', has_2nd_grad=True),
    'elu':      dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.elu(x),                 def_alpha=0,    def_gain=1,             cuda_idx=6, ref='y', has_2nd_grad=True),
    'selu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.selu(x),                def_alpha=0,    def_gain=1,             cuda_idx=7, ref='y', has_2nd_grad=True),
    'softplus': dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.softplus(x),            def_alpha=0,    def_gain=1,             cuda_idx=8, ref='y', has_2nd_grad=True),
    'swish':    dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x) * x,                       def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=9, ref='x', has_2nd_grad=True),
}

#----------------------------------------------------------------------------

_inited = False
_plugin = None
_null_tensor = torch.empty([0])

def _init():
    global _inited, _plugin
    if not _inited:
        _inited = True
        sources = ['bias_act.cpp', 'bias_act.cu']
        sources = [os.path.join(os.path.dirname(__file__), s) for s in sources]
        try:
            _plugin = custom_ops.get_plugin('bias_act_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
        except:
            warnings.warn('Failed to build CUDA kernels for bias_act. Falling back to slow reference implementation. Details:\n\n' + traceback.format_exc())
    return _plugin is not None

#----------------------------------------------------------------------------

def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can be of any shape.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `dim`.
        dim:    The dimension in `x` corresponding to the elements of `b`.
                The value of `dim` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1.
        clamp:  Clamp the output values to `[-clamp, +clamp]`, or `None` to disable
                the clamping (default).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _bias_act_cuda(dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp).apply(x, b)
    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)

#----------------------------------------------------------------------------

@misc.profiled_function
def _bias_act_ref(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Slow reference implementation of `bias_act()` using standard TensorFlow ops.
    """
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp) # pylint: disable=invalid-unary-operand-type
    return x

#----------------------------------------------------------------------------

_bias_act_cuda_cache = dict()

def _bias_act_cuda(dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Fast CUDA implementation of `bias_act()` using custom ops.
    """
    # Parse arguments.
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Lookup from cache.
    key = (dim, act, alpha, gain, clamp)
    if key in _bias_act_cuda_cache:
        return _bias_act_cuda_cache[key]

    # Forward op.
    class BiasActCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, b): # pylint: disable=arguments-differ
            ctx.memory_format = torch.channels_last if x.ndim > 2 and x.stride()[1] == 1 else torch.contiguous_format
            x = x.contiguous(memory_format=ctx.memory_format)
            b = b.contiguous() if b is not None else _null_tensor
            y = x
            if act != 'linear' or gain != 1 or clamp >= 0 or b is not _null_tensor:
                y = _plugin.bias_act(x, b, _null_tensor, _null_tensor, _null_tensor, 0, dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(
                x if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor,
                b if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor,
                y if 'y' in spec.ref else _null_tensor)
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            dy = dy.contiguous(memory_format=ctx.memory_format)
            x, b, y = ctx.saved_tensors
            dx = None
            db = None

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                dx = dy
                if act != 'linear' or gain != 1 or clamp >= 0:
                    dx = BiasActCudaGrad.apply(dy, x, b, y)

            if ctx.needs_input_grad[1]:
                db = dx.sum([i for i in range(dx.ndim) if i != dim])

            return dx, db

    # Backward op.
    class BiasActCudaGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dy, x, b, y): # pylint: disable=arguments-differ
            ctx.memory_format = torch.channels_last if dy.ndim > 2 and dy.stride()[1] == 1 else torch.contiguous_format
            dx = _plugin.bias_act(dy, b, x, y, _null_tensor, 1, dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(
                dy if spec.has_2nd_grad else _null_tensor,
                x, b, y)
            return dx

        @staticmethod
        def backward(ctx, d_dx): # pylint: disable=arguments-differ
            d_dx = d_dx.contiguous(memory_format=ctx.memory_format)
            dy, x, b, y = ctx.saved_tensors
            d_dy = None
            d_x = None
            d_b = None
            d_y = None

            if ctx.needs_input_grad[0]:
                d_dy = BiasActCudaGrad.apply(d_dx, x, b, y)

            if spec.has_2nd_grad and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
                d_x = _plugin.bias_act(d_dx, b, x, y, dy, 2, dim, spec.cuda_idx, alpha, gain, clamp)

            if spec.has_2nd_grad and ctx.needs_input_grad[2]:
                d_b = d_x.sum([i for i in range(d_x.ndim) if i != dim])

            return d_dy, d_x, d_b, d_y

    # Add to cache.
    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda

#----------------------------------------------------------------------------
