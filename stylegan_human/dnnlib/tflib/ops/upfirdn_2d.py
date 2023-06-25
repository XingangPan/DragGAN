# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Custom TensorFlow ops for efficient resampling of 2D images."""

import os
import numpy as np
import tensorflow as tf
from .. import custom_ops

def _get_plugin():
    return custom_ops.get_plugin(os.path.splitext(__file__)[0] + '.cu')

#----------------------------------------------------------------------------

def upfirdn_2d(x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0, impl='cuda'):
    r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.

    Accepts a batch of 2D images of the shape `[majorDim, inH, inW, minorDim]`
    and performs the following operations for each image, batched across
    `majorDim` and `minorDim`:

    1. Pad the image with zeros by the specified number of pixels on each side
       (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
       corresponds to cropping the image.

    2. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).

    3. Convolve the image with the specified 2D FIR filter (`k`), shrinking the
       image so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by throwing away pixels (`downx`, `downy`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:      Input tensor of the shape `[majorDim, inH, inW, minorDim]`.
        k:      2D FIR filter of the shape `[firH, firW]`.
        upx:    Integer upsampling factor along the X-axis (default: 1).
        upy:    Integer upsampling factor along the Y-axis (default: 1).
        downx:  Integer downsampling factor along the X-axis (default: 1).
        downy:  Integer downsampling factor along the Y-axis (default: 1).
        padx0:  Number of pixels to pad on the left side (default: 0).
        padx1:  Number of pixels to pad on the right side (default: 0).
        pady0:  Number of pixels to pad on the top side (default: 0).
        pady1:  Number of pixels to pad on the bottom side (default: 0).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[majorDim, outH, outW, minorDim]`, and same datatype as `x`.
    """

    impl_dict = {
        'ref':  _upfirdn_2d_ref,
        'cuda': _upfirdn_2d_cuda,
    }
    return impl_dict[impl](x=x, k=k, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1)

#----------------------------------------------------------------------------

def _upfirdn_2d_ref(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    assert x.shape.rank == 4
    inH = x.shape[1].value
    inW = x.shape[2].value
    minorDim = _shape(x, 3)
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

    # Pad (crop if negative).
    x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [0, 0]])
    x = x[:, max(-pady0, 0) : x.shape[1].value - max(-pady1, 0), max(-padx0, 0) : x.shape[2].value - max(-padx1, 0), :]

    # Convolve with filter.
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1])
    w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
    x = tf.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1])
    x = tf.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, :]

#----------------------------------------------------------------------------

def _upfirdn_2d_cuda(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Fast CUDA implementation of `upfirdn_2d()` using custom ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    majorDim, inH, inW, minorDim = x.shape.as_list()
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    outW = (inW * upx + padx0 + padx1 - kernelW) // downx + 1
    outH = (inH * upy + pady0 + pady1 - kernelH) // downy + 1
    assert outW >= 1 and outH >= 1

    kc = tf.constant(k, dtype=x.dtype)
    gkc = tf.constant(k[::-1, ::-1], dtype=x.dtype)
    gpadx0 = kernelW - padx0 - 1
    gpady0 = kernelH - pady0 - 1
    gpadx1 = inW * upx - outW * downx + padx0 - upx + 1
    gpady1 = inH * upy - outH * downy + pady0 - upy + 1

    @tf.custom_gradient
    def func(x):
        y = _get_plugin().up_fir_dn2d(x=x, k=kc, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1)
        y.set_shape([majorDim, outH, outW, minorDim])
        @tf.custom_gradient
        def grad(dy):
            dx = _get_plugin().up_fir_dn2d(x=dy, k=gkc, upx=downx, upy=downy, downx=upx, downy=upy, padx0=gpadx0, padx1=gpadx1, pady0=gpady0, pady1=gpady1)
            dx.set_shape([majorDim, inH, inW, minorDim])
            return dx, func
        return y, grad
    return func(x)

#----------------------------------------------------------------------------

def filter_2d(x, k, gain=1, data_format='NCHW', impl='cuda'):
    r"""Filter a batch of 2D images with the given FIR filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and filters each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """

    k = _setup_kernel(k) * gain
    p = k.shape[0] - 1
    return _simple_upfirdn_2d(x, k, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def upsample_2d(x, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda'):
    r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded with
    zeros so that its shape is a multiple of the upsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to nearest-neighbor
                      upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or
        `[N, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    return _simple_upfirdn_2d(x, k, up=factor, pad0=(p+1)//2+factor-1, pad1=p//2, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def downsample_2d(x, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda'):
    r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded with
    zeros so that its shape is a multiple of the downsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return _simple_upfirdn_2d(x, k, down=factor, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def upsample_conv_2d(x, w, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda'):
    r"""Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`.
                      Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to nearest-neighbor
                      upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or
        `[N, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    w = tf.convert_to_tensor(w)
    assert w.shape.rank == 4
    convH = w.shape[0].value
    convW = w.shape[1].value
    inC = _shape(w, 2)
    outC = _shape(w, 3)
    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (convW - 1)

    # Determine data dimensions.
    if data_format == 'NCHW':
        stride = [1, 1, factor, factor]
        output_shape = [_shape(x, 0), outC, (_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW]
        num_groups = _shape(x, 1) // inC
    else:
        stride = [1, factor, factor, 1]
        output_shape = [_shape(x, 0), (_shape(x, 1) - 1) * factor + convH, (_shape(x, 2) - 1) * factor + convW, outC]
        num_groups = _shape(x, 3) // inC

    # Transpose weights.
    w = tf.reshape(w, [convH, convW, inC, num_groups, -1])
    w = tf.transpose(w[::-1, ::-1], [0, 1, 4, 3, 2])
    w = tf.reshape(w, [convH, convW, -1, num_groups * inC])

    # Execute.
    x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride, padding='VALID', data_format=data_format)
    return _simple_upfirdn_2d(x, k, pad0=(p+1)//2+factor-1, pad1=p//2+1, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def conv_downsample_2d(x, w, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda'):
    r"""Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`.
                      Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    w = tf.convert_to_tensor(w)
    convH, convW, _inC, _outC = w.shape.as_list()
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    if data_format == 'NCHW':
        s = [1, 1, factor, factor]
    else:
        s = [1, factor, factor, 1]
    x = _simple_upfirdn_2d(x, k, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)
    return tf.nn.conv2d(x, w, strides=s, padding='VALID', data_format=data_format)

#----------------------------------------------------------------------------
# Internal helper funcs.

def _shape(tf_expr, dim_idx):
    if tf_expr.shape.rank is not None:
        dim = tf_expr.shape[dim_idx].value
        if dim is not None:
            return dim
    return tf.shape(tf_expr)[dim_idx]

def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k

def _simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0, data_format='NCHW', impl='cuda'):
    assert data_format in ['NCHW', 'NHWC']
    assert x.shape.rank == 4
    y = x
    if data_format == 'NCHW':
        y = tf.reshape(y, [-1, _shape(y, 2), _shape(y, 3), 1])
    y = upfirdn_2d(y, k, upx=up, upy=up, downx=down, downy=down, padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1, impl=impl)
    if data_format == 'NCHW':
        y = tf.reshape(y, [-1, _shape(x, 1), _shape(y, 1), _shape(y, 2)])
    return y

#----------------------------------------------------------------------------
