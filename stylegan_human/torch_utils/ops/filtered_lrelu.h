// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <cuda_runtime.h>

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct filtered_lrelu_kernel_params
{
    // These parameters decide which kernel to use.
    int             up;         // upsampling ratio (1, 2, 4)
    int             down;       // downsampling ratio (1, 2, 4)
    int2            fuShape;    // [size, 1] | [size, size]
    int2            fdShape;    // [size, 1] | [size, size]

    int             _dummy;     // Alignment.

    // Rest of the parameters.
    const void*     x;          // Input tensor.
    void*           y;          // Output tensor.
    const void*     b;          // Bias tensor.
    unsigned char*  s;          // Sign tensor in/out. NULL if unused.
    const float*    fu;         // Upsampling filter.
    const float*    fd;         // Downsampling filter.

    int2            pad0;       // Left/top padding.
    float           gain;       // Additional gain factor.
    float           slope;      // Leaky ReLU slope on negative side.
    float           clamp;      // Clamp after nonlinearity.
    int             flip;       // Filter kernel flip for gradient computation.

    int             tilesXdim;  // Original number of horizontal output tiles.
    int             tilesXrep;  // Number of horizontal tiles per CTA.
    int             blockZofs;  // Block z offset to support large minibatch, channel dimensions.

    int4            xShape;     // [width, height, channel, batch]
    int4            yShape;     // [width, height, channel, batch]
    int2            sShape;     // [width, height] - width is in bytes. Contiguous. Zeros if unused.
    int2            sOfs;       // [ofs_x, ofs_y] - offset between upsampled data and sign tensor.
    int             swLimit;    // Active width of sign tensor in bytes.

    longlong4       xStride;    // Strides of all tensors except signs, same component order as shapes.
    longlong4       yStride;    //
    int64_t         bStride;    //
    longlong3       fuStride;   //
    longlong3       fdStride;   //
};

struct filtered_lrelu_act_kernel_params
{
    void*           x;          // Input/output, modified in-place.
    unsigned char*  s;          // Sign tensor in/out. NULL if unused.

    float           gain;       // Additional gain factor.
    float           slope;      // Leaky ReLU slope on negative side.
    float           clamp;      // Clamp after nonlinearity.

    int4            xShape;     // [width, height, channel, batch]
    longlong4       xStride;    // Input/output tensor strides, same order as in shape.
    int2            sShape;     // [width, height] - width is in elements. Contiguous. Zeros if unused.
    int2            sOfs;       // [ofs_x, ofs_y] - offset between upsampled data and sign tensor.
};

//------------------------------------------------------------------------
// CUDA kernel specialization.

struct filtered_lrelu_kernel_spec
{
    void*   setup;              // Function for filter kernel setup.
    void*   exec;               // Function for main operation.
    int2    tileOut;            // Width/height of launch tile.
    int     numWarps;           // Number of warps per thread block, determines launch block size.
    int     xrep;               // For processing multiple horizontal tiles per thread block.
    int     dynamicSharedKB;    // How much dynamic shared memory the exec kernel wants.
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T, class index_t, bool signWrite, bool signRead> filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel(const filtered_lrelu_kernel_params& p, int sharedKB);
template <class T, bool signWrite, bool signRead> void* choose_filtered_lrelu_act_kernel(void);
template <bool signWrite, bool signRead> cudaError_t copy_filters(cudaStream_t stream);

//------------------------------------------------------------------------