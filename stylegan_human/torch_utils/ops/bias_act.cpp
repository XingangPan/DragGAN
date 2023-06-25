// Copyright (c) SenseTime Research. All rights reserved.

// Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "bias_act.h"

//------------------------------------------------------------------------

static bool has_same_layout(torch::Tensor x, torch::Tensor y)
{
    if (x.dim() != y.dim())
        return false;
    for (int64_t i = 0; i < x.dim(); i++)
    {
        if (x.size(i) != y.size(i))
            return false;
        if (x.size(i) >= 2 && x.stride(i) != y.stride(i))
            return false;
    }
    return true;
}

//------------------------------------------------------------------------

static torch::Tensor bias_act(torch::Tensor x, torch::Tensor b, torch::Tensor xref, torch::Tensor yref, torch::Tensor dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(b.numel() == 0 || (b.dtype() == x.dtype() && b.device() == x.device()), "b must have the same dtype and device as x");
    TORCH_CHECK(xref.numel() == 0 || (xref.sizes() == x.sizes() && xref.dtype() == x.dtype() && xref.device() == x.device()), "xref must have the same shape, dtype, and device as x");
    TORCH_CHECK(yref.numel() == 0 || (yref.sizes() == x.sizes() && yref.dtype() == x.dtype() && yref.device() == x.device()), "yref must have the same shape, dtype, and device as x");
    TORCH_CHECK(dy.numel() == 0 || (dy.sizes() == x.sizes() && dy.dtype() == x.dtype() && dy.device() == x.device()), "dy must have the same dtype and device as x");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(b.dim() == 1, "b must have rank 1");
    TORCH_CHECK(b.numel() == 0 || (dim >= 0 && dim < x.dim()), "dim is out of bounds");
    TORCH_CHECK(b.numel() == 0 || b.numel() == x.size(dim), "b has wrong number of elements");
    TORCH_CHECK(grad >= 0, "grad must be non-negative");

    // Validate layout.
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be non-overlapping and dense");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(xref.numel() == 0 || has_same_layout(xref, x), "xref must have the same layout as x");
    TORCH_CHECK(yref.numel() == 0 || has_same_layout(yref, x), "yref must have the same layout as x");
    TORCH_CHECK(dy.numel() == 0 || has_same_layout(dy, x), "dy must have the same layout as x");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    torch::Tensor y = torch::empty_like(x);
    TORCH_CHECK(has_same_layout(y, x), "y must have the same layout as x");

    // Initialize CUDA kernel parameters.
    bias_act_kernel_params p;
    p.x     = x.data_ptr();
    p.b     = (b.numel()) ? b.data_ptr() : NULL;
    p.xref  = (xref.numel()) ? xref.data_ptr() : NULL;
    p.yref  = (yref.numel()) ? yref.data_ptr() : NULL;
    p.dy    = (dy.numel()) ? dy.data_ptr() : NULL;
    p.y     = y.data_ptr();
    p.grad  = grad;
    p.act   = act;
    p.alpha = alpha;
    p.gain  = gain;
    p.clamp = clamp;
    p.sizeX = (int)x.numel();
    p.sizeB = (int)b.numel();
    p.stepB = (b.numel()) ? (int)x.stride(dim) : 1;

    // Choose CUDA kernel.
    void* kernel;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        kernel = choose_bias_act_kernel<scalar_t>(p);
    });
    TORCH_CHECK(kernel, "no CUDA kernel found for the specified activation func");

    // Launch CUDA kernel.
    p.loopX = 4;
    int blockSize = 4 * 32;
    int gridSize = (p.sizeX - 1) / (p.loopX * blockSize) + 1;
    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bias_act", &bias_act);
}

//------------------------------------------------------------------------
