// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "upfirdn2d.h"

//------------------------------------------------------------------------

static torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(f.device() == x.device(), "f must reside on the same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(f.numel() <= INT_MAX, "f is too large");
    TORCH_CHECK(x.numel() > 0, "x has zero size");
    TORCH_CHECK(f.numel() > 0, "f has zero size");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(f.dim() == 2, "f must be rank 2");
    TORCH_CHECK((x.size(0)-1)*x.stride(0) + (x.size(1)-1)*x.stride(1) + (x.size(2)-1)*x.stride(2) + (x.size(3)-1)*x.stride(3) <= INT_MAX, "x memory footprint is too large");
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");
    TORCH_CHECK(upx >= 1 && upy >= 1, "upsampling factor must be at least 1");
    TORCH_CHECK(downx >= 1 && downy >= 1, "downsampling factor must be at least 1");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int outW = ((int)x.size(3) * upx + padx0 + padx1 - (int)f.size(1) + downx) / downx;
    int outH = ((int)x.size(2) * upy + pady0 + pady1 - (int)f.size(0) + downy) / downy;
    TORCH_CHECK(outW >= 1 && outH >= 1, "output must be at least 1x1");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW}, x.options(), x.suggest_memory_format());
    TORCH_CHECK(y.numel() <= INT_MAX, "output is too large");
    TORCH_CHECK((y.size(0)-1)*y.stride(0) + (y.size(1)-1)*y.stride(1) + (y.size(2)-1)*y.stride(2) + (y.size(3)-1)*y.stride(3) <= INT_MAX, "output memory footprint is too large");

    // Initialize CUDA kernel parameters.
    upfirdn2d_kernel_params p;
    p.x             = x.data_ptr();
    p.f             = f.data_ptr<float>();
    p.y             = y.data_ptr();
    p.up            = make_int2(upx, upy);
    p.down          = make_int2(downx, downy);
    p.pad0          = make_int2(padx0, pady0);
    p.flip          = (flip) ? 1 : 0;
    p.gain          = gain;
    p.inSize        = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.inStride      = make_int4((int)x.stride(3), (int)x.stride(2), (int)x.stride(1), (int)x.stride(0));
    p.filterSize    = make_int2((int)f.size(1), (int)f.size(0));
    p.filterStride  = make_int2((int)f.stride(1), (int)f.stride(0));
    p.outSize       = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.outStride     = make_int4((int)y.stride(3), (int)y.stride(2), (int)y.stride(1), (int)y.stride(0));
    p.sizeMajor     = (p.inStride.z == 1) ? p.inSize.w : p.inSize.w * p.inSize.z;
    p.sizeMinor     = (p.inStride.z == 1) ? p.inSize.z : 1;

    // Choose CUDA kernel.
    upfirdn2d_kernel_spec spec;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        spec = choose_upfirdn2d_kernel<scalar_t>(p);
    });

    // Set looping options.
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor     = spec.loopMinor;
    p.loopX         = spec.loopX;
    p.launchMinor   = (p.sizeMinor - 1) / p.loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;

    // Compute grid size.
    dim3 blockSize, gridSize;
    if (spec.tileOutW < 0) // large
    {
        blockSize = dim3(4, 32, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / blockSize.x + 1) * p.launchMinor,
            (p.outSize.x - 1) / (blockSize.y * p.loopX) + 1,
            p.launchMajor);
    }
    else // small
    {
        blockSize = dim3(256, 1, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
            (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1,
            p.launchMajor);
    }

    // Launch CUDA kernel.
    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("upfirdn2d", &upfirdn2d);
}

//------------------------------------------------------------------------
