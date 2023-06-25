// Copyright (c) SenseTime Research. All rights reserved.

// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#define EIGEN_USE_GPU
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

//------------------------------------------------------------------------
// Helpers.

#define OP_CHECK_CUDA_ERROR(CTX, CUDA_CALL) do { cudaError_t err = CUDA_CALL; OP_REQUIRES(CTX, err == cudaSuccess, errors::Internal(cudaGetErrorName(err))); } while (false)

static __host__ __device__ __forceinline__ int floorDiv(int a, int b)
{
    int c = a / b;
    if (c * b > a)
        c--;
    return c;
}

//------------------------------------------------------------------------
// CUDA kernel params.

template <class T>
struct UpFirDn2DKernelParams
{
    const T*    x;          // [majorDim, inH, inW, minorDim]
    const T*    k;          // [kernelH, kernelW]
    T*          y;          // [majorDim, outH, outW, minorDim]

    int         upx;
    int         upy;
    int         downx;
    int         downy;
    int         padx0;
    int         padx1;
    int         pady0;
    int         pady1;

    int         majorDim;
    int         inH;
    int         inW;
    int         minorDim;
    int         kernelH;
    int         kernelW;
    int         outH;
    int         outW;
    int         loopMajor;
    int         loopX;
};

//------------------------------------------------------------------------
// General CUDA implementation for large filter kernels.

template <class T>
static __global__ void UpFirDn2DKernel_large(const UpFirDn2DKernelParams<T> p)
{
    // Calculate thread index.
    int minorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = minorIdx / p.minorDim;
    minorIdx -= outY * p.minorDim;
    int outXBase = blockIdx.y * p.loopX * blockDim.y + threadIdx.y;
    int majorIdxBase = blockIdx.z * p.loopMajor;
    if (outXBase >= p.outW || outY >= p.outH || majorIdxBase >= p.majorDim)
        return;

    // Setup Y receptive field.
    int midY = outY * p.downy + p.upy - 1 - p.pady0;
    int inY = min(max(floorDiv(midY, p.upy), 0), p.inH);
    int h = min(max(floorDiv(midY + p.kernelH, p.upy), 0), p.inH) - inY;
    int kernelY = midY + p.kernelH - (inY + 1) * p.upy;

    // Loop over majorDim and outX.
    for (int loopMajor = 0, majorIdx = majorIdxBase; loopMajor < p.loopMajor && majorIdx < p.majorDim; loopMajor++, majorIdx++)
    for (int loopX = 0, outX = outXBase; loopX < p.loopX && outX < p.outW; loopX++, outX += blockDim.y)
    {
        // Setup X receptive field.
        int midX = outX * p.downx + p.upx - 1 - p.padx0;
        int inX = min(max(floorDiv(midX, p.upx), 0), p.inW);
        int w = min(max(floorDiv(midX + p.kernelW, p.upx), 0), p.inW) - inX;
        int kernelX = midX + p.kernelW - (inX + 1) * p.upx;

        // Initialize pointers.
        const T* xp = &p.x[((majorIdx * p.inH + inY) * p.inW + inX) * p.minorDim + minorIdx];
        const T* kp = &p.k[kernelY * p.kernelW + kernelX];
        int xpx = p.minorDim;
        int kpx = -p.upx;
        int xpy = p.inW * p.minorDim;
        int kpy = -p.upy * p.kernelW;

        // Inner loop.
        float v = 0.0f;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                v += (float)(*xp) * (float)(*kp);
                xp += xpx;
                kp += kpx;
            }
            xp += xpy - w * xpx;
            kp += kpy - w * kpx;
        }

        // Store result.
        p.y[((majorIdx * p.outH + outY) * p.outW + outX) * p.minorDim + minorIdx] = (T)v;
    }
}

//------------------------------------------------------------------------
// Specialized CUDA implementation for small filter kernels.

template <class T, int upx, int upy, int downx, int downy, int kernelW, int kernelH, int tileOutW, int tileOutH>
static __global__ void UpFirDn2DKernel_small(const UpFirDn2DKernelParams<T> p)
{
    //assert(kernelW % upx == 0);
    //assert(kernelH % upy == 0);
    const int tileInW = ((tileOutW - 1) * downx + kernelW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + kernelH - 1) / upy + 1;
    __shared__ volatile float sk[kernelH][kernelW];
    __shared__ volatile float sx[tileInH][tileInW];

    // Calculate tile index.
    int minorIdx = blockIdx.x;
    int tileOutY = minorIdx / p.minorDim;
    minorIdx -= tileOutY * p.minorDim;
    tileOutY *= tileOutH;
    int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
    int majorIdxBase = blockIdx.z * p.loopMajor;
    if (tileOutXBase >= p.outW | tileOutY >= p.outH | majorIdxBase >= p.majorDim)
        return;

    // Load filter kernel (flipped).
    for (int tapIdx = threadIdx.x; tapIdx < kernelH * kernelW; tapIdx += blockDim.x)
    {
        int ky = tapIdx / kernelW;
        int kx = tapIdx - ky * kernelW;
        float v = 0.0f;
        if (kx < p.kernelW & ky < p.kernelH)
            v = (float)p.k[(p.kernelH - 1 - ky) * p.kernelW + (p.kernelW - 1 - kx)];
        sk[ky][kx] = v;
    }

    // Loop over majorDim and outX.
    for (int loopMajor = 0, majorIdx = majorIdxBase; loopMajor < p.loopMajor & majorIdx < p.majorDim; loopMajor++, majorIdx++)
    for (int loopX = 0, tileOutX = tileOutXBase; loopX < p.loopX & tileOutX < p.outW; loopX++, tileOutX += tileOutW)
    {
        // Load input pixels.
        int tileMidX = tileOutX * downx + upx - 1 - p.padx0;
        int tileMidY = tileOutY * downy + upy - 1 - p.pady0;
        int tileInX = floorDiv(tileMidX, upx);
        int tileInY = floorDiv(tileMidY, upy);
        __syncthreads();
        for (int inIdx = threadIdx.x; inIdx < tileInH * tileInW; inIdx += blockDim.x)
        {
            int relInY = inIdx / tileInW;
            int relInX = inIdx - relInY * tileInW;
            int inX = relInX + tileInX;
            int inY = relInY + tileInY;
            float v = 0.0f;
            if (inX >= 0 & inY >= 0 & inX < p.inW & inY < p.inH)
                v = (float)p.x[((majorIdx * p.inH + inY) * p.inW + inX) * p.minorDim + minorIdx];
            sx[relInY][relInX] = v;
        }

        // Loop over output pixels.
        __syncthreads();
        for (int outIdx = threadIdx.x; outIdx < tileOutH * tileOutW; outIdx += blockDim.x)
        {
            int relOutY = outIdx / tileOutW;
            int relOutX = outIdx - relOutY * tileOutW;
            int outX = relOutX + tileOutX;
            int outY = relOutY + tileOutY;

            // Setup receptive field.
            int midX = tileMidX + relOutX * downx;
            int midY = tileMidY + relOutY * downy;
            int inX = floorDiv(midX, upx);
            int inY = floorDiv(midY, upy);
            int relInX = inX - tileInX;
            int relInY = inY - tileInY;
            int kernelX = (inX + 1) * upx - midX - 1; // flipped
            int kernelY = (inY + 1) * upy - midY - 1; // flipped

            // Inner loop.
            float v = 0.0f;
            #pragma unroll
            for (int y = 0; y < kernelH / upy; y++)
                #pragma unroll
                for (int x = 0; x < kernelW / upx; x++)
                    v += sx[relInY + y][relInX + x] * sk[kernelY + y * upy][kernelX + x * upx];

            // Store result.
            if (outX < p.outW & outY < p.outH)
                p.y[((majorIdx * p.outH + outY) * p.outW + outX) * p.minorDim + minorIdx] = (T)v;
        }
    }
}

//------------------------------------------------------------------------
// TensorFlow op.

template <class T>
struct UpFirDn2DOp : public OpKernel
{
    UpFirDn2DKernelParams<T> m_attribs;

    UpFirDn2DOp(OpKernelConstruction* ctx) : OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("upx", &m_attribs.upx));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("upy", &m_attribs.upy));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("downx", &m_attribs.downx));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("downy", &m_attribs.downy));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("padx0", &m_attribs.padx0));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("padx1", &m_attribs.padx1));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("pady0", &m_attribs.pady0));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("pady1", &m_attribs.pady1));
        OP_REQUIRES(ctx, m_attribs.upx >= 1 && m_attribs.upy >= 1, errors::InvalidArgument("upx and upy must be at least 1x1"));
        OP_REQUIRES(ctx, m_attribs.downx >= 1 && m_attribs.downy >= 1, errors::InvalidArgument("downx and downy must be at least 1x1"));
    }

    void Compute(OpKernelContext* ctx)
    {
        UpFirDn2DKernelParams<T> p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        const Tensor& x = ctx->input(0); // [majorDim, inH, inW, minorDim]
        const Tensor& k = ctx->input(1); // [kernelH, kernelW]
        p.x = x.flat<T>().data();
        p.k = k.flat<T>().data();
        OP_REQUIRES(ctx, x.dims() == 4, errors::InvalidArgument("input must have rank 4"));
        OP_REQUIRES(ctx, k.dims() == 2, errors::InvalidArgument("kernel must have rank 2"));
        OP_REQUIRES(ctx, x.NumElements() <= kint32max, errors::InvalidArgument("input too large"));
        OP_REQUIRES(ctx, k.NumElements() <= kint32max, errors::InvalidArgument("kernel too large"));

        p.majorDim  = (int)x.dim_size(0);
        p.inH       = (int)x.dim_size(1);
        p.inW       = (int)x.dim_size(2);
        p.minorDim  = (int)x.dim_size(3);
        p.kernelH   = (int)k.dim_size(0);
        p.kernelW   = (int)k.dim_size(1);
        OP_REQUIRES(ctx, p.kernelW >= 1 && p.kernelH >= 1, errors::InvalidArgument("kernel must be at least 1x1"));

        p.outW = (p.inW * p.upx + p.padx0 + p.padx1 - p.kernelW + p.downx) / p.downx;
        p.outH = (p.inH * p.upy + p.pady0 + p.pady1 - p.kernelH + p.downy) / p.downy;
        OP_REQUIRES(ctx, p.outW >= 1 && p.outH >= 1, errors::InvalidArgument("output must be at least 1x1"));

        Tensor* y = NULL; // [majorDim, outH, outW, minorDim]
        TensorShape ys;
        ys.AddDim(p.majorDim);
        ys.AddDim(p.outH);
        ys.AddDim(p.outW);
        ys.AddDim(p.minorDim);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ys, &y));
        p.y = y->flat<T>().data();
        OP_REQUIRES(ctx, y->NumElements() <= kint32max, errors::InvalidArgument("output too large"));

        // Choose CUDA kernel to use.
        void* cudaKernel = (void*)UpFirDn2DKernel_large<T>;
        int tileOutW = -1;
        int tileOutH = -1;
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 7 && p.kernelH <= 7) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 7,7, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 6 && p.kernelH <= 6) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 6,6, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 5 && p.kernelH <= 5) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 5,5, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 4 && p.kernelH <= 4) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 4,4, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 3 && p.kernelH <= 3) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 3,3, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 8 && p.kernelH <= 8) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 8,8, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 6 && p.kernelH <= 6) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 6,6, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 4 && p.kernelH <= 4) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 4,4, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 2 && p.kernelH <= 2) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 2,2, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 8 && p.kernelH <= 8) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 8,8, 32,8>;  tileOutW = 32; tileOutH = 8;  }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 6 && p.kernelH <= 6) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 6,6, 32,8>;  tileOutW = 32; tileOutH = 8;  }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 4 && p.kernelH <= 4) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 4,4, 32,8>;  tileOutW = 32; tileOutH = 8;  }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 2 && p.kernelH <= 2) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 2,2, 32,8>;  tileOutW = 32; tileOutH = 8;  }

        // Choose launch params.
        dim3 blockSize;
        dim3 gridSize;
        if (tileOutW > 0 && tileOutH > 0) // small
        {
            p.loopMajor = (p.majorDim - 1) / 16384 + 1;
            p.loopX = 1;
            blockSize = dim3(32 * 8, 1, 1);
            gridSize = dim3(((p.outH - 1) / tileOutH + 1) * p.minorDim, (p.outW - 1) / (p.loopX * tileOutW) + 1, (p.majorDim - 1) / p.loopMajor + 1);
        }
        else // large
        {
            p.loopMajor = (p.majorDim - 1) / 16384 + 1;
            p.loopX = 4;
            blockSize = dim3(4, 32, 1);
            gridSize = dim3((p.outH * p.minorDim - 1) / blockSize.x + 1, (p.outW - 1) / (p.loopX * blockSize.y) + 1, (p.majorDim - 1) / p.loopMajor + 1);
        }

        // Launch CUDA kernel.
        void* args[] = {&p};
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(cudaKernel, gridSize, blockSize, args, 0, stream));
    }
};

REGISTER_OP("UpFirDn2D")
    .Input      ("x: T")
    .Input      ("k: T")
    .Output     ("y: T")
    .Attr       ("T: {float, half}")
    .Attr       ("upx: int = 1")
    .Attr       ("upy: int = 1")
    .Attr       ("downx: int = 1")
    .Attr       ("downy: int = 1")
    .Attr       ("padx0: int = 0")
    .Attr       ("padx1: int = 0")
    .Attr       ("pady0: int = 0")
    .Attr       ("pady1: int = 0");
REGISTER_KERNEL_BUILDER(Name("UpFirDn2D").Device(DEVICE_GPU).TypeConstraint<float>("T"), UpFirDn2DOp<float>);
REGISTER_KERNEL_BUILDER(Name("UpFirDn2D").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), UpFirDn2DOp<Eigen::half>);

//------------------------------------------------------------------------
