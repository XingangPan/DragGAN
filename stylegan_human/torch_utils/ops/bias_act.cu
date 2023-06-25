// Copyright (c) SenseTime Research. All rights reserved.

// Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <c10/util/Half.h>
#include "bias_act.h"

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

//------------------------------------------------------------------------
// CUDA kernel.

template <class T, int A>
__global__ void bias_act_kernel(bias_act_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    int G                 = p.grad;
    scalar_t alpha        = (scalar_t)p.alpha;
    scalar_t gain         = (scalar_t)p.gain;
    scalar_t clamp        = (scalar_t)p.clamp;
    scalar_t one          = (scalar_t)1;
    scalar_t two          = (scalar_t)2;
    scalar_t expRange     = (scalar_t)80;
    scalar_t halfExpRange = (scalar_t)40;
    scalar_t seluScale    = (scalar_t)1.0507009873554804934193349852946;
    scalar_t seluAlpha    = (scalar_t)1.6732632423543772848170429916717;

    // Loop over elements.
    int xi = blockIdx.x * p.loopX * blockDim.x + threadIdx.x;
    for (int loopIdx = 0; loopIdx < p.loopX && xi < p.sizeX; loopIdx++, xi += blockDim.x)
    {
        // Load.
        scalar_t x = (scalar_t)((const T*)p.x)[xi];
        scalar_t b = (p.b) ? (scalar_t)((const T*)p.b)[(xi / p.stepB) % p.sizeB] : 0;
        scalar_t xref = (p.xref) ? (scalar_t)((const T*)p.xref)[xi] : 0;
        scalar_t yref = (p.yref) ? (scalar_t)((const T*)p.yref)[xi] : 0;
        scalar_t dy = (p.dy) ? (scalar_t)((const T*)p.dy)[xi] : one;
        scalar_t yy = (gain != 0) ? yref / gain : 0;
        scalar_t y = 0;

        // Apply bias.
        ((G == 0) ? x : xref) += b;

        // linear
        if (A == 1)
        {
            if (G == 0) y = x;
            if (G == 1) y = x;
        }

        // relu
        if (A == 2)
        {
            if (G == 0) y = (x > 0) ? x : 0;
            if (G == 1) y = (yy > 0) ? x : 0;
        }

        // lrelu
        if (A == 3)
        {
            if (G == 0) y = (x > 0) ? x : x * alpha;
            if (G == 1) y = (yy > 0) ? x : x * alpha;
        }

        // tanh
        if (A == 4)
        {
            if (G == 0) { scalar_t c = exp(x); scalar_t d = one / c; y = (x < -expRange) ? -one : (x > expRange) ? one : (c - d) / (c + d); }
            if (G == 1) y = x * (one - yy * yy);
            if (G == 2) y = x * (one - yy * yy) * (-two * yy);
        }

        // sigmoid
        if (A == 5)
        {
            if (G == 0) y = (x < -expRange) ? 0 : one / (exp(-x) + one);
            if (G == 1) y = x * yy * (one - yy);
            if (G == 2) y = x * yy * (one - yy) * (one - two * yy);
        }

        // elu
        if (A == 6)
        {
            if (G == 0) y = (x >= 0) ? x : exp(x) - one;
            if (G == 1) y = (yy >= 0) ? x : x * (yy + one);
            if (G == 2) y = (yy >= 0) ? 0 : x * (yy + one);
        }

        // selu
        if (A == 7)
        {
            if (G == 0) y = (x >= 0) ? seluScale * x : (seluScale * seluAlpha) * (exp(x) - one);
            if (G == 1) y = (yy >= 0) ? x * seluScale : x * (yy + seluScale * seluAlpha);
            if (G == 2) y = (yy >= 0) ? 0 : x * (yy + seluScale * seluAlpha);
        }

        // softplus
        if (A == 8)
        {
            if (G == 0) y = (x > expRange) ? x : log(exp(x) + one);
            if (G == 1) y = x * (one - exp(-yy));
            if (G == 2) { scalar_t c = exp(-yy); y = x * c * (one - c); }
        }

        // swish
        if (A == 9)
        {
            if (G == 0)
                y = (x < -expRange) ? 0 : x / (exp(-x) + one);
            else
            {
                scalar_t c = exp(xref);
                scalar_t d = c + one;
                if (G == 1)
                    y = (xref > halfExpRange) ? x : x * c * (xref + d) / (d * d);
                else
                    y = (xref > halfExpRange) ? 0 : x * c * (xref * (two - d) + two * d) / (d * d * d);
                yref = (xref < -expRange) ? 0 : xref / (exp(-xref) + one) * gain;
            }
        }

        // Apply gain.
        y *= gain * dy;

        // Clamp.
        if (clamp >= 0)
        {
            if (G == 0)
                y = (y > -clamp & y < clamp) ? y : (y >= 0) ? clamp : -clamp;
            else
                y = (yref > -clamp & yref < clamp) ? y : 0;
        }

        // Store.
        ((T*)p.y)[xi] = (T)y;
    }
}

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p)
{
    if (p.act == 1) return (void*)bias_act_kernel<T, 1>;
    if (p.act == 2) return (void*)bias_act_kernel<T, 2>;
    if (p.act == 3) return (void*)bias_act_kernel<T, 3>;
    if (p.act == 4) return (void*)bias_act_kernel<T, 4>;
    if (p.act == 5) return (void*)bias_act_kernel<T, 5>;
    if (p.act == 6) return (void*)bias_act_kernel<T, 6>;
    if (p.act == 7) return (void*)bias_act_kernel<T, 7>;
    if (p.act == 8) return (void*)bias_act_kernel<T, 8>;
    if (p.act == 9) return (void*)bias_act_kernel<T, 9>;
    return NULL;
}

//------------------------------------------------------------------------
// Template specializations.

template void* choose_bias_act_kernel<double>       (const bias_act_kernel_params& p);
template void* choose_bias_act_kernel<float>        (const bias_act_kernel_params& p);
template void* choose_bias_act_kernel<c10::Half>    (const bias_act_kernel_params& p);

//------------------------------------------------------------------------
