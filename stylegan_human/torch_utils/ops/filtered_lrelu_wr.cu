// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "filtered_lrelu.cu"

// Template/kernel specializations for sign write mode.

// Full op, 32-bit indexing.
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int32_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int32_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);

// Full op, 64-bit indexing.
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int64_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int64_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);

// Activation/signs only for generic variant. 64-bit indexing.
template void* choose_filtered_lrelu_act_kernel<c10::Half, true, false>(void);
template void* choose_filtered_lrelu_act_kernel<float,     true, false>(void);
template void* choose_filtered_lrelu_act_kernel<double,    true, false>(void);

// Copy filters to constant memory.
template cudaError_t copy_filters<true, false>(cudaStream_t stream);