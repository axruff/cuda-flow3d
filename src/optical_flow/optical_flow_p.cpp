/**
* @file    3D Optical flow using NVIDIA CUDA
* @author  Institute for Photon Science and Synchrotron Radiation, Karlsruhe Institute of Technology
*
* @date    2015-2018
* @version 0.5.0
*
*
* @section LICENSE
*
* This program is copyrighted by the author and Institute for Photon Science and Synchrotron Radiation,
* Karlsruhe Institute of Technology, Karlsruhe, Germany;
*
*
*/

#include "src/optical_flow/optical_flow_p.h"

#include <algorithm>
#include <cstdio>

#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

OpticalFlowP::OpticalFlowP()
  : OpticalFlowBase("Optical Flow Single GPU Piecemeal Processing")
{
  cuda_operations_.push_front(&cuop_register_p_);
  cuda_operations_.push_front(&cuop_resample_p_);
  cuda_operations_.push_front(&cuop_solve_p_);
  cuda_operations_.push_front(&cuop_stat_p_);
  cuda_operations_.push_front(&cuop_add_p_);
}

bool OpticalFlowP::Initialize(const DataSize4& data_size)
{
  initialized_ = true;
  
  data_size_ = data_size;

  std::printf("Initialization of cuda operations...\n");

  for (CudaOperationBase* cuop : cuda_operations_) {
    std::printf("%-18s: ", cuop->GetName());
    bool result = cuop->Initialize();
    if (result) {
      std::printf("OK\n");
    } else {
      Destroy();
      initialized_ = false;
    }
  }  

  return initialized_;
}

void OpticalFlowP::ComputeFlow(Data3D& frame_0, Data3D& frame_1, Data3D& flow_u, Data3D& flow_v, Data3D&flow_w, OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  /* Optical flow algorithm's parameters */
  size_t  warp_levels_count;
  float   warp_scale_factor;
  size_t  outer_iterations_count;
  size_t  inner_iterations_count;
  float   equation_alpha;
  float   equation_smoothness;
  float   equation_data;
  size_t  median_radius;
  float   gaussian_sigma;

  /* Lambda function for correct working of GET_PARAM_OR_RETURN */
  GET_PARAM_OR_RETURN(params, size_t, warp_levels_count,      "warp_levels_count");
  GET_PARAM_OR_RETURN(params, float,  warp_scale_factor,      "warp_scale_factor");
  GET_PARAM_OR_RETURN(params, size_t, outer_iterations_count, "outer_iterations_count");
  GET_PARAM_OR_RETURN(params, size_t, inner_iterations_count, "inner_iterations_count");
  GET_PARAM_OR_RETURN(params, float,  equation_alpha,         "equation_alpha");
  GET_PARAM_OR_RETURN(params, float,  equation_smoothness,    "equation_smoothness");
  GET_PARAM_OR_RETURN(params, float,  equation_data,          "equation_data");
  GET_PARAM_OR_RETURN(params, size_t, median_radius,          "median_radius");
  GET_PARAM_OR_RETURN(params, float,  gaussian_sigma,         "gaussian_sigma");

  /* Auxiliary variables */
  float hx; // spacing in x-direction (current resol.)
  float hy; // spacing in y-direction (current resol.)
  float hz; // spacing in z-direction (current resol.)
  DataSize4 original_data_size = { frame_0.Width(), frame_0.Height(), frame_0.Depth(), 0 };
  DataSize4 current_data_size = { 0 };
  DataSize4 prev_data_size = { 0 };
  Stat3 flow_stat = { 0.f };
  
  size_t max_warp_level = GetMaxWarpLevel(original_data_size.width, original_data_size.height, original_data_size.depth, warp_scale_factor);
  int current_warp_level = std::min(warp_levels_count, max_warp_level) - 1;

  /* Allocate memory for additional buffers */
  std::printf("Allocating additional memory on the host...\n");
  std::printf("Total RAM memory usage: %.0fMB\n",
    (5 + 10) * // Number of Data3D containers: 5 - arguments; 10 - additional containers allocated below.
    (original_data_size.width * original_data_size.height * original_data_size.depth * sizeof(float)) /
    (1024.f * 1024.f)
    );

  Data3D frame_0_res(original_data_size.width, original_data_size.height, original_data_size.depth);
  Data3D frame_1_res_br(original_data_size.width, original_data_size.height, original_data_size.depth);
  
  Data3D flow_du(original_data_size.width, original_data_size.height, original_data_size.depth);
  Data3D flow_dv(original_data_size.width, original_data_size.height, original_data_size.depth);
  Data3D flow_dw(original_data_size.width, original_data_size.height, original_data_size.depth);

  Data3D phi(original_data_size.width, original_data_size.height, original_data_size.depth);
  Data3D ksi(original_data_size.width, original_data_size.height, original_data_size.depth);

  Data3D temp_0(original_data_size.width, original_data_size.height, original_data_size.depth);
  Data3D temp_1(original_data_size.width, original_data_size.height, original_data_size.depth);
  Data3D temp_2(original_data_size.width, original_data_size.height, original_data_size.depth);

  /* Use pointers for input frames to have a possibility to swap it */
  Data3D* p_frame_0 = &frame_0;
  Data3D* p_frame_1 = &frame_1;

  Data3D* p_frame_0_res = &frame_0_res;
  Data3D* p_frame_1_res_br = &frame_1_res_br;

  /* Create CUDA event for the time measure */
  CUevent cu_event_start;
  CUevent cu_event_stop;

  CheckCudaError(cuEventCreate(&cu_event_start, CU_EVENT_DEFAULT));
  CheckCudaError(cuEventCreate(&cu_event_stop, CU_EVENT_DEFAULT));

  CheckCudaError(cuEventRecord(cu_event_start, NULL));
  
  std::printf("\nStarting optical flow computation...\n");
  OperationParameters op;

  /* Main loop */
  while (current_warp_level >= 0) {
    float scale = std::pow(warp_scale_factor, static_cast<float>(current_warp_level));
    current_data_size.width = static_cast<size_t>(std::ceil(original_data_size.width * scale));
    current_data_size.height = static_cast<size_t>(std::ceil(original_data_size.height * scale));
    current_data_size.depth = static_cast<size_t>(std::ceil(original_data_size.depth * scale));
    hx = original_data_size.width / static_cast<float>(current_data_size.width);
    hy = original_data_size.height / static_cast<float>(current_data_size.height);
    hz = original_data_size.depth / static_cast<float>(current_data_size.depth);

    if (!silent)
        std::printf("Solve level %2d (%4d x%4d x%4d) \n", current_warp_level, current_data_size.width, current_data_size.height, current_data_size.depth);

    /* Data resampling */
    {
      if (current_warp_level == 0) {
        std::swap(p_frame_0,p_frame_0_res);
        std::swap(p_frame_1,p_frame_1_res_br);
      } else {
        op.Clear();
        op.PushValuePtr("input",         p_frame_0);
        op.PushValuePtr("output",        p_frame_0_res);
        op.PushValuePtr("data_size",     &original_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_p_.Execute(op);

        op.Clear();
        op.PushValuePtr("input",         p_frame_1);
        op.PushValuePtr("output",        p_frame_1_res_br);
        op.PushValuePtr("data_size",     &original_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_p_.Execute(op);
      }
    }

    /* Flow field resampling */
    {
      if (prev_data_size.width == 0) {
        flow_u.ZeroData();
        flow_v.ZeroData();
        flow_w.ZeroData();
      } else {
        op.Clear();
        op.PushValuePtr("input",         &flow_u);
        op.PushValuePtr("output",        &flow_u);
        op.PushValuePtr("data_size",     &prev_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_p_.Execute(op);
  
        op.Clear();
        op.PushValuePtr("input",         &flow_v);
        op.PushValuePtr("output",        &flow_v);
        op.PushValuePtr("data_size",     &prev_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_p_.Execute(op);
  
        op.Clear();
        op.PushValuePtr("input",         &flow_w);
        op.PushValuePtr("output",        &flow_w);
        op.PushValuePtr("data_size",     &prev_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_p_.Execute(op);
      }
    }

    /* Backward registration */
    {
      size_t max_magnitude = std::ceil(flow_stat.max / warp_scale_factor);

      op.Clear();
      op.PushValuePtr("frame_0",   p_frame_0_res);
      op.PushValuePtr("frame_1",   p_frame_1_res_br);
      op.PushValuePtr("flow_u",    &flow_u);
      op.PushValuePtr("flow_v",    &flow_v);
      op.PushValuePtr("flow_w",    &flow_w);
      op.PushValuePtr("temp",      &temp_0);
      op.PushValuePtr("hx",        &hx);
      op.PushValuePtr("hy",        &hy);
      op.PushValuePtr("hz",        &hz);
      op.PushValuePtr("data_size", &current_data_size);
      op.PushValuePtr("max_mag",   &max_magnitude);

      cuop_register_p_.Execute(op);
    }

    /* Difference problem solver */
    {
      op.Clear();
      op.PushValuePtr("frame_0", p_frame_0_res);
      op.PushValuePtr("frame_1", p_frame_1_res_br);
      op.PushValuePtr("flow_u",  &flow_u);
      op.PushValuePtr("flow_v",  &flow_v);
      op.PushValuePtr("flow_w",  &flow_w);
      op.PushValuePtr("flow_du", &flow_du);
      op.PushValuePtr("flow_dv", &flow_dv);
      op.PushValuePtr("flow_dw", &flow_dw);
      op.PushValuePtr("phi",     &phi);
      op.PushValuePtr("ksi",     &ksi);
      op.PushValuePtr("temp_du", &temp_0);
      op.PushValuePtr("temp_dv", &temp_1);
      op.PushValuePtr("temp_dw", &temp_2);


      op.PushValuePtr("outer_iterations_count", &outer_iterations_count);
      op.PushValuePtr("inner_iterations_count", &inner_iterations_count);
      op.PushValuePtr("equation_alpha",         &equation_alpha);
      op.PushValuePtr("equation_smoothness",    &equation_smoothness);
      op.PushValuePtr("equation_data",          &equation_data);
      op.PushValuePtr("data_size",              &current_data_size);
      op.PushValuePtr("hx",                     &hx);
      op.PushValuePtr("hy",                     &hy);
      op.PushValuePtr("hz",                     &hz);

      cuop_solve_p_.silent = silent;
      cuop_solve_p_.Execute(op);
    }

    /* Add the solved flow increment to the global flow */
    {
      op.Clear();
      op.PushValuePtr("operand_0", &flow_u);
      op.PushValuePtr("operand_1", &flow_du);
      op.PushValuePtr("data_size", &current_data_size);
      cuop_add_p_.Execute(op);

      op.Clear();
      op.PushValuePtr("operand_0", &flow_v);
      op.PushValuePtr("operand_1", &flow_dv);
      op.PushValuePtr("data_size", &current_data_size);
      cuop_add_p_.Execute(op);

      op.Clear();
      op.PushValuePtr("operand_0", &flow_w);
      op.PushValuePtr("operand_1", &flow_dw);
      op.PushValuePtr("data_size", &current_data_size);
      cuop_add_p_.Execute(op);
    }

     prev_data_size = current_data_size;
    --current_warp_level;

    /* Flow field median filtering */
    {
      //CUdeviceptr dev_temp = cuda_memory_ptrs_.top();
      //cuda_memory_ptrs_.pop();

      //op.Clear();
      //op.PushValuePtr("dev_input",  &dev_flow_u);
      //op.PushValuePtr("dev_output", &dev_temp);
      //op.PushValuePtr("data_size",  &current_data_size);
      //op.PushValuePtr("radius",     &median_radius);
      //cuop_median_.Execute(op);
      //std::swap(dev_flow_u, dev_temp);

      //op.Clear();
      //op.PushValuePtr("dev_input",  &dev_flow_v);
      //op.PushValuePtr("dev_output", &dev_temp);
      //op.PushValuePtr("data_size",  &current_data_size);
      //op.PushValuePtr("radius",     &median_radius);
      //cuop_median_.Execute(op);
      //std::swap(dev_flow_v, dev_temp);

      //op.Clear();
      //op.PushValuePtr("dev_input",  &dev_flow_w);
      //op.PushValuePtr("dev_output", &dev_temp);
      //op.PushValuePtr("data_size",  &current_data_size);
      //op.PushValuePtr("radius",     &median_radius);
      //cuop_median_.Execute(op);
      //std::swap(dev_flow_w, dev_temp);

      //cuda_memory_ptrs_.push(dev_temp);
    }
  }
  /* Estimate GPU computation time */
  CheckCudaError(cuEventRecord(cu_event_stop, NULL));
  CheckCudaError(cuEventSynchronize(cu_event_stop));

  float elapsed_time;
  CheckCudaError(cuEventElapsedTime(&elapsed_time, cu_event_start, cu_event_stop));

  std::printf("Total GPU computation time: % 4.4fs\n", elapsed_time / 1000.);

  CheckCudaError(cuEventDestroy(cu_event_start));
  CheckCudaError(cuEventDestroy(cu_event_stop));
}

void OpticalFlowP::Destroy()
{
  for (CudaOperationBase* cuop : cuda_operations_) {
    cuop->Destroy();
  }

  initialized_ = false;
}

OpticalFlowP::~OpticalFlowP()
{
  if (initialized_) {
    Destroy();
  }
}