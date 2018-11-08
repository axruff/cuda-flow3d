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

#include "src/optical_flow/optical_flow_e.h"

#include <algorithm>
#include <cstdio>
#include <string>

#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

#ifndef NO_VISUALIZATION
    #include "../viewflow3d/src/visualization.h"
#endif

OpticalFlowE::OpticalFlowE()
  : OpticalFlowBase("Optical Flow Single GPU")
{
  cuda_operations_.push_front(&cuop_add_);
  cuda_operations_.push_front(&cuop_convolution_);
  cuda_operations_.push_front(&cuop_median_);
  cuda_operations_.push_front(&cuop_register_);
  cuda_operations_.push_front(&cuop_resample_);
  cuda_operations_.push_front(&cuop_solve_);
}

bool OpticalFlowE::Initialize(const DataSize4& data_size)
{
  dev_container_size_ = data_size;
  dev_container_size_.pitch = 0;

  initialized_ = InitCudaMemory() && InitCudaOperations();
  return initialized_;
}

bool OpticalFlowE::InitCudaOperations()
{
  if (dev_container_size_.pitch == 0) {
    std::printf("Initialization failed. Device pitch is 0.\n");
    return false;
  }

  std::printf("Initialization of cuda operations...\n");

  OperationParameters op;
  op.PushValuePtr("container_size", &dev_container_size_);

  for (CudaOperationBase* cuop : cuda_operations_) {
    std::printf("%-18s: ", cuop->GetName());
    bool result = cuop->Initialize(&op);
    if (result) {
      std::printf("OK\n");
    } else {
      Destroy();
      return false;
    }
  }
  return true;
}

bool OpticalFlowE::InitCudaMemory()
{
  std::printf("Allocating memory on the device...\n");
  /* Check available memory on the cuda device */
  size_t free_memory;
  size_t total_memory;

  CheckCudaError(cuMemGetInfo(&free_memory, &total_memory));

  std::printf("Available\t:\t%.0fMB / %.0fMB\n", 
              free_memory / static_cast<float>(1024 * 1024),
              total_memory / static_cast<float>(1024 * 1024));

  /* Estimate needed memory (approx.)*/
  int alignment = 512;
  CUdevice cu_device;
  CheckCudaError(cuCtxGetDevice(&cu_device));
  CheckCudaError(cuDeviceGetAttribute(&alignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, cu_device));
  alignment /= sizeof(float);

  size_t pitch = ((dev_container_size_.width % alignment == 0) ?
                  (dev_container_size_.width) :
                  (dev_container_size_.width + alignment - (dev_container_size_.width % alignment))) *
                 sizeof(float);
  size_t needed_memory = pitch * dev_container_size_.height * dev_container_size_.depth * dev_containers_count_;
  std::printf("Needed (approx.):\t%.0fMB\n", needed_memory / static_cast<float>(1024 * 1024));

  size_t allocated_memory = 0;
  if (needed_memory < free_memory) {
    for (size_t i = 0; i < dev_containers_count_; ++i) {
      CUdeviceptr dev_container;
      bool error = CheckCudaError(cuMemAllocPitch(&dev_container,
                                                  &pitch,
                                                  dev_container_size_.width * sizeof(float),
                                                  dev_container_size_.height * dev_container_size_.depth,
                                                  sizeof(float)));
      /* Pitch should be the same for all containers */
      if (error || (i != 0 && dev_container_size_.pitch != pitch)) {
        std::printf("Error during device memory allocation.");
        Destroy();
        return false;
      } else {
        size_t container_size;
        CheckCudaError(cuMemGetAddressRange(NULL, &container_size, dev_container));
        allocated_memory += container_size;

        cuda_memory_ptrs_.push(dev_container);
        dev_container_size_.pitch = pitch;
      }
    }
    std::printf("Allocated\t:\t%.0fMB\n", allocated_memory / static_cast<float>(1024 * 1024));
    return true;
  } 
  return false;
}

void OpticalFlowE::ComputeFlow(Data3D& frame_0, Data3D& frame_1, Data3D& flow_u, Data3D& flow_v, Data3D&flow_w, OperationParameters& params)
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

  std::printf("\nStarting optical flow computation...\n");

  /* Create CUDA event for the time measure */
  CUevent cu_event_start;
  CUevent cu_event_stop;

  CheckCudaError(cuEventCreate(&cu_event_start, CU_EVENT_DEFAULT));
  CheckCudaError(cuEventCreate(&cu_event_stop, CU_EVENT_DEFAULT));

  CheckCudaError(cuEventRecord(cu_event_start, NULL));

  /* Auxiliary variables */
  float hx; // spacing in x-direction (current resol.) 
  float hy; // spacing in y-direction (current resol.) 
  float hz; // spacing in z-direction (current resol.) 
  DataSize4 original_data_size = { frame_0.Width(), frame_0.Height(), frame_0.Depth(), 0 };
  DataSize4 current_data_size = { 0 };
  DataSize4 prev_data_size = { 0 };
  
  size_t max_warp_level = GetMaxWarpLevel(original_data_size.width, original_data_size.height, original_data_size.depth, warp_scale_factor);
  int current_warp_level = std::min(warp_levels_count, max_warp_level) - 1;

  OperationParameters op;

  /* Copy input data to the device */
  CUdeviceptr dev_frame_0 = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();
  CUdeviceptr dev_frame_1 = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();
  CUdeviceptr dev_frame_0_res = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();
  CUdeviceptr dev_frame_1_res_br = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();

  CUdeviceptr dev_flow_u = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();
  CUdeviceptr dev_flow_v = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();
  CUdeviceptr dev_flow_w = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();

  CUdeviceptr dev_flow_du = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();
  CUdeviceptr dev_flow_dv = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();
  CUdeviceptr dev_flow_dw = cuda_memory_ptrs_.top();
  cuda_memory_ptrs_.pop();

  CopyData3DtoDevice(frame_0, dev_frame_0, dev_container_size_.height, dev_container_size_.pitch);
  CopyData3DtoDevice(frame_1, dev_frame_1, dev_container_size_.height, dev_container_size_.pitch);


  /* Gaussian blurring */
  if (gaussian_sigma > 0.0) {

      CUdeviceptr dev_temp = cuda_memory_ptrs_.top();
      cuda_memory_ptrs_.pop();

      op.Clear();
      op.PushValuePtr("dev_input", &dev_frame_0);
      op.PushValuePtr("dev_output", &dev_flow_u);
      op.PushValuePtr("dev_temp", &dev_temp);
      op.PushValuePtr("data_size", &original_data_size);
      op.PushValuePtr("gaussian_sigma", &gaussian_sigma);
      cuop_convolution_.Execute(op);

      //CheckCudaError(cuStreamSynchronize(NULL));
      //std::this_thread::sleep_for(std::chrono::milliseconds(500));


      op.Clear();
      op.PushValuePtr("dev_input", &dev_frame_1);
      op.PushValuePtr("dev_output", &dev_flow_v);
      op.PushValuePtr("dev_temp", &dev_temp);
      op.PushValuePtr("data_size", &original_data_size);
      op.PushValuePtr("gaussian_sigma", &gaussian_sigma);
      cuop_convolution_.Execute(op);

      std::swap(dev_frame_0, dev_flow_u);
      std::swap(dev_frame_1, dev_flow_v);

      cuda_memory_ptrs_.push(dev_temp);


      // Test: Save smoothed image
      CopyData3DFromDevice(dev_frame_0, frame_0, dev_container_size_.height, dev_container_size_.pitch);
      CopyData3DFromDevice(dev_frame_1, frame_1, dev_container_size_.height, dev_container_size_.pitch);

      std::string filename =
      "-" + std::to_string(dev_container_size_.width) +
      "-" + std::to_string(dev_container_size_.height) +
      "-" + std::to_string(dev_container_size_.depth) + ".raw";


      frame_0.WriteRAWToFileF32(std::string("./data/output/test1_smooth" + filename).c_str());

      frame_1.WriteRAWToFileF32(std::string("./data/output/test2_smooth" + filename).c_str());

  }


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
        std::swap(dev_frame_0, dev_frame_0_res);
        std::swap(dev_frame_1, dev_frame_1_res_br);
      } else {
        CUdeviceptr dev_temp = cuda_memory_ptrs_.top();
        cuda_memory_ptrs_.pop();

        op.Clear();
        op.PushValuePtr("dev_input",     &dev_frame_0);
        op.PushValuePtr("dev_output",    &dev_frame_0_res);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &original_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_.Execute(op);

        op.Clear();
        op.PushValuePtr("dev_input",     &dev_frame_1);
        op.PushValuePtr("dev_output",    &dev_frame_1_res_br);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &original_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_.Execute(op);
  
        cuda_memory_ptrs_.push(dev_temp);
      }
    }

    /* Flow field resampling */
    {
      if (prev_data_size.width == 0) {
        CheckCudaError(cuMemsetD2D8(dev_flow_u, dev_container_size_.pitch, 0, dev_container_size_.width * sizeof(float), 
                                                dev_container_size_.height * dev_container_size_.depth));
        CheckCudaError(cuMemsetD2D8(dev_flow_v, dev_container_size_.pitch, 0, dev_container_size_.width  * sizeof(float), 
                                                dev_container_size_.height * dev_container_size_.depth));
        CheckCudaError(cuMemsetD2D8(dev_flow_w, dev_container_size_.pitch, 0, dev_container_size_.width  * sizeof(float), 
                                                dev_container_size_.height * dev_container_size_.depth));
      } else {
        CUdeviceptr dev_temp = cuda_memory_ptrs_.top();
        cuda_memory_ptrs_.pop();
  
        op.Clear();
        op.PushValuePtr("dev_input",     &dev_flow_u);
        op.PushValuePtr("dev_output",    &dev_flow_du);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &prev_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_.Execute(op);
  
        op.Clear();
        op.PushValuePtr("dev_input",     &dev_flow_v);
        op.PushValuePtr("dev_output",    &dev_flow_dv);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &prev_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_.Execute(op);
  
        op.Clear();
        op.PushValuePtr("dev_input",     &dev_flow_w);
        op.PushValuePtr("dev_output",    &dev_flow_dw);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &prev_data_size);
        op.PushValuePtr("resample_size", &current_data_size);
        cuop_resample_.Execute(op);

        std::swap(dev_flow_u, dev_flow_du);
        std::swap(dev_flow_v, dev_flow_dv);
        std::swap(dev_flow_w, dev_flow_dw);

        cuda_memory_ptrs_.push(dev_temp);
      }
    }

    /* Backward registration */
    {
      CUdeviceptr dev_temp = cuda_memory_ptrs_.top();
      cuda_memory_ptrs_.pop();

      op.Clear();
      op.PushValuePtr("dev_frame_0", &dev_frame_0_res);
      op.PushValuePtr("dev_frame_1", &dev_frame_1_res_br);
      op.PushValuePtr("dev_flow_u",  &dev_flow_u);
      op.PushValuePtr("dev_flow_v",  &dev_flow_v);
      op.PushValuePtr("dev_flow_w",  &dev_flow_w);
      op.PushValuePtr("dev_output",  &dev_temp);
      op.PushValuePtr("data_size",   &current_data_size);
      op.PushValuePtr("hx",          &hx);
      op.PushValuePtr("hy",          &hy);
      op.PushValuePtr("hz",          &hz);

      cuop_register_.Execute(op);

      std::swap(dev_frame_1_res_br, dev_temp);

      cuda_memory_ptrs_.push(dev_temp);
    }

    /* Difference problem solver */
    {
      CUdeviceptr dev_phi = cuda_memory_ptrs_.top();
      cuda_memory_ptrs_.pop();
      CUdeviceptr dev_ksi = cuda_memory_ptrs_.top();
      cuda_memory_ptrs_.pop();
      CUdeviceptr dev_temp_du = cuda_memory_ptrs_.top();
      cuda_memory_ptrs_.pop();
      CUdeviceptr dev_temp_dv = cuda_memory_ptrs_.top();
      cuda_memory_ptrs_.pop();
      CUdeviceptr dev_temp_dw = cuda_memory_ptrs_.top();
      cuda_memory_ptrs_.pop();

      op.Clear();
      op.PushValuePtr("dev_frame_0", &dev_frame_0_res);
      op.PushValuePtr("dev_frame_1", &dev_frame_1_res_br);
      op.PushValuePtr("dev_flow_u",  &dev_flow_u);
      op.PushValuePtr("dev_flow_v",  &dev_flow_v);
      op.PushValuePtr("dev_flow_w",  &dev_flow_w);
      op.PushValuePtr("dev_flow_du", &dev_flow_du);
      op.PushValuePtr("dev_flow_dv", &dev_flow_dv);
      op.PushValuePtr("dev_flow_dw", &dev_flow_dw);
      op.PushValuePtr("dev_phi",     &dev_phi);
      op.PushValuePtr("dev_ksi",     &dev_ksi);
      op.PushValuePtr("dev_temp_du", &dev_temp_du);
      op.PushValuePtr("dev_temp_dv", &dev_temp_dv);
      op.PushValuePtr("dev_temp_dw", &dev_temp_dw);

      op.PushValuePtr("outer_iterations_count", &outer_iterations_count);
      op.PushValuePtr("inner_iterations_count", &inner_iterations_count);
      op.PushValuePtr("equation_alpha",         &equation_alpha);
      op.PushValuePtr("equation_smoothness",    &equation_smoothness);
      op.PushValuePtr("equation_data",          &equation_data);
      op.PushValuePtr("data_size",              &current_data_size);
      op.PushValuePtr("hx",                     &hx);
      op.PushValuePtr("hy",                     &hy);
      op.PushValuePtr("hz",                     &hz);

      cuop_solve_.silent = silent;
      cuop_solve_.Execute(op);

      cuda_memory_ptrs_.push(dev_phi);
      cuda_memory_ptrs_.push(dev_ksi);
      cuda_memory_ptrs_.push(dev_temp_du);
      cuda_memory_ptrs_.push(dev_temp_dv);
      cuda_memory_ptrs_.push(dev_temp_dw);
    }

    /* Add the solved flow increment to the global flow */
    {
      op.Clear();
      op.PushValuePtr("operand_0", &dev_flow_u);
      op.PushValuePtr("operand_1", &dev_flow_du);
      op.PushValuePtr("data_size", &current_data_size);
      cuop_add_.Execute(op);

      op.Clear();
      op.PushValuePtr("operand_0", &dev_flow_v);
      op.PushValuePtr("operand_1", &dev_flow_dv);
      op.PushValuePtr("data_size", &current_data_size);
      cuop_add_.Execute(op);

      op.Clear();
      op.PushValuePtr("operand_0", &dev_flow_w);
      op.PushValuePtr("operand_1", &dev_flow_dw);
      op.PushValuePtr("data_size", &current_data_size);
      cuop_add_.Execute(op);
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

#ifndef NO_VISUALIZATION
    /* Get data for visualization */
    {
      Visualization& visualization = Visualization::GetInstance();

      if (visualization.IsInitialized()) {
        CUdeviceptr dev_temp_u = cuda_memory_ptrs_.top();
        cuda_memory_ptrs_.pop();
        CUdeviceptr dev_temp_v = cuda_memory_ptrs_.top();
        cuda_memory_ptrs_.pop();
        CUdeviceptr dev_temp_w = cuda_memory_ptrs_.top();
        cuda_memory_ptrs_.pop();
        CUdeviceptr dev_temp = cuda_memory_ptrs_.top();
        cuda_memory_ptrs_.pop();

        /* Resample the flow field to original size */
        op.Clear();
        op.PushValuePtr("dev_input",     &dev_flow_u);
        op.PushValuePtr("dev_output",    &dev_temp_u);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &current_data_size);
        op.PushValuePtr("resample_size", &original_data_size);
        cuop_resample_.Execute(op);

        op.Clear();
        op.PushValuePtr("dev_input",     &dev_flow_v);
        op.PushValuePtr("dev_output",    &dev_temp_v);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &current_data_size);
        op.PushValuePtr("resample_size", &original_data_size);
        cuop_resample_.Execute(op);

        op.Clear();
        op.PushValuePtr("dev_input",     &dev_flow_w);
        op.PushValuePtr("dev_output",    &dev_temp_w);
        op.PushValuePtr("dev_temp",      &dev_temp);
        op.PushValuePtr("data_size",     &current_data_size);
        op.PushValuePtr("resample_size", &original_data_size);
        cuop_resample_.Execute(op);

        /* Read output data back from the device */
        CopyData3DFromDevice(dev_temp_u, flow_u, dev_container_size_.height, dev_container_size_.pitch);
        CopyData3DFromDevice(dev_temp_v, flow_v, dev_container_size_.height, dev_container_size_.pitch);
        CopyData3DFromDevice(dev_temp_w, flow_w, dev_container_size_.height, dev_container_size_.pitch);

        /* Update 3D texture */
        visualization.Load3DFlowTextureFromData3DUVW(flow_u, flow_v, flow_w);

        cuda_memory_ptrs_.push(dev_temp);
        cuda_memory_ptrs_.push(dev_temp_u);
        cuda_memory_ptrs_.push(dev_temp_v);
        cuda_memory_ptrs_.push(dev_temp_w);

        visualization.DoNextStep();
      }
    }
#endif

  }

  /* DEBUG Apply computed flow to the input data */
  if (false) {
    CopyData3DtoDevice(frame_0, dev_frame_0, dev_container_size_.height, dev_container_size_.pitch);
    CopyData3DtoDevice(frame_1, dev_frame_1, dev_container_size_.height, dev_container_size_.pitch);

    CheckCudaError(cuStreamSynchronize(NULL));

    CUdeviceptr dev_temp = cuda_memory_ptrs_.top();
    cuda_memory_ptrs_.pop();

    hx = 1.f;
    hy = 1.f;
    hz = 1.f;

    op.Clear();
    op.PushValuePtr("dev_frame_0", &dev_frame_0);
    op.PushValuePtr("dev_frame_1", &dev_frame_1);
    op.PushValuePtr("dev_flow_u", &dev_flow_u);
    op.PushValuePtr("dev_flow_v", &dev_flow_v);
    op.PushValuePtr("dev_flow_w", &dev_flow_w);
    op.PushValuePtr("dev_output", &dev_temp);
    op.PushValuePtr("data_size", &original_data_size);
    op.PushValuePtr("hx", &hx);
    op.PushValuePtr("hy", &hy);
    op.PushValuePtr("hz", &hz);

    cuop_register_.Execute(op);
    CheckCudaError(cuStreamSynchronize(NULL));
    CopyData3DFromDevice(dev_temp, flow_u, dev_container_size_.height, dev_container_size_.pitch);
    CheckCudaError(cuStreamSynchronize(NULL));

    flow_u.WriteRAWToFileU8("./data/output/registrated-180-180-151.raw");
    flow_u.WriteRAWToFileF32("./data/output/registrated-180-180-151-f.raw");


    cuda_memory_ptrs_.push(dev_temp);
  }

  /* Read output data back from the device */
  CopyData3DFromDevice(dev_flow_u, flow_u, dev_container_size_.height, dev_container_size_.pitch);
  CopyData3DFromDevice(dev_flow_v, flow_v, dev_container_size_.height, dev_container_size_.pitch);
  CopyData3DFromDevice(dev_flow_w, flow_w, dev_container_size_.height, dev_container_size_.pitch);

  /* Estimate GPU computation time */
  CheckCudaError(cuEventRecord(cu_event_stop, NULL));
  CheckCudaError(cuEventSynchronize(cu_event_stop));

  float elapsed_time;
  CheckCudaError(cuEventElapsedTime(&elapsed_time, cu_event_start, cu_event_stop));
  
  std::printf("Total GPU computation time: % 4.4fs\n", elapsed_time / 1000.);

  CheckCudaError(cuEventDestroy(cu_event_start));
  CheckCudaError(cuEventDestroy(cu_event_stop));

  /* Return all memory pointers to stack */
  cuda_memory_ptrs_.push(dev_frame_0);
  cuda_memory_ptrs_.push(dev_frame_1);
  cuda_memory_ptrs_.push(dev_frame_0_res);
  cuda_memory_ptrs_.push(dev_frame_1_res_br);
  cuda_memory_ptrs_.push(dev_flow_u);
  cuda_memory_ptrs_.push(dev_flow_v);
  cuda_memory_ptrs_.push(dev_flow_w);
  cuda_memory_ptrs_.push(dev_flow_du);
  cuda_memory_ptrs_.push(dev_flow_dv);
  cuda_memory_ptrs_.push(dev_flow_dw);
}

void OpticalFlowE::Destroy()
{
  for (CudaOperationBase* cuop : cuda_operations_) {
    cuop->Destroy();
  }

  size_t freed_containers_count = 0;
  while (!cuda_memory_ptrs_.empty()) {
    CUdeviceptr dev_container = cuda_memory_ptrs_.top();
    CheckCudaError(cuMemFree(dev_container));
    
    ++freed_containers_count;
    cuda_memory_ptrs_.pop();
  }
  if (freed_containers_count && freed_containers_count != dev_containers_count_) {
    std::printf("Warning. Not all device memory allocations were freed.\n");
  }

  initialized_ = false;
}

OpticalFlowE::~OpticalFlowE()
{
  Destroy();
}