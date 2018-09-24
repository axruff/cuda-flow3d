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

#include "src/cuda_operations/entire_data/cuda_operation_solve.h"

#include <cstdio>
#include <cstring>

#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"


CudaOperationSolve::CudaOperationSolve()
  : CudaOperationBase("CUDA Solve")
{
}

bool CudaOperationSolve::Initialize(const OperationParameters* params)
{
  initialized_ = false;
  
  if (!params) {
    std::printf("Operation: '%s'. Initialization parameters are missing.\n", GetName());
    return initialized_;
  }

  DataSize4 container_size;
  GET_PARAM_OR_RETURN_VALUE(*params, DataSize4, container_size, "container_size", initialized_);
 
  dev_container_size_ = container_size;

  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/solve_3d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_compute_phi_ksi_, cu_module_, "compute_phi_ksi_3d")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_solve_, cu_module_, "solve_3d"))
        ) {
      size_t const_size;

      /* Get the pointer to the constant memory and copy data */
      if (!CheckCudaError(cuModuleGetGlobal(&dev_constants_, &const_size, cu_module_, "container_size"))) {
        if (const_size == sizeof(container_size)) {
          if (!CheckCudaError(cuMemcpyHtoD(dev_constants_, &container_size, sizeof(container_size)))) {
             initialized_ = true;
          }
        }
      }

    } else {
      CheckCudaError(cuModuleUnload(cu_module_));
      cu_module_ = nullptr;
    }
  }
  return initialized_;
}

void CudaOperationSolve::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  CUdeviceptr dev_frame_0;
  CUdeviceptr dev_frame_1;
  CUdeviceptr dev_flow_u;
  CUdeviceptr dev_flow_v;
  CUdeviceptr dev_flow_w;
  CUdeviceptr dev_phi;
  CUdeviceptr dev_ksi;

  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_frame_0, "dev_frame_0");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_frame_1, "dev_frame_1");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_u,  "dev_flow_u");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_v,  "dev_flow_v");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_w,  "dev_flow_w");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_phi,     "dev_phi");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_ksi,     "dev_ksi");

  /* Use references to guaranty correct pointer values
     outside this operation after using std::swap() */
  CUdeviceptr* dev_flow_du_ptr;
  CUdeviceptr* dev_flow_dv_ptr;
  CUdeviceptr* dev_flow_dw_ptr;
  CUdeviceptr* dev_temp_du_ptr;
  CUdeviceptr* dev_temp_dv_ptr;
  CUdeviceptr* dev_temp_dw_ptr;

  GET_PARAM_PTR_OR_RETURN(params, CUdeviceptr, dev_flow_du_ptr, "dev_flow_du");
  GET_PARAM_PTR_OR_RETURN(params, CUdeviceptr, dev_flow_dv_ptr, "dev_flow_dv");
  GET_PARAM_PTR_OR_RETURN(params, CUdeviceptr, dev_flow_dw_ptr, "dev_flow_dw");
  GET_PARAM_PTR_OR_RETURN(params, CUdeviceptr, dev_temp_du_ptr, "dev_temp_du");
  GET_PARAM_PTR_OR_RETURN(params, CUdeviceptr, dev_temp_dv_ptr, "dev_temp_dv");
  GET_PARAM_PTR_OR_RETURN(params, CUdeviceptr, dev_temp_dw_ptr, "dev_temp_dw");

  CUdeviceptr& dev_flow_du = *dev_flow_du_ptr;
  CUdeviceptr& dev_flow_dv = *dev_flow_dv_ptr;
  CUdeviceptr& dev_flow_dw = *dev_flow_dw_ptr;
  CUdeviceptr& dev_temp_du = *dev_temp_du_ptr;
  CUdeviceptr& dev_temp_dv = *dev_temp_dv_ptr;
  CUdeviceptr& dev_temp_dw = *dev_temp_dw_ptr;

  size_t outer_iterations_count;
  size_t inner_iterations_count;
  float equation_alpha;
  float equation_smoothness;
  float equation_data;
  float hx;
  float hy;
  float hz;
  DataSize4 data_size;

  GET_PARAM_OR_RETURN(params, size_t,    outer_iterations_count, "outer_iterations_count");
  GET_PARAM_OR_RETURN(params, size_t,    inner_iterations_count, "inner_iterations_count");
  GET_PARAM_OR_RETURN(params, float,     equation_alpha,         "equation_alpha");
  GET_PARAM_OR_RETURN(params, float,     equation_smoothness,    "equation_smoothness");
  GET_PARAM_OR_RETURN(params, float,     equation_data,          "equation_data");
  GET_PARAM_OR_RETURN(params, float,     hx,                     "hx");
  GET_PARAM_OR_RETURN(params, float,     hy,                     "hy");
  GET_PARAM_OR_RETURN(params, float,     hz,                     "hz");
  GET_PARAM_OR_RETURN(params, DataSize4, data_size,              "data_size");


  /* Run solver kernels */
  dim3 block_dim = { 16, 8, 4 };

  int shared_memory_size;
  CUdevice cu_device;
  CheckCudaError(cuDeviceGet(&cu_device, 0));
  CheckCudaError(cuDeviceGetAttribute(&shared_memory_size, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, cu_device));

  /* We need 8 extended blocks in the shared memory */
  int number_shared_blocks = 8;
  int needed_shared_memory_size =
    (block_dim.x + 2) * (block_dim.y + 2) * (block_dim.z + 2) * sizeof(float) * number_shared_blocks;

  int number_shared_blocks_solve = 10;
  int needed_shared_memory_size_solve =
    (block_dim.x + 2) * (block_dim.y + 2) * (block_dim.z + 2) * sizeof(float) * number_shared_blocks_solve;

  if (needed_shared_memory_size > shared_memory_size || needed_shared_memory_size_solve > shared_memory_size) {
    std::printf("<%s>: Error shared memory allocation. Reduce thread block size.\n", GetName());
    std::printf("Shared memory: %d Needed: %d | %d\n", shared_memory_size, needed_shared_memory_size, needed_shared_memory_size_solve);
    return;
  }

  /* Mesure execution time */
  size_t total_iterations_count = outer_iterations_count * inner_iterations_count;

  CUevent cu_event_start;
  CUevent cu_event_stop;

  CheckCudaError(cuEventCreate(&cu_event_start, CU_EVENT_DEFAULT));
  CheckCudaError(cuEventCreate(&cu_event_stop, CU_EVENT_DEFAULT));

 
  CheckCudaError(cuEventRecord(cu_event_start, NULL));

  if (!silent) {
      /* Display computation status */
      Utils::PrintProgressBar(0.f);
      std::printf(" % 3.0f%%", 0.f);
  }

  /* Initialize flow increment buffers with 0 */
  CheckCudaError(cuMemsetD2D8(dev_flow_du, dev_container_size_.pitch, 0, data_size.width * sizeof(float),
                                           dev_container_size_.height * dev_container_size_.depth));
  CheckCudaError(cuMemsetD2D8(dev_flow_dv, dev_container_size_.pitch, 0, data_size.width * sizeof(float),
                                           dev_container_size_.height * dev_container_size_.depth));
  CheckCudaError(cuMemsetD2D8(dev_flow_dw, dev_container_size_.pitch, 0, data_size.width * sizeof(float),
                                           dev_container_size_.height * dev_container_size_.depth));

  dim3 grid_dim = { static_cast<unsigned int>((data_size.width + block_dim.x - 1) / block_dim.x),
                    static_cast<unsigned int>((data_size.height + block_dim.y - 1) / block_dim.y),
                    static_cast<unsigned int>((data_size.depth + block_dim.z - 1) / block_dim.z) };

  for (size_t i = 0; i < outer_iterations_count; ++i) {
    void* args[18] = { 
      &dev_frame_0,
      &dev_frame_1,
      &dev_flow_u,
      &dev_flow_v,
      &dev_flow_w,
      &dev_flow_du,
      &dev_flow_dv,
      &dev_flow_dw,
      &data_size.width,
      &data_size.height,
      &data_size.depth,
      &hx,
      &hy,
      &hz,
      &equation_smoothness,
      &equation_data,
      &dev_phi,
      &dev_ksi };

    CheckCudaError(cuLaunchKernel(cuf_compute_phi_ksi_,
                                  grid_dim.x, grid_dim.y, grid_dim.z,
                                  block_dim.x, block_dim.y, block_dim.z,
                                  needed_shared_memory_size,
                                  NULL,
                                  args,
                                  NULL));

    for (size_t j = 0; j < inner_iterations_count; ++j) {
      void* args[20] = {
        &dev_frame_0,
        &dev_frame_1,
        &dev_flow_u,
        &dev_flow_v,
        &dev_flow_w,
        &dev_flow_du,
        &dev_flow_dv,
        &dev_flow_dw,
        &dev_phi,
        &dev_ksi,
        &data_size.width,
        &data_size.height,
        &data_size.depth,
        &hx,
        &hy,
        &hz,
        &equation_alpha,
        &dev_temp_du,
        &dev_temp_dv,
        &dev_temp_dw };

      CheckCudaError(cuLaunchKernel(cuf_solve_,
                                    grid_dim.x, grid_dim.y, grid_dim.z,
                                    block_dim.x, block_dim.y, block_dim.z,
                                    needed_shared_memory_size_solve,
                                    NULL,
                                    args,
                                    NULL));
      std::swap(dev_flow_du, dev_temp_du);
      std::swap(dev_flow_dv, dev_temp_dv);
      std::swap(dev_flow_dw, dev_temp_dw);
      
      CheckCudaError(cuStreamSynchronize(NULL));

      /* Display computation status */
      if (!silent) {
          float complete = (i * inner_iterations_count + j) / static_cast<float>(total_iterations_count);
          Utils::PrintProgressBar(complete);
          std::printf(" % 3.0f%%", complete * 100);
      }
    }
  }
  /* Estimate GPU computation time */
  CheckCudaError(cuEventRecord(cu_event_stop, NULL));
  CheckCudaError(cuEventSynchronize(cu_event_stop));

  float elapsed_time;
  CheckCudaError(cuEventElapsedTime(&elapsed_time, cu_event_start, cu_event_stop));
  
  if (!silent) {
      Utils::PrintProgressBar(1.f);
      std::printf(" %8.4fs\n", elapsed_time / 1000.);
  }

  CheckCudaError(cuEventDestroy(cu_event_start));
  CheckCudaError(cuEventDestroy(cu_event_stop));
}