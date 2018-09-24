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

#include "src/cuda_operations/entire_data/cuda_operation_registration.h"

#include <cuda.h>
#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationRegistration::CudaOperationRegistration()
  : CudaOperationBase("CUDA Registration")
{
}

bool CudaOperationRegistration::Initialize(const OperationParameters* params)
{
  initialized_ = false;
  
  if (!params) {
    std::printf("Operation: '%s'. Initialization parameters are missing.\n", GetName());
    return initialized_;
  }

  DataSize4 container_size;
  GET_PARAM_OR_RETURN_VALUE(*params, DataSize4, container_size, "container_size", initialized_);
 
  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/registration_3d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_registration_, cu_module_, "registration_3d"))) {
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

void CudaOperationRegistration::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  CUdeviceptr dev_frame_0;
  CUdeviceptr dev_frame_1;
  CUdeviceptr dev_flow_u;
  CUdeviceptr dev_flow_v;
  CUdeviceptr dev_flow_w;
  CUdeviceptr dev_output;

  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_frame_0, "dev_frame_0");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_frame_1, "dev_frame_1");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_u,  "dev_flow_u");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_v,  "dev_flow_v");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_w,  "dev_flow_w");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_output,  "dev_output");

  float hx;
  float hy;
  float hz;
  DataSize4 data_size;

  GET_PARAM_OR_RETURN(params, float,     hx,        "hx");
  GET_PARAM_OR_RETURN(params, float,     hy,        "hy");
  GET_PARAM_OR_RETURN(params, float,     hz,        "hz");
  GET_PARAM_OR_RETURN(params, DataSize4, data_size, "data_size");

  if (dev_frame_1 == dev_output) {
    std::printf("Operation '%s': Error. Input buffer cannot serve as output buffer.", GetName());
    return;
  }

  dim3 block_dim = { 16, 8, 4 };
  dim3 grid_dim = { static_cast<unsigned int>((data_size.width + block_dim.x - 1) / block_dim.x),
                    static_cast<unsigned int>((data_size.height + block_dim.y - 1) / block_dim.y),
                    static_cast<unsigned int>((data_size.depth + block_dim.z - 1) / block_dim.z) };

  void* args[12] = { 
      &dev_frame_0,
      &dev_frame_1,
      &dev_flow_u,
      &dev_flow_v,
      &dev_flow_w,
      &data_size.width,
      &data_size.height,
      &data_size.depth,
      &hx,
      &hy,
      &hz,
      &dev_output};

  CheckCudaError(cuLaunchKernel(cuf_registration_,
                                grid_dim.x, grid_dim.y, grid_dim.z,
                                block_dim.x, block_dim.y, block_dim.z,
                                0,
                                NULL,
                                args,
                                NULL));
}