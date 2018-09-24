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

#include "src/cuda_operations/entire_data/cuda_operation_add.h"

#include <cstring>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationAdd::CudaOperationAdd()
  : CudaOperationBase("CUDA Add")
{
}

bool CudaOperationAdd::Initialize(const OperationParameters* params)
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
  std::strcat(exec_path, "/kernels/add.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_add_, cu_module_, "add"))) {
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

void CudaOperationAdd::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  CUdeviceptr operand_0 = 0;
  CUdeviceptr operand_1 = 0;
  DataSize4 data_size;

  GET_PARAM_OR_RETURN(params, CUdeviceptr, operand_0, "operand_0");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, operand_1, "operand_1");
  GET_PARAM_OR_RETURN(params, DataSize4, data_size, "data_size");

  size_t block_dim3[3] = { 16, 8, 4 };
  size_t grid_dim3[3] = { (data_size.width + block_dim3[0] - 1) / block_dim3[0],
                          (data_size.height + block_dim3[1] - 1) / block_dim3[1],
                          (data_size.depth + block_dim3[2] - 1) / block_dim3[2] };

  void* args[5] = { 
    &operand_0, 
    &operand_1,
    &data_size.width, 
    &data_size.height, 
    &data_size.depth };

  CheckCudaError(cuLaunchKernel(cuf_add_,
                                grid_dim3[0], grid_dim3[1], grid_dim3[2],
                                block_dim3[0], block_dim3[1], block_dim3[2],
                                0,
                                NULL,
                                args,
                                NULL));
}