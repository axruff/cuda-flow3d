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

#include "src/cuda_operations/entire_data/cuda_operation_resample.h"

#include <cstring>

#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationResample::CudaOperationResample()
  : CudaOperationBase("CUDA Resample")
{
}

bool CudaOperationResample::Initialize(const OperationParameters* params)
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
  std::strcat(exec_path, "/kernels/resample.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_resample_x_, cu_module_, "resample_x")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_resample_y_, cu_module_, "resample_y")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_resample_z_, cu_module_, "resample_z"))
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

void CudaOperationResample::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }
  CUdeviceptr dev_input = 0;
  CUdeviceptr dev_output = 0;
  CUdeviceptr dev_temp = 0;

  DataSize4 data_size;
  DataSize4 resample_size;

  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_input, "dev_input");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_output, "dev_output");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_temp, "dev_temp");
  GET_PARAM_OR_RETURN(params, DataSize4, data_size, "data_size");
  GET_PARAM_OR_RETURN(params, DataSize4, resample_size, "resample_size");

  if (dev_input == dev_output) {
    std::printf("Operation '%s': Error. Input buffer cannot serve as output buffer.", GetName());
    return;
  }

  DataSize4 output_size = data_size;
  output_size.width = resample_size.width;
  ResampleX(dev_input, dev_output, data_size, output_size);

  data_size.width = output_size.width;
  output_size.height = resample_size.height;
  ResampleY(dev_output, dev_temp, data_size, output_size);

  data_size.height = output_size.height;
  output_size.depth = resample_size.depth;
  ResampleZ(dev_temp, dev_output, data_size, output_size);
}

void CudaOperationResample::ResampleX(CUdeviceptr input, CUdeviceptr output, DataSize4& input_size, DataSize4& output_size) const
{
  dim3 block_dim = { 16, 8, 8 };
  dim3 grid_dim =  { static_cast<unsigned int>((output_size.width + block_dim.x - 1) / block_dim.x),
                     static_cast<unsigned int>((output_size.height + block_dim.y - 1) / block_dim.y),
                     static_cast<unsigned int>((output_size.depth + block_dim.z - 1) / block_dim.z) };

  void* args[6] = { &input, 
                    &output,
                    &output_size.width,
                    &output_size.height,
                    &output_size.depth,
                    &input_size.width };

  CheckCudaError(cuLaunchKernel(cuf_resample_x_,
                                grid_dim.x, grid_dim.y, grid_dim.z,
                                block_dim.x, block_dim.y, block_dim.z,
                                0,
                                NULL,
                                args,
                                NULL));
}

void CudaOperationResample::ResampleY(CUdeviceptr input, CUdeviceptr output, DataSize4& input_size, DataSize4& output_size) const
{
  dim3 block_dim = { 16, 8, 8 };
  dim3 grid_dim =  { static_cast<unsigned int>((output_size.width + block_dim.x - 1) / block_dim.x),
                     static_cast<unsigned int>((output_size.height + block_dim.y - 1) / block_dim.y),
                     static_cast<unsigned int>((output_size.depth + block_dim.z - 1) / block_dim.z) };

  void* args[6] = { &input,
                    &output,
                    &output_size.width,
                    &output_size.height,
                    &output_size.depth,
                    &input_size.height };

  CheckCudaError(cuLaunchKernel(cuf_resample_y_,
                                grid_dim.x, grid_dim.y, grid_dim.z,
                                block_dim.x, block_dim.y, block_dim.z,
                                0,
                                NULL,
                                args,
                                NULL));
}

void CudaOperationResample::ResampleZ(CUdeviceptr input, CUdeviceptr output, DataSize4& input_size, DataSize4& output_size) const
{
  dim3 block_dim = { 16, 8, 8 };
  dim3 grid_dim =  { static_cast<unsigned int>((output_size.width + block_dim.x - 1) / block_dim.x),
                     static_cast<unsigned int>((output_size.height + block_dim.y - 1) / block_dim.y),
                     static_cast<unsigned int>((output_size.depth + block_dim.z - 1) / block_dim.z) };

  void* args[6] = { &input,
                    &output,
                    &output_size.width,
                    &output_size.height,
                    &output_size.depth,
                    &input_size.depth };

  CheckCudaError(cuLaunchKernel(cuf_resample_z_,
                                grid_dim.x, grid_dim.y, grid_dim.z,
                                block_dim.x, block_dim.y, block_dim.z,
                                0,
                                NULL,
                                args,
                                NULL));
}