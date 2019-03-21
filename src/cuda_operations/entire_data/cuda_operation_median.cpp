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

#include "src/cuda_operations/entire_data/cuda_operation_median.h"

#include <cstring>

#include <cuda.h>
#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationMedian::CudaOperationMedian()
  : CudaOperationBase("CUDA Median")
{
}

bool CudaOperationMedian::Initialize(const OperationParameters* params)
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
  std::strcat(exec_path, "/kernels/median_3d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_median_, cu_module_, "median_3d"))) {
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

void CudaOperationMedian::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  CUdeviceptr dev_input;
  CUdeviceptr dev_output;

  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_input,  "dev_input");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_output, "dev_output");

  DataSize4 data_size;
  size_t radius;
  GET_PARAM_OR_RETURN(params, DataSize4, data_size, "data_size");
  GET_PARAM_OR_RETURN(params, size_t,    radius,    "radius");

  if (dev_input == dev_output) {
    std::printf("Operation '%s': Error. Input buffer cannot serve as output buffer.", GetName());
    return;
  }

  /* If hte radius equals 1 skip the filtering and copy data from the input to the output */
  if (radius == 1) {
    CheckCudaError(cuMemcpyDtoD(dev_output, dev_input,
      dev_container_size_.pitch * dev_container_size_.height *dev_container_size_.depth));
    return;
  }

  if (radius % 2 == 0) {
    std::printf("Warning. Median raduis is even (%d), decresaing by 1...\n", radius);
    radius -= 1;
  }

  if (radius >= 3 && radius <= 7) {
    /* Be careful with the thread block size. Thread block size in each dimension
       should be greater than radius / 2, because radius / 2 threads are used to 
       load halo around the thread block. */
    dim3 block_dim = { 16, 8, 4 };

    /* Calculate the needed shared memory size */
    int shared_memory_size;
    CUdevice cu_device;
    CheckCudaError(cuDeviceGet(&cu_device, 0));
    CheckCudaError(cuDeviceGetAttribute(&shared_memory_size, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, cu_device));

    int radius_2 = radius / 2;
    int needed_shared_memory_size =
      (block_dim.x + 2 * radius_2) * (block_dim.y + 2 * radius_2) * (block_dim.z + 2 * radius_2) * sizeof(float);

    if (needed_shared_memory_size > shared_memory_size) {
      std::printf("<%s>: Error shared memory allocation. Reduce the thread block size.\n", GetName());
      std::printf("Shared memory: %d Needed: %d \n", shared_memory_size, needed_shared_memory_size);
      return;
    }

    dim3 grid_dim = { static_cast<unsigned int>((data_size.width + block_dim.x - 1) / block_dim.x),
                      static_cast<unsigned int>((data_size.height + block_dim.y - 1) / block_dim.y),
                      static_cast<unsigned int>((data_size.depth + block_dim.z - 1) / block_dim.z) };
    void* args[6] = { 
      &dev_input,
      &data_size.width,
      &data_size.height,
      &data_size.depth,
      &radius,
      &dev_output};

    CheckCudaError(cuLaunchKernel(cuf_median_,
                                  grid_dim.x, grid_dim.y, grid_dim.z,
                                  block_dim.x, block_dim.y, block_dim.z,
                                  needed_shared_memory_size,
                                  NULL,
                                  args,
                                  NULL));
  } else {
    std::printf("Error. Wrong median raduis (%d). Supported values: 3, 5, 7\n", radius);
  }
}