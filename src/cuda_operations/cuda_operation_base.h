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

#ifndef GPUFLOW3D_CUDA_OPERATIONS_CUDA_OPERATION_BASE_H_
#define GPUFLOW3D_CUDA_OPERATIONS_CUDA_OPERATION_BASE_H_

#include <cuda.h>

#include "src/data_types/operation_parameters.h"

class CudaOperationBase {
private:
  const char* name_ = nullptr;

protected:
  CUmodule cu_module_ = nullptr;

  CUdeviceptr dev_constants_ = 0;

  bool initialized_ = false;

  CudaOperationBase(const char* name);

  bool IsInitialized() const;

public:
  const char* GetName() const;
  
  virtual bool Initialize(const OperationParameters* params = nullptr) = 0;
  virtual void Execute(OperationParameters& params);
  virtual void Destroy();

  virtual ~CudaOperationBase();
  
};

#endif // !GPUFLOW3D_CUDA_OPERATIONS_CUDA_OPERATION_BASE_H_
