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

#ifndef GPUFLOW3D_CUDA_OPERATIONS_PARTIAL_DATA_CUDA_OPERATION_STAT_P_H_
#define GPUFLOW3D_CUDA_OPERATIONS_PARTIAL_DATA_CUDA_OPERATION_STAT_P_H_

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/data_types/operation_parameters.h"

class CudaOperationStatP : public CudaOperationBase {
private:
  CUfunction cuf_stat_;

  const int alignment_ = 128 / sizeof(float);
  size_t free_memory_;

public:
  CudaOperationStatP();

  bool Initialize(const OperationParameters* params = nullptr) override;
  void Execute(OperationParameters& params) override;
};

#endif // !GPUFLOW3D_CUDA_OPERATIONS_PARTIAL_DATA_CUDA_OPERATION_STAT_P_H_
