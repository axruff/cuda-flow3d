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

#ifndef GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_E_H_
#define GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_E_H_

#include <forward_list>
#include <stack>

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/cuda_operations/entire_data/cuda_operation_add.h"
#include "src/cuda_operations/entire_data/cuda_operation_convolution.h"
#include "src/cuda_operations/entire_data/cuda_operation_median.h"
#include "src/cuda_operations/entire_data/cuda_operation_registration.h"
#include "src/cuda_operations/entire_data/cuda_operation_resample.h"
#include "src/cuda_operations/entire_data/cuda_operation_solve.h"

#include "src/data_types/operation_parameters.h"
#include "src/data_types/data3d.h"

#include "src/optical_flow/optical_flow_base.h"

class OpticalFlowE : public OpticalFlowBase {
private:
  const size_t dev_containers_count_ = 15;
  DataSize4 dev_container_size_;

  std::forward_list<CudaOperationBase*> cuda_operations_;
  std::stack<CUdeviceptr> cuda_memory_ptrs_;

  CudaOperationAdd cuop_add_;
  CudaOperationMedian cuop_median_;
  CudaOperationConvolution3D cuop_convolution_;
  CudaOperationRegistration cuop_register_;
  CudaOperationResample cuop_resample_;
  CudaOperationSolve cuop_solve_;

  bool InitCudaOperations();
  bool InitCudaMemory();

public:
  OpticalFlowE();

  bool Initialize(const DataSize4& data_size) override;
  void ComputeFlow(Data3D& frame_0, Data3D& frame_1, Data3D& flow_u, Data3D& flow_v, Data3D&flow_w, OperationParameters& params) override;
  void Destroy() override;

  bool silent = false;

  ~OpticalFlowE() override;
};


#endif // !GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_E_H_
