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

#ifndef GPUFLOW3D_CUDA_OPERATIONS_ENTIRE_DATA_CUDA_OPERATION_SOLVE_H_
#define GPUFLOW3D_CUDA_OPERATIONS_ENTIRE_DATA_CUDA_OPERATION_SOLVE_H_

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/data_types/data_structs.h"

class CudaOperationSolve : public CudaOperationBase {
private:
  CUfunction cuf_compute_phi_ksi_;
  CUfunction cuf_solve_;

  DataSize4 dev_container_size_;

public:
  CudaOperationSolve();

  bool Initialize(const OperationParameters* params = nullptr) override;
  void Execute(OperationParameters& params) override;
};

#endif // !GPUFLOW3D_CUDA_OPERATIONS_ENTIRE_DATA_CUDA_OPERATION_SOLVE_H_
