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

#ifndef GPUFLOW3D_CUDA_OPERATIONS_ENTIRE_DATA_CUDA_OPERATION_RESAMPLE_H_
#define GPUFLOW3D_CUDA_OPERATIONS_ENTIRE_DATA_CUDA_OPERATION_RESAMPLE_H_

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/data_types/data_structs.h"

class CudaOperationResample : public CudaOperationBase {
private: 
  CUfunction cuf_resample_x_;
  CUfunction cuf_resample_y_;
  CUfunction cuf_resample_z_;

  void ResampleX(CUdeviceptr input, CUdeviceptr output, DataSize4& input_size, DataSize4& output_size) const;
  void ResampleY(CUdeviceptr input, CUdeviceptr output, DataSize4& input_size, DataSize4& output_size) const;
  void ResampleZ(CUdeviceptr input, CUdeviceptr output, DataSize4& input_size, DataSize4& output_size) const;

public:
  CudaOperationResample();

  bool Initialize(const OperationParameters* params = nullptr) override;
  void Execute(OperationParameters& params) override;
};

#endif // !GPUFLOW3D_CUDA_OPERATIONS_ENTIRE_DATA_CUDA_OPERATION_RESAMPLE_H_
