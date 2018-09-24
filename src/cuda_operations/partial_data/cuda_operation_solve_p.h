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

#ifndef GPUFLOW3D_CUDA_OPERATIONS_PARTIAL_DATA_CUDA_OPERATION_SOLVE_P_H_
#define GPUFLOW3D_CUDA_OPERATIONS_PARTIAL_DATA_CUDA_OPERATION_SOLVE_P_H_

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/data_types/data3d.h"
#include "src/data_types/data_structs.h"

class CudaOperationSolveP : public CudaOperationBase {
private:
  CUfunction cuf_compute_phi_ksi_p_;
  CUfunction cuf_solve_p_;

  int alignment_;
  size_t free_memory_;

  void ComputePhiKsi(Data3D& frame_0, Data3D& frame_1,
                     Data3D& flow_u, Data3D& flow_v, Data3D& flow_w,
                     Data3D& flow_du, Data3D& flow_dv, Data3D& flow_dw,
                     DataSize4 data_size,
                     float hx, float hy, float hz,
                     float equation_smoothness, float equation_data,
                     Data3D& phi, Data3D& ksi);

  void Solve(Data3D& frame_0, Data3D& frame_1,
             Data3D& flow_u, Data3D& flow_v, Data3D& flow_w,
             Data3D& flow_du, Data3D& flow_dv, Data3D& flow_dw,
             Data3D& phi, Data3D& ksi,
             DataSize4 data_size,
             float hx, float hy, float hz,
             float equation_alpha,
             Data3D& flow_ouptut_du, Data3D& flow_output_dv, Data3D& flow_output_dw);

public:
  CudaOperationSolveP();

  bool Initialize(const OperationParameters* params = nullptr) override;
  void Execute(OperationParameters& params) override;

  bool silent = false;
};


#endif // !GPUFLOW3D_CUDA_OPERATIONS_PARTIAL_DATA_CUDA_OPERATION_SOLVE_P_H_
