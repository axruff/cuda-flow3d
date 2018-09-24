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

#ifndef GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_P_H_
#define GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_P_H_

#include <forward_list>


#include "src/cuda_operations/cuda_operation_base.h"
#include "src/cuda_operations/partial_data/cuda_operation_add_p.h"
#include "src/cuda_operations/partial_data/cuda_operation_register_p.h"
#include "src/cuda_operations/partial_data/cuda_operation_resample_p.h"
#include "src/cuda_operations/partial_data/cuda_operation_solve_p.h"
#include "src/cuda_operations/partial_data/cuda_operation_stat_p.h"

#include "src/data_types/data_structs.h"

#include "src/optical_flow/optical_flow_base.h"

class OpticalFlowP : public OpticalFlowBase {
private:
  DataSize4 data_size_;

  std::forward_list<CudaOperationBase*> cuda_operations_;

  CudaOperationRegistrationP cuop_register_p_;
  CudaOperationResampleP cuop_resample_p_;
  CudaOperationSolveP cuop_solve_p_;
  CudaOperationStatP cuop_stat_p_;
  CudaOperationAddP cuop_add_p_;

public:
  OpticalFlowP();

  bool Initialize(const DataSize4& data_size) override;
  void ComputeFlow(Data3D& frame_0, Data3D& frame_1, Data3D& flow_u, Data3D& flow_v, Data3D&flow_w, OperationParameters& params) override;
  void Destroy() override;

  bool silent = false;

  ~OpticalFlowP() override;
};
#endif // !GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_P_H_
