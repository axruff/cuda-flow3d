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

#include "src/cuda_operations/partial_data/cuda_operation_stat_p.h"

#include <algorithm>

#include "src/data_types/data3d.h"
#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationStatP::CudaOperationStatP()
  : CudaOperationBase("CUDA Stat Piecemeal")
{
}

bool CudaOperationStatP::Initialize(const OperationParameters* params)
{
  initialized_ = false;

  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/add.ptx");

  //if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
  //  if (!CheckCudaError(cuModuleGetFunction(&cuf_stat_, cu_module_, "add"))) {
  //    initialized_ = true;
  //  } else {
  //    CheckCudaError(cuModuleUnload(cu_module_));
  //    cu_module_ = nullptr;
  //  }
  //}

  initialized_ = true;

  //??

  return initialized_;
}

void CudaOperationStatP::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  Data3D* p_flow_u;
  Data3D* p_flow_v;
  Data3D* p_flow_w;

  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_u, "flow_u");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_v, "flow_v");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_w, "flow_w");

  Data3D& flow_u = *p_flow_u;
  Data3D& flow_v = *p_flow_v;
  Data3D& flow_w = *p_flow_w;

  DataSize4 data_size;

  GET_PARAM_OR_RETURN(params, DataSize4, data_size, "data_size");

  Stat3* p_stat;
  GET_PARAM_PTR_OR_RETURN(params, Stat3, p_stat, "stat");


  std::printf("Compute statistics...\n");

  p_stat->min = std::numeric_limits<float>::max();
  p_stat->max = std::numeric_limits<float>::min();
  p_stat->avg = 0.f;

  for (size_t z = 0; z < data_size.depth; ++z) {
    for (size_t y = 0; y < data_size.height; ++y) {
      for (size_t x = 0; x < data_size.width; ++x) {
        float vec[3] = { flow_u.Data(x, y, z), flow_v.Data(x, y, z), flow_w.Data(x, y, z) };

        float magnitude = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);

        p_stat->min = std::fmin(p_stat->min, magnitude);
        p_stat->max = std::fmax(p_stat->max, magnitude);
        p_stat->avg += magnitude;
      }
    }
  }
  p_stat->avg /= static_cast<float>(data_size.width * data_size.height * data_size.depth);

  std::printf("Min: %8.4f Max: %8.4f Avg: %8.4f\n", p_stat->min, p_stat->max, p_stat->avg);

}