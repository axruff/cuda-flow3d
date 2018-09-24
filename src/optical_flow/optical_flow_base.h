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

#ifndef GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_BASE_H_
#define GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_BASE_H_

#include "src/data_types/data3d.h"
#include "src/data_types/data_structs.h"
#include "src/data_types/operation_parameters.h"

class OpticalFlowBase {
private:
  const char* name_ = nullptr;

protected:
  bool initialized_ = false;

  OpticalFlowBase(const char* name);

  size_t GetMaxWarpLevel(size_t width, size_t height, size_t depth, float scale_factor) const;

  bool IsInitialized() const;

public:
  const char* GetName() const;

  virtual bool Initialize(const DataSize4& data_size) = 0;
  virtual void ComputeFlow(Data3D& frame_0, Data3D& frame_1, Data3D& flow_u, Data3D& flow_v, Data3D&flow_w, OperationParameters& params);
  virtual void Destroy();

  virtual ~OpticalFlowBase();
};



#endif // !GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_BASE_H_
