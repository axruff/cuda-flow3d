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

#include "src/optical_flow/optical_flow_base.h"

OpticalFlowBase::OpticalFlowBase(const char* name)
  : name_(name)
{
}

const char* OpticalFlowBase::GetName() const
{
  return name_;
}

size_t OpticalFlowBase::GetMaxWarpLevel(size_t width, size_t height, size_t depth, float scale_factor) const
{
  /* Compute maximum number of warping levels for given image size and warping reduction factor */
  size_t r_width = 1;
  size_t r_height = 1;
  size_t r_depth = 1;
  size_t level_counter = 1;

  while (scale_factor < 1.f) {
    float scale = std::pow(scale_factor, static_cast<float>(level_counter));
    r_width = static_cast<size_t>(std::ceil(width * scale));
    r_height = static_cast<size_t>(std::ceil(height * scale));
    r_depth = static_cast<size_t>(std::ceil(depth * scale));

    if (r_width < 4 || r_height < 4 || r_depth < 4) {
      break;
    }
    ++level_counter;
  }

  if (r_width == 1 || r_height == 1 || r_depth == 1) {
    --level_counter;
  }

  return level_counter;
}



bool OpticalFlowBase::IsInitialized() const
{
  if (!initialized_) {
    std::printf("Error: '%s' was not initialized.\n", name_);
  }
  return initialized_;
}

void OpticalFlowBase::ComputeFlow(Data3D& frame_0, Data3D& frame_1, Data3D& flow_u, Data3D& flow_v, Data3D&flow_w, OperationParameters& params)
{
  std::printf("Warning: '%s' ComputeFlow() was not defined.\n", name_);

}

void OpticalFlowBase::Destroy()
{
  initialized_ = false;
}

OpticalFlowBase::~OpticalFlowBase()
{
  if (initialized_) {
    Destroy();
  }
}
