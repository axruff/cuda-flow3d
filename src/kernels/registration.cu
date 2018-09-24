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

#include <device_launch_parameters.h>

#define __CUDACC__
#include <math_functions.h>

#include "src/data_types/data_structs.h"

#define IND(X, Y, Z) (((Z) * container_size.height + (Y)) * (container_size.pitch / sizeof(float)) + (X)) 

__constant__ DataSize4 container_size;

extern "C" __global__ void registration(
  const float* frame_0,
  const float* frame_1,
  const float* flow_u,
  const float* flow_v,
  const float* flow_w,
        size_t width,
        size_t height,
        size_t depth,
        float  hx,
        float  hy,
        float  hz,
        float* output)
{
  dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y,
                blockDim.z * blockIdx.z + threadIdx.z);

  if (global_id.x < width && global_id.y < height && global_id.z < depth) {
    float x_f = global_id.x + (flow_u[IND(global_id.x, global_id.y, global_id.z)] * (1.f / hx));
    float y_f = global_id.y + (flow_v[IND(global_id.x, global_id.y, global_id.z)] * (1.f / hy));
    float z_f = global_id.z + (flow_w[IND(global_id.x, global_id.y, global_id.z)] * (1.f / hz));

    if ((x_f < 0.) || (x_f > width - 1) || (y_f < 0.) || (y_f > height - 1) || (z_f < 0.) || (z_f > depth - 1) || 
      isnan(x_f) || isnan(y_f) || isnan(z_f)) { 
      output[IND(global_id.x, global_id.y, global_id.z)] = frame_0[IND(global_id.x, global_id.y, global_id.z)];
    } else {
      int x = (int) floorf(x_f); 
      int y = (int) floorf(y_f); 
      int z = (int) floorf(z_f); 
      float delta_x = x_f - (float) x;
      float delta_y = y_f - (float) y;
      float delta_z = z_f - (float) z;

      int x_1 = min(width -1, size_t(x + 1));
      int y_1 = min(height - 1, size_t(y + 1));
      int z_1 = min(depth - 1, size_t(z + 1));

      float value_0 =
        (1.f - delta_x) * (1.f - delta_y) * frame_1[IND(x  , y  , z  )] +
        (      delta_x) * (1.f - delta_y) * frame_1[IND(x_1, y  , z  )] +
        (1.f - delta_x) * (      delta_y) * frame_1[IND(x  , y_1, z  )] +
        (      delta_x) * (      delta_y) * frame_1[IND(x_1, y_1, z  )];

      float value_1 =
        (1.f - delta_x) * (1.f - delta_y) * frame_1[IND(x  , y  , z_1)] +
        (      delta_x) * (1.f - delta_y) * frame_1[IND(x_1, y  , z_1)] +
        (1.f - delta_x) * (      delta_y) * frame_1[IND(x  , y_1, z_1)] +
        (      delta_x) * (      delta_y) * frame_1[IND(x_1, y_1, z_1)];

      output[IND(global_id.x, global_id.y, global_id.z)] =
        (1.f - delta_z) * value_0 + delta_z * value_1;
    }
  }
}