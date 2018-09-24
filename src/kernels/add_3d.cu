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
#include <vector_types.h>

#include "src/data_types/data_structs.h"

#define IND(X, Y, Z) (((Z) * container_size.height + (Y)) * (container_size.pitch / sizeof(float)) + (X)) 

__constant__ DataSize4 container_size;

extern "C" __global__ void add_3d(
        float* operand_0,
  const float* operand_1,
        size_t width,
        size_t height,
        size_t depth)
{
  dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y,
                blockDim.z * blockIdx.z + threadIdx.z);

  if (global_id.x < width && global_id.y < height && global_id.z < depth) {
    operand_0[IND(global_id.x, global_id.y, global_id.z)] +=
      operand_1[IND(global_id.x, global_id.y, global_id.z)];
  }
}
