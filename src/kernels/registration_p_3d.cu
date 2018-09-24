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
#include <device_functions.h>
#include <math_functions.h>

#include "src/data_types/data_structs.h"

#define IND(X, Y, Z) (((Z) * container_size.height + (Y)) * (container_size.pitch / sizeof(float)) + (X)) 

__constant__ DataSize4 container_size;

/* Declare texture references */
texture<float, cudaTextureType3D, cudaReadModeElementType> t_frame_0;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_frame_1;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_flow_u;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_flow_v;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_flow_w;


extern "C" __global__ void registration_p_3d(
        size_t width,
        size_t height,
        size_t depth,
        size_t container_depth,
        size_t chunk_depth,
        size_t chunk_num,
        size_t max_mag,
        float  hx,
        float  hy,
        float  hz,
        float* output)
{
  dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y,
                blockDim.z * blockIdx.z + threadIdx.z);

  if (global_id.x < width && global_id.y < height && global_id.z < chunk_depth) {
    size_t global_z = chunk_num * container_depth + global_id.z;
    float x_f = global_id.x + (tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + max_mag)) * (1.f / hx));
    float y_f = global_id.y + (tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + max_mag)) * (1.f / hy));
    
    float dz  = (tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + max_mag)) * (1.f / hz));
    float z_f = global_z + dz;

    if ((x_f < 0.) || (x_f > width - 1) || (y_f < 0.) || (y_f > height - 1) || (z_f < 0.) || (z_f > depth - 1) || 
      isnan(x_f) || isnan(y_f) || isnan(z_f)) { 
      output[IND(global_id.x, global_id.y, global_id.z)] = tex3D(t_frame_0, global_id.x, global_id.y, (global_id.z + max_mag));
    } else {
      output[IND(global_id.x, global_id.y, global_id.z)] = tex3D(t_frame_1, x_f + 0.5f, y_f + 0.5f, (global_id.z + max_mag) + dz + 0.5f);
    }
  }
}