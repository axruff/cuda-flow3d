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

#define IIND(X, Y, Z) (((Z) * input_size.height + (Y)) * (input_size.pitch / sizeof(float)) + (X)) 
#define OIND(X, Y, Z) (((Z) * output_size.height + (Y)) * (output_size.pitch / sizeof(float)) + (X)) 

__constant__ DataSize4 input_size;
__constant__ DataSize4 output_size;

extern "C" __global__ void resample_x_p_3d(
  const float* input,
        float* output,
        size_t out_width,
        size_t out_height,
        size_t out_depth,
        size_t in_width)
{
  dim3 globalID(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y,
                blockDim.z * blockIdx.z + threadIdx.z);

  if (globalID.x < out_width &&  globalID.y < out_height && globalID.z < out_depth) {
    float delta = in_width / static_cast<float>(out_width);
    float normalization = out_width / static_cast<float>(in_width);

    float left_f = globalID.x * delta;
    float right_f = (globalID.x + 1) * delta;

    int left_i = static_cast<int>(floor(left_f));
    int right_i = min(in_width, static_cast<size_t>(ceil(right_f)));

    float value = 0.f;
    
    for (int j = 0; j < (right_i - left_i); j++) {
      float frac = 1.f;

      /* left boundary */
      if (j == 0) {
        frac = static_cast<float>(left_i + 1) - left_f;
      }
      /* right boundary */
      if (j == (right_i - left_i) - 1) {
        frac = right_f - static_cast<float>(left_i + j);
      }
      /* if the left and right boundaries are in the same cell */
      if ((right_i - left_i) == 1) {
        frac = delta;
      }
      value += input[IIND(left_i + j, globalID.y, globalID.z)] * frac;
    }
    output[OIND(globalID.x, globalID.y, globalID.z)] = value * normalization;
  }
}

extern "C" __global__ void resample_y_p_3d(
  const float* input,
        float* output,
        size_t out_width,
        size_t out_height,
        size_t out_depth,
        size_t in_height)
{
  dim3 globalID(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y,
                blockDim.z * blockIdx.z + threadIdx.z);

  if (globalID.x < out_width &&  globalID.y < out_height && globalID.z < out_depth) {
    float delta = in_height / static_cast<float>(out_height);
    float normalization = out_height / static_cast<float>(in_height);

    float left_f = globalID.y * delta;
    float right_f = (globalID.y + 1) * delta;

    int left_i = static_cast<int>(floor(left_f));
    int right_i = min(in_height, static_cast<size_t>(ceil(right_f)));

    float value = 0.f;
    
    for (int j = 0; j < (right_i - left_i); j++) {
      float frac = 1.f;

      /* left boundary */
      if (j == 0) {
        frac = static_cast<float>(left_i + 1) - left_f;
      }
      /* right boundary */
      if (j == (right_i - left_i) - 1) {
        frac = right_f - static_cast<float>(left_i + j);
      }
      /* if the left and right boundaries are in the same cell */
      if ((right_i - left_i) == 1) {
        frac = delta;
      }
      value += input[IIND(globalID.x, left_i + j, globalID.z)] * frac;
    }
    output[OIND(globalID.x, globalID.y, globalID.z)] = value * normalization;
  }
}

extern "C" __global__ void resample_z_p_3d(
  const float* input,
        float* output,
        size_t out_width,
        size_t out_height,
        size_t out_depth,
        size_t in_depth)
{
  dim3 globalID(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y,
                blockDim.z * blockIdx.z + threadIdx.z);

  if (globalID.x < out_width &&  globalID.y < out_height && globalID.z < out_depth) {
    float delta = in_depth / static_cast<float>(out_depth);
    float normalization = out_depth / static_cast<float>(in_depth);

    float left_f = globalID.z * delta;
    float right_f = (globalID.z + 1) * delta;

    int left_i = static_cast<int>(floor(left_f));
    int right_i = min(in_depth, static_cast<size_t>(ceil(right_f)));

    float value = 0.f;
    
    for (int j = 0; j < (right_i - left_i); j++) {
      float frac = 1.f;

      /* left boundary */
      if (j == 0) {
        frac = static_cast<float>(left_i + 1) - left_f;
      }
      /* right boundary */
      if (j == (right_i - left_i) - 1) {
        frac = right_f - static_cast<float>(left_i + j);
      }
      /* if the left and right boundaries are in the same cell */
      if ((right_i - left_i) == 1) {
        frac = delta;
      }
      value += input[IIND(globalID.x, globalID.y, left_i + j)] * frac;
    }
    output[OIND(globalID.x, globalID.y, globalID.z)] = value * normalization;
  }
}