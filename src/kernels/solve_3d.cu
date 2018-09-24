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
#define SIND(X, Y, Z) ((((Z) + 1) * shared_block_size.y + ((Y) + 1)) * shared_block_size.x + ((X) + 1))

__constant__ DataSize4 container_size;

extern __shared__ float shared[];

extern "C" __global__ void compute_phi_ksi_3d(
  const float* frame_0,
  const float* frame_1,
  const float* flow_u,
  const float* flow_v,
  const float* flow_w,
  const float* flow_du,
  const float* flow_dv,
  const float* flow_dw,
        size_t width,
        size_t height,
        size_t depth,
        float  hx,
        float  hy,
        float  hz,
        float  equation_smootness,
        float  equation_data,
        float* phi,
        float* ksi)
{
  dim3 shared_block_size(
    blockDim.x + 2,
    blockDim.y + 2,
    blockDim.z + 2);

  float* shared_frame_0 = &shared[0 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_frame_1 = &shared[1 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_u  = &shared[2 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_v  = &shared[3 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_w  = &shared[4 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_du = &shared[5 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_dv = &shared[6 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_dw = &shared[7 * shared_block_size.x * shared_block_size.y * shared_block_size.z];

  dim3 global_id(
    blockDim.x * blockIdx.x + threadIdx.x,
    blockDim.y * blockIdx.y + threadIdx.y,
    blockDim.z * blockIdx.z + threadIdx.z);

  /* Load the main area of datasets */
  size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
  size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;
  size_t global_z = global_id.z < depth ? global_id.z : 2 * depth - global_id.z - 2;
  {
    shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = frame_0[IND(global_x, global_y, global_z)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = frame_1[IND(global_x, global_y, global_z)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_u [IND(global_x, global_y, global_z)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_v [IND(global_x, global_y, global_z)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_w [IND(global_x, global_y, global_z)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_du[IND(global_x, global_y, global_z)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_dv[IND(global_x, global_y, global_z)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_dw[IND(global_x, global_y, global_z)];
  }

  /* Load the left slice */
  if (threadIdx.x == 0) {
    int offset = global_x - 1;
    size_t global_x_l = offset >= 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = frame_0[IND(global_x_l, global_y, global_z)];
    shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = frame_1[IND(global_x_l, global_y, global_z)];
    shared_flow_u [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_u [IND(global_x_l, global_y, global_z)];
    shared_flow_v [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_v [IND(global_x_l, global_y, global_z)];
    shared_flow_w [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_w [IND(global_x_l, global_y, global_z)];
    shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_du[IND(global_x_l, global_y, global_z)];
    shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_dv[IND(global_x_l, global_y, global_z)];
    shared_flow_dw[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_dw[IND(global_x_l, global_y, global_z)];

  }

  /* Load the right slice */
  if (threadIdx.x == blockDim.x - 1) {
    int offset = global_x + 1;
    size_t global_x_r = offset < width ? offset : 2 * width - offset - 2;
    shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = frame_0[IND(global_x_r, global_y, global_z)];
    shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = frame_1[IND(global_x_r, global_y, global_z)];
    shared_flow_u [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_u [IND(global_x_r, global_y, global_z)];
    shared_flow_v [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_v [IND(global_x_r, global_y, global_z)];
    shared_flow_w [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_w [IND(global_x_r, global_y, global_z)];
    shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_du[IND(global_x_r, global_y, global_z)];
    shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_dv[IND(global_x_r, global_y, global_z)];
    shared_flow_dw[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_dw[IND(global_x_r, global_y, global_z)];
  }

  /* Load the upper slice */
  if (threadIdx.y == 0) {
    int offset = global_y - 1;
    size_t global_y_u = offset > 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = frame_0[IND(global_x, global_y_u, global_z)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = frame_1[IND(global_x, global_y_u, global_z)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_u [IND(global_x, global_y_u, global_z)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_v [IND(global_x, global_y_u, global_z)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_w [IND(global_x, global_y_u, global_z)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_du[IND(global_x, global_y_u, global_z)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_dv[IND(global_x, global_y_u, global_z)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_dw[IND(global_x, global_y_u, global_z)];
  }

  /* Load the bottom slice */
  if (threadIdx.y == blockDim.y - 1) {
    int offset = global_y + 1;
    size_t global_y_b = offset < height ? offset : 2 * height - offset - 2;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = frame_0[IND(global_x, global_y_b, global_z)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = frame_1[IND(global_x, global_y_b, global_z)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_u [IND(global_x, global_y_b, global_z)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_v [IND(global_x, global_y_b, global_z)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_w [IND(global_x, global_y_b, global_z)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_du[IND(global_x, global_y_b, global_z)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_dv[IND(global_x, global_y_b, global_z)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_dw[IND(global_x, global_y_b, global_z)];
  }

  /* Load the front slice */
  if (threadIdx.z == 0) {
    int offset = global_z - 1;
    size_t global_z_f = offset > 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = frame_0[IND(global_x, global_y, global_z_f)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = frame_1[IND(global_x, global_y, global_z_f)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_u [IND(global_x, global_y, global_z_f)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_v [IND(global_x, global_y, global_z_f)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_w [IND(global_x, global_y, global_z_f)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_du[IND(global_x, global_y, global_z_f)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_dv[IND(global_x, global_y, global_z_f)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_dw[IND(global_x, global_y, global_z_f)];
  }

  /* Load the rear slice */
  if (threadIdx.z == blockDim.z - 1) {
    int offset = global_z + 1;
    size_t global_z_r = offset < depth ? offset : 2 * depth - offset - 2;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = frame_0[IND(global_x, global_y, global_z_r)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = frame_1[IND(global_x, global_y, global_z_r)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_u [IND(global_x, global_y, global_z_r)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_v [IND(global_x, global_y, global_z_r)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_w [IND(global_x, global_y, global_z_r)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_du[IND(global_x, global_y, global_z_r)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_dv[IND(global_x, global_y, global_z_r)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_dw[IND(global_x, global_y, global_z_r)];
  }

  __syncthreads();

  /* Compute flow-driven terms */
  if (global_id.x < width && global_id.y < height && global_id.z < depth) {

    float dux =
      (shared_flow_u [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_u [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] +
       shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)]) /
      (2.f * hx);
    float duy =
      (shared_flow_u [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_u [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] +
       shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)]) /
      (2.f * hy);
    float duz = 
      (shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] +
       shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)]) /
      (2.f * hz);

    float dvx =
      (shared_flow_v [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_v [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] +
       shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)]) /
      (2.f * hx);
    float dvy =
      (shared_flow_v [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_v [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] +
       shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)]) /
      (2.f * hy);
    float dvz = 
      (shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] +
       shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)]) /
      (2.f * hz);

    float dwx =
      (shared_flow_w [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_w [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] +
       shared_flow_dw[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_dw[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)]) /
      (2.f * hx);
    float dwy =
      (shared_flow_w [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_w [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] +
       shared_flow_dw[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_dw[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)]) /
      (2.f * hy);
    float dwz = 
      (shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] +
       shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)]) /
      (2.f * hz);

    /* Flow-driven term */
    phi[IND(global_id.x, global_id.y, global_id.z)] =
      1.f / (2.f * sqrtf(dux*dux + duy*duy + duz*duz + dvx*dvx + dvy*dvy + dvz*dvz + dwx*dwx + dwy*dwy + dwz*dwz + equation_smootness * equation_smootness));

    float fx = 
      (shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] +
       shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)]) / 
       (4.f * hx);
    float fy = 
      (shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] +
       shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)]) / 
       (4.f * hy);
    float fz = 
      (shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] +
       shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)]) / 
       (4.f * hz);
    float ft = 
      shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z)];

    float J11 = fx * fx;
    float J22 = fy * fy;
    float J33 = fz * fz;
    float J12 = fx * fy;
    float J13 = fx * fz;
    float J23 = fy * fz;
    float J14 = fx * ft;
    float J24 = fy * ft;
    float J34 = fz * ft;
    float J44 = ft * ft;

    float& du = shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z)];
    float& dv = shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z)];
    float& dw = shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z)];

    float s =
      (J11 * du + J12 * dv + J13 * dw + J14) * du +
      (J12 * du + J22 * dv + J23 * dw + J24) * dv +
      (J13 * du + J23 * dv + J33 * dw + J34) * dw +
      (J14 * du + J24 * dv + J34 * dw + J44);

    s = (s > 0) * s;

    /* Penalizer function for the data term */
    ksi[IND(global_id.x, global_id.y, global_id.z)] =
      1.f / (2.f * sqrtf(s + equation_data * equation_data));
  }
}

extern "C" __global__ void solve_3d(
  const float* frame_0,
  const float* frame_1,
  const float* flow_u,
  const float* flow_v,
  const float* flow_w,
  const float* flow_du,
  const float* flow_dv,
  const float* flow_dw,
  const float* phi,
  const float* ksi,
        size_t width,
        size_t height,
        size_t depth,
        float  hx,
        float  hy,
        float  hz,
        float  equation_alpha,
        float* temp_du,
        float* temp_dv,
        float* temp_dw)
{
  dim3 shared_block_size(
    blockDim.x + 2,
    blockDim.y + 2,
    blockDim.z + 2);

  float* shared_frame_0 = &shared[0 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_frame_1 = &shared[1 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_u  = &shared[2 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_v  = &shared[3 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_w  = &shared[4 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_du = &shared[5 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_dv = &shared[6 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_flow_dw = &shared[7 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_phi     = &shared[8 * shared_block_size.x * shared_block_size.y * shared_block_size.z];
  float* shared_ksi     = &shared[9 * shared_block_size.x * shared_block_size.y * shared_block_size.z];


  dim3 global_id(
    blockDim.x * blockIdx.x + threadIdx.x,
    blockDim.y * blockIdx.y + threadIdx.y,
    blockDim.z * blockIdx.z + threadIdx.z);

  /* Load the main area of datasets */
  size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
  size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;
  size_t global_z = global_id.z < depth ? global_id.z : 2 * depth - global_id.z - 2;
  {
    shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = frame_0[IND(global_x, global_y, global_z)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = frame_1[IND(global_x, global_y, global_z)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_u [IND(global_x, global_y, global_z)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_v [IND(global_x, global_y, global_z)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_w [IND(global_x, global_y, global_z)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_du[IND(global_x, global_y, global_z)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_dv[IND(global_x, global_y, global_z)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] = flow_dw[IND(global_x, global_y, global_z)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] =     phi[IND(global_x, global_y, global_z)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y, threadIdx.z)] =     ksi[IND(global_x, global_y, global_z)];
  }

  /* Load the left slice */
  if (threadIdx.x == 0) {
    int offset = global_x - 1;
    size_t global_x_l = offset >= 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = frame_0[IND(global_x_l, global_y, global_z)];
    shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = frame_1[IND(global_x_l, global_y, global_z)];
    shared_flow_u [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_u [IND(global_x_l, global_y, global_z)];
    shared_flow_v [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_v [IND(global_x_l, global_y, global_z)];
    shared_flow_w [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_w [IND(global_x_l, global_y, global_z)];
    shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_du[IND(global_x_l, global_y, global_z)];
    shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_dv[IND(global_x_l, global_y, global_z)];
    shared_flow_dw[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] = flow_dw[IND(global_x_l, global_y, global_z)];
    shared_phi    [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] =     phi[IND(global_x_l, global_y, global_z)];
    shared_ksi    [SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] =     ksi[IND(global_x_l, global_y, global_z)];
  }

  /* Load the right slice */
  if (threadIdx.x == blockDim.x - 1) {
    int offset = global_x + 1;
    size_t global_x_r = offset < width ? offset : 2 * width - offset - 2;
    shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = frame_0[IND(global_x_r, global_y, global_z)];
    shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = frame_1[IND(global_x_r, global_y, global_z)];
    shared_flow_u [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_u [IND(global_x_r, global_y, global_z)];
    shared_flow_v [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_v [IND(global_x_r, global_y, global_z)];
    shared_flow_w [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_w [IND(global_x_r, global_y, global_z)];
    shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_du[IND(global_x_r, global_y, global_z)];
    shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_dv[IND(global_x_r, global_y, global_z)];
    shared_flow_dw[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] = flow_dw[IND(global_x_r, global_y, global_z)];
    shared_phi    [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] =     phi[IND(global_x_r, global_y, global_z)];
    shared_ksi    [SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] =     ksi[IND(global_x_r, global_y, global_z)];
  }

  /* Load the upper slice */
  if (threadIdx.y == 0) {
    int offset = global_y - 1;
    size_t global_y_u = offset > 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = frame_0[IND(global_x, global_y_u, global_z)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = frame_1[IND(global_x, global_y_u, global_z)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_u [IND(global_x, global_y_u, global_z)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_v [IND(global_x, global_y_u, global_z)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_w [IND(global_x, global_y_u, global_z)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_du[IND(global_x, global_y_u, global_z)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_dv[IND(global_x, global_y_u, global_z)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] = flow_dw[IND(global_x, global_y_u, global_z)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] =     phi[IND(global_x, global_y_u, global_z)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] =     ksi[IND(global_x, global_y_u, global_z)];
  }

  /* Load the bottom slice */
  if (threadIdx.y == blockDim.y - 1) {
    int offset = global_y + 1;
    size_t global_y_b = offset < height ? offset : 2 * height - offset - 2;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = frame_0[IND(global_x, global_y_b, global_z)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = frame_1[IND(global_x, global_y_b, global_z)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_u [IND(global_x, global_y_b, global_z)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_v [IND(global_x, global_y_b, global_z)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_w [IND(global_x, global_y_b, global_z)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_du[IND(global_x, global_y_b, global_z)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_dv[IND(global_x, global_y_b, global_z)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] = flow_dw[IND(global_x, global_y_b, global_z)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] =     phi[IND(global_x, global_y_b, global_z)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] =     ksi[IND(global_x, global_y_b, global_z)];
  }

  /* Load the front slice */
  if (threadIdx.z == 0) {
    int offset = global_z - 1;
    size_t global_z_f = offset > 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = frame_0[IND(global_x, global_y, global_z_f)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = frame_1[IND(global_x, global_y, global_z_f)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_u [IND(global_x, global_y, global_z_f)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_v [IND(global_x, global_y, global_z_f)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_w [IND(global_x, global_y, global_z_f)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_du[IND(global_x, global_y, global_z_f)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_dv[IND(global_x, global_y, global_z_f)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] = flow_dw[IND(global_x, global_y, global_z_f)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] =     phi[IND(global_x, global_y, global_z_f)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] =     ksi[IND(global_x, global_y, global_z_f)];
  }

  /* Load the rear slice */
  if (threadIdx.z == blockDim.z - 1) {
    int offset = global_z + 1;
    size_t global_z_r = offset < depth ? offset : 2 * depth - offset - 2;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = frame_0[IND(global_x, global_y, global_z_r)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = frame_1[IND(global_x, global_y, global_z_r)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_u [IND(global_x, global_y, global_z_r)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_v [IND(global_x, global_y, global_z_r)];
    shared_flow_w [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_w [IND(global_x, global_y, global_z_r)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_du[IND(global_x, global_y, global_z_r)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_dv[IND(global_x, global_y, global_z_r)];
    shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] = flow_dw[IND(global_x, global_y, global_z_r)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] =     phi[IND(global_x, global_y, global_z_r)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] =     ksi[IND(global_x, global_y, global_z_r)];
  }

  __syncthreads();

  if (global_id.x < width && global_id.y < height && global_id.z < depth) {
    /* Compute derivatives */
    float fx = 
      (shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] +
       shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)]) / 
       (4.f * hx);
    float fy = 
      (shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] +
       shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)]) / 
       (4.f * hy);
    float fz = 
      (shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] +
       shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)]) / 
       (4.f * hz);
    float ft = 
      shared_frame_1[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y, threadIdx.z)];

    float J11 = fx * fx;
    float J22 = fy * fy;
    float J33 = fz * fz;
    float J12 = fx * fy;
    float J13 = fx * fz;
    float J23 = fy * fz;
    float J14 = fx * ft;
    float J24 = fy * ft;
    float J34 = fz * ft;
 
    /* Compute weights */
    float hx_2 = equation_alpha / (hx * hx);
    float hy_2 = equation_alpha / (hy * hy);
    float hz_2 = equation_alpha / (hz * hz);
    
    float xp = (global_id.x < width - 1)  * hx_2;
    float xm = (global_id.x > 0)          * hx_2;
    float yp = (global_id.y < height - 1) * hy_2;
    float ym = (global_id.y > 0)          * hy_2;
    float zp = (global_id.z < depth - 1)  * hz_2;
    float zm = (global_id.z > 0)          * hz_2;

    float phi_xp = (shared_phi[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] + shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) / 2.f;
    float phi_xm = (shared_phi[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] + shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) / 2.f;
    float phi_yp = (shared_phi[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] + shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) / 2.f;
    float phi_ym = (shared_phi[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] + shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) / 2.f;
    float phi_zp = (shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] + shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) / 2.f;
    float phi_zm = (shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] + shared_phi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) / 2.f;

    float sumH = (xp*phi_xp + xm*phi_xm + yp*phi_yp + ym*phi_ym + zp*phi_zp + zm*phi_zm);
    float sumU =
      phi_xp * xp * (shared_flow_u[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] + shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_xm * xm * (shared_flow_u[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] + shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_yp * yp * (shared_flow_u[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] + shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_ym * ym * (shared_flow_u[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] + shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_zp * zp * (shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] + shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_zm * zm * (shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] + shared_flow_du[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]);
    float sumV =
      phi_xp * xp * (shared_flow_v[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] + shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_xm * xm * (shared_flow_v[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] + shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_yp * yp * (shared_flow_v[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] + shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_ym * ym * (shared_flow_v[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] + shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_zp * zp * (shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] + shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_zm * zm * (shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] + shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]);
    float sumW =
      phi_xp * xp * (shared_flow_w[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] + shared_flow_dw[SIND(threadIdx.x + 1, threadIdx.y, threadIdx.z)] - shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_xm * xm * (shared_flow_w[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] + shared_flow_dw[SIND(threadIdx.x - 1, threadIdx.y, threadIdx.z)] - shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_yp * yp * (shared_flow_w[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] + shared_flow_dw[SIND(threadIdx.x, threadIdx.y + 1, threadIdx.z)] - shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_ym * ym * (shared_flow_w[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] + shared_flow_dw[SIND(threadIdx.x, threadIdx.y - 1, threadIdx.z)] - shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_zp * zp * (shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] + shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z + 1)] - shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) +
      phi_zm * zm * (shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] + shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z - 1)] - shared_flow_w[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]);

    float result_du =
      (shared_ksi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] * (-J14 - J12 * shared_flow_dv[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] - J13 * shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) + sumU) /
      (shared_ksi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] * J11 + sumH);

    float result_dv =
      (shared_ksi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] * (-J24 - J12 * result_du - J23 * shared_flow_dw[SIND(threadIdx.x, threadIdx.y, threadIdx.z)]) + sumV) /
      (shared_ksi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] * J22 + sumH);

    float result_dw =
      (shared_ksi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] * (-J34 - J13 * result_du - J23 * result_dv) + sumW) /
      (shared_ksi[SIND(threadIdx.x, threadIdx.y, threadIdx.z)] * J33 + sumH);

    temp_du[IND(global_id.x, global_id.y, global_id.z)] = result_du;
    temp_dv[IND(global_id.x, global_id.y, global_id.z)] = result_dv;
    temp_dw[IND(global_id.x, global_id.y, global_id.z)] = result_dw;
  }
}