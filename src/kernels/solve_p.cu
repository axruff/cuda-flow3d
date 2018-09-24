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
texture<float, cudaTextureType3D, cudaReadModeElementType> t_flow_du;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_flow_dv;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_flow_dw;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_phi;
texture<float, cudaTextureType3D, cudaReadModeElementType> t_ksi;


extern "C" __global__ void compute_phi_ksi_p(
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
  dim3 global_id(
    blockDim.x * blockIdx.x + threadIdx.x,
    blockDim.y * blockIdx.y + threadIdx.y,
    blockDim.z * blockIdx.z + threadIdx.z);

  /* Texture access:                             */
  /* _m_o - minus one; _p_o - plus one           */
  /* x = x;        x - 1 = x_m_o;  x + 1 = x_p_o */
  /* y = y;        y - 1 = y_m_o;  y + 1 = y_p_o */
  /* z = (z + 1);  z - 1 = z;      z + 1 = z + 2 */

  if (global_id.x < width && global_id.y < height && global_id.z < depth) {
    unsigned int x_m_o = global_id.x == 0 ? 1 : global_id.x - 1;
    unsigned int x_p_o = global_id.x == width - 1 ? width - 2 : global_id.x + 1;

    unsigned int y_m_o = global_id.y == 0 ? 1 : global_id.y - 1;
    unsigned int y_p_o = global_id.y == height - 1 ? height - 2 : global_id.y + 1;

    float dux =
      (tex3D(t_flow_u , x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_u , x_m_o, global_id.y, (global_id.z + 1)) +
       tex3D(t_flow_du, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_du, x_m_o, global_id.y, (global_id.z + 1))) /
      (2.f * hx);
    float duy =
      (tex3D(t_flow_u , global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_u , global_id.x, y_m_o, (global_id.z + 1)) +
       tex3D(t_flow_du, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_du, global_id.x, y_m_o, (global_id.z + 1))) /
      (2.f * hy);
    float duz = 
      (tex3D(t_flow_u , global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_u , global_id.x, global_id.y, (global_id.z + 1) - 1) +
       tex3D(t_flow_du, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_du, global_id.x, global_id.y, (global_id.z + 1) - 1)) /
      (2.f * hz);

    float dvx =
      (tex3D(t_flow_v , x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_v , x_m_o, global_id.y, (global_id.z + 1)) +
       tex3D(t_flow_dv, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_dv, x_m_o, global_id.y, (global_id.z + 1))) /
      (2.f * hx);
    float dvy =
      (tex3D(t_flow_v , global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_v , global_id.x, y_m_o, (global_id.z + 1)) +
       tex3D(t_flow_dv, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_dv, global_id.x, y_m_o, (global_id.z + 1))) /
      (2.f * hy);
    float dvz = 
      (tex3D(t_flow_v , global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_v , global_id.x, global_id.y, (global_id.z + 1) - 1) +
       tex3D(t_flow_dv, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_dv, global_id.x, global_id.y, (global_id.z + 1) - 1)) /
      (2.f * hz);

    float dwx =
      (tex3D(t_flow_w , x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_w , x_m_o, global_id.y, (global_id.z + 1)) +
       tex3D(t_flow_dw, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_dw, x_m_o, global_id.y, (global_id.z + 1))) /
      (2.f * hx);
    float dwy =
      (tex3D(t_flow_w , global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_w , global_id.x, y_m_o, (global_id.z + 1)) +
       tex3D(t_flow_dw, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_dw, global_id.x, y_m_o, (global_id.z + 1))) /
      (2.f * hy);
    float dwz = 
      (tex3D(t_flow_w , global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_w , global_id.x, global_id.y, (global_id.z + 1) - 1) +
       tex3D(t_flow_dw, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_dw, global_id.x, global_id.y, (global_id.z + 1) - 1)) /
      (2.f * hz);

    /* Flow-driven term */
    phi[IND(global_id.x, global_id.y, global_id.z)] =
      1.f / (2.f * sqrtf(dux*dux + duy*duy + duz*duz + dvx*dvx + dvy*dvy + dvz*dvz + dwx*dwx + dwy*dwy + dwz*dwz + equation_smootness * equation_smootness));

    float fx = 
      (tex3D(t_frame_0, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_frame_0, x_m_o, global_id.y, (global_id.z + 1)) +
       tex3D(t_frame_1, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_frame_1, x_m_o, global_id.y, (global_id.z + 1))) / 
       (4.f * hx);
    float fy = 
      (tex3D(t_frame_0, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_frame_0, global_id.x, y_m_o, (global_id.z + 1)) +
       tex3D(t_frame_1, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_frame_1, global_id.x, y_m_o, (global_id.z + 1))) / 
       (4.f * hy);
    float fz = 
      (tex3D(t_frame_0, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_frame_0, global_id.x, global_id.y, (global_id.z + 1) - 1) +
       tex3D(t_frame_1, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_frame_1, global_id.x, global_id.y, (global_id.z + 1) - 1)) / 
       (4.f * hz);
    float ft = 
      tex3D(t_frame_1, global_id.x, global_id.y, (global_id.z + 1)) - tex3D(t_frame_0, global_id.x, global_id.y, (global_id.z + 1));

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

    float du = tex3D(t_flow_du, global_id.x, global_id.y, (global_id.z + 1));
    float dv = tex3D(t_flow_dv, global_id.x, global_id.y, (global_id.z + 1));
    float dw = tex3D(t_flow_dw, global_id.x, global_id.y, (global_id.z + 1));

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

extern "C" __global__ void solve_p(
        size_t width,
        size_t height,
        size_t depth,
        size_t chunk_depth,
        size_t z_start,
        float  hx,
        float  hy,
        float  hz,
        float  equation_alpha,
        float* output_du,
        float* output_dv,
        float* output_dw)
{
  dim3 global_id(
    blockDim.x * blockIdx.x + threadIdx.x,
    blockDim.y * blockIdx.y + threadIdx.y,
    blockDim.z * blockIdx.z + threadIdx.z);

  /* Texture access:                             */
  /* _m_o - minus one; _p_o - plus one           */
  /* x = x;        x - 1 = x_m_o;  x + 1 = x_p_o */
  /* y = y;        y - 1 = y_m_o;  y + 1 = y_p_o */
  /* z = (z + 1);  z - 1 = z;      z + 1 = z + 2 */

  if (global_id.x < width && global_id.y < height && global_id.z < chunk_depth) {
    unsigned int x_m_o = global_id.x == 0 ? 1 : global_id.x - 1;
    unsigned int x_p_o = global_id.x == width - 1 ? width - 2 : global_id.x + 1;

    unsigned int y_m_o = global_id.y == 0 ? 1 : global_id.y - 1;
    unsigned int y_p_o = global_id.y == height - 1 ? height - 2 : global_id.y + 1;

    /* Compute derivatives */
    float fx = 
      (tex3D(t_frame_0, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_frame_0, x_m_o, global_id.y, (global_id.z + 1)) +
       tex3D(t_frame_1, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_frame_1, x_m_o, global_id.y, (global_id.z + 1))) / 
       (4.f * hx);
    float fy = 
      (tex3D(t_frame_0, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_frame_0, global_id.x, y_m_o, (global_id.z + 1)) +
       tex3D(t_frame_1, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_frame_1, global_id.x, y_m_o, (global_id.z + 1))) / 
       (4.f * hy);
    float fz = 
      (tex3D(t_frame_0, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_frame_0, global_id.x, global_id.y, (global_id.z + 1) - 1) +
       tex3D(t_frame_1, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_frame_1, global_id.x, global_id.y, (global_id.z + 1) - 1)) / 
       (4.f * hz);
    float ft = 
      tex3D(t_frame_1, global_id.x, global_id.y, (global_id.z + 1)) - tex3D(t_frame_0, global_id.x, global_id.y, (global_id.z + 1));

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
    
    float xp = (global_id.x < width - 1)             * hx_2;
    float xm = (global_id.x > 0)                     * hx_2;
    float yp = (global_id.y < height - 1)            * hy_2;
    float ym = (global_id.y > 0)                     * hy_2;
    float zp = ((z_start + global_id.z) < depth - 1) * hz_2;
    float zm = ((z_start + global_id.z) > 0)         * hz_2;

    float phi_xp = (tex3D(t_phi, x_p_o, global_id.y, (global_id.z + 1)) + tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1))) / 2.f;
    float phi_xm = (tex3D(t_phi, x_m_o, global_id.y, (global_id.z + 1)) + tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1))) / 2.f;
    float phi_yp = (tex3D(t_phi, global_id.x, y_p_o, (global_id.z + 1)) + tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1))) / 2.f;
    float phi_ym = (tex3D(t_phi, global_id.x, y_m_o, (global_id.z + 1)) + tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1))) / 2.f;
    float phi_zp = (tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1) + 1) + tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1))) / 2.f;
    float phi_zm = (tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1) - 1) + tex3D(t_phi, global_id.x, global_id.y, (global_id.z + 1))) / 2.f;

    float sumH = (xp*phi_xp + xm*phi_xm + yp*phi_yp + ym*phi_ym + zp*phi_zp + zm*phi_zm);
    float sumU =
      phi_xp * xp * (tex3D(t_flow_u, x_p_o, global_id.y, (global_id.z + 1)) + tex3D(t_flow_du, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_xm * xm * (tex3D(t_flow_u, x_m_o, global_id.y, (global_id.z + 1)) + tex3D(t_flow_du, x_m_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_yp * yp * (tex3D(t_flow_u, global_id.x, y_p_o, (global_id.z + 1)) + tex3D(t_flow_du, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_ym * ym * (tex3D(t_flow_u, global_id.x, y_m_o, (global_id.z + 1)) + tex3D(t_flow_du, global_id.x, y_m_o, (global_id.z + 1)) - tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_zp * zp * (tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1) + 1) + tex3D(t_flow_du, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_zm * zm * (tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1) - 1) + tex3D(t_flow_du, global_id.x, global_id.y, (global_id.z + 1) - 1) - tex3D(t_flow_u, global_id.x, global_id.y, (global_id.z + 1)));
    float sumV =
      phi_xp * xp * (tex3D(t_flow_v, x_p_o, global_id.y, (global_id.z + 1)) + tex3D(t_flow_dv, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_xm * xm * (tex3D(t_flow_v, x_m_o, global_id.y, (global_id.z + 1)) + tex3D(t_flow_dv, x_m_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_yp * yp * (tex3D(t_flow_v, global_id.x, y_p_o, (global_id.z + 1)) + tex3D(t_flow_dv, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_ym * ym * (tex3D(t_flow_v, global_id.x, y_m_o, (global_id.z + 1)) + tex3D(t_flow_dv, global_id.x, y_m_o, (global_id.z + 1)) - tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_zp * zp * (tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1) + 1) + tex3D(t_flow_dv, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_zm * zm * (tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1) - 1) + tex3D(t_flow_dv, global_id.x, global_id.y, (global_id.z + 1) - 1) - tex3D(t_flow_v, global_id.x, global_id.y, (global_id.z + 1)));
    float sumW =
      phi_xp * xp * (tex3D(t_flow_w, x_p_o, global_id.y, (global_id.z + 1)) + tex3D(t_flow_dw, x_p_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_xm * xm * (tex3D(t_flow_w, x_m_o, global_id.y, (global_id.z + 1)) + tex3D(t_flow_dw, x_m_o, global_id.y, (global_id.z + 1)) - tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_yp * yp * (tex3D(t_flow_w, global_id.x, y_p_o, (global_id.z + 1)) + tex3D(t_flow_dw, global_id.x, y_p_o, (global_id.z + 1)) - tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_ym * ym * (tex3D(t_flow_w, global_id.x, y_m_o, (global_id.z + 1)) + tex3D(t_flow_dw, global_id.x, y_m_o, (global_id.z + 1)) - tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_zp * zp * (tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1) + 1) + tex3D(t_flow_dw, global_id.x, global_id.y, (global_id.z + 1) + 1) - tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1))) +
      phi_zm * zm * (tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1) - 1) + tex3D(t_flow_dw, global_id.x, global_id.y, (global_id.z + 1) - 1) - tex3D(t_flow_w, global_id.x, global_id.y, (global_id.z + 1)));

    float result_du =
      (tex3D(t_ksi, global_id.x, global_id.y, (global_id.z + 1)) * (-J14 - J12 * tex3D(t_flow_dv, global_id.x, global_id.y, (global_id.z + 1)) - J13 * tex3D(t_flow_dw, global_id.x, global_id.y, (global_id.z + 1))) + sumU) /
      (tex3D(t_ksi, global_id.x, global_id.y, (global_id.z + 1)) * J11 + sumH);

    float result_dv =
      (tex3D(t_ksi, global_id.x, global_id.y, (global_id.z + 1)) * (-J24 - J12 * result_du - J23 * tex3D(t_flow_dw, global_id.x, global_id.y, (global_id.z + 1))) + sumV) /
      (tex3D(t_ksi, global_id.x, global_id.y, (global_id.z + 1)) * J22 + sumH);
    
    float result_dw =
      (tex3D(t_ksi, global_id.x, global_id.y, (global_id.z + 1)) * (-J34 - J13 * result_du - J23 * result_dv) + sumW) /
      (tex3D(t_ksi, global_id.x, global_id.y, (global_id.z + 1)) * J33 + sumH);

    output_du[IND(global_id.x, global_id.y, global_id.z)] = result_du;
    output_dv[IND(global_id.x, global_id.y, global_id.z)] = result_dv;
    output_dw[IND(global_id.x, global_id.y, global_id.z)] = result_dw;
  }
}