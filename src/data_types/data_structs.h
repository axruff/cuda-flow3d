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

#ifndef GPUFLOW3D_DATA_TYPES_DATA_STRUCTS_H_
#define GPUFLOW3D_DATA_TYPES_DATA_STRUCTS_H_

struct DataSize4 {
  size_t width;
  size_t height;
  size_t depth;
  size_t pitch;
};



struct Stat3 {
  float min;
  float max;
  float avg;
};

#endif // !GPUFLOW3D_DATA_TYPES_DATA_STRUCTS_H_
