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

#include "src/utils/cuda_utils.h"

#include <cstring>

bool InitCudaContextWithFirstAvailableDevice(CUcontext* cu_context)
{
  if (CheckCudaError(cuInit(0))) {
    return false;
  }

  int cu_device_count;
  if (CheckCudaError(cuDeviceGetCount(&cu_device_count))) {
    return false;
  }
  CUdevice cu_device;

  if (cu_device_count == 0) {
    printf("There are no cuda capable devices.");
    return false;
  }

  if (CheckCudaError(cuDeviceGet(&cu_device, 0))) {
    return false;
  }

  char cu_device_name[64];
  if (CheckCudaError(cuDeviceGetName(cu_device_name, 64, cu_device))) {
    return false;
  }

  int launch_timeout;
  CheckCudaError(cuDeviceGetAttribute(&launch_timeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, cu_device));

  printf("CUDA Device: %s. Launch timeout: %s\n", cu_device_name, (launch_timeout ? "Yes" : "No"));

  if (CheckCudaError(cuCtxCreate(cu_context, 0, cu_device))) {
    return false;
  }

  return true;
}

void CopyData3DtoDevice(Data3D& data3d, CUdeviceptr device_ptr, size_t device_height, size_t device_pitch)
{
  CUDA_MEMCPY3D cu_copy3d;
  std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

  cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
  cu_copy3d.srcHost = data3d.DataPtr();
  cu_copy3d.srcPitch = data3d.Width() * sizeof(float);
  cu_copy3d.srcHeight = data3d.Height();

  cu_copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  cu_copy3d.dstDevice = device_ptr;
  cu_copy3d.dstPitch = device_pitch;
  cu_copy3d.dstHeight = device_height;

  cu_copy3d.WidthInBytes = data3d.Width() * sizeof(float);
  cu_copy3d.Height = data3d.Height();
  cu_copy3d.Depth = data3d.Depth();

  CheckCudaError(cuMemcpy3D(&cu_copy3d));
}


void CopyData3DFromDevice(CUdeviceptr device_ptr, Data3D& data3d, size_t device_height, size_t device_pitch)
{
  CUDA_MEMCPY3D cu_copy3d;
  std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

  cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  cu_copy3d.srcDevice = device_ptr;
  cu_copy3d.srcPitch = device_pitch;
  cu_copy3d.srcHeight = device_height;

  cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
  cu_copy3d.dstHost = data3d.DataPtr();
  cu_copy3d.dstPitch = data3d.Width() * sizeof(float);
  cu_copy3d.dstHeight = data3d.Height();

  cu_copy3d.WidthInBytes = data3d.Width() * sizeof(float);
  cu_copy3d.Height = data3d.Height();
  cu_copy3d.Depth = data3d.Depth();
  CheckCudaError(cuMemcpy3D(&cu_copy3d));
}
