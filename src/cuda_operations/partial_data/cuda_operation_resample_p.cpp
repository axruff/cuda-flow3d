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

#include "src/cuda_operations/partial_data/cuda_operation_resample_p.h"

#include <algorithm>
#include <cstdio>

#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationResampleP::CudaOperationResampleP()
  : CudaOperationBase("CUDA Resample Piecemeal")
{
}

bool CudaOperationResampleP::Initialize(const OperationParameters* params)
{
  initialized_ = false;

  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/resample_p_3d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_resample_x_, cu_module_, "resample_x_p_3d")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_resample_y_, cu_module_, "resample_y_p_3d")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_resample_z_, cu_module_, "resample_z_p_3d"))
        ) {
      initialized_ = true;
    } else {
      CheckCudaError(cuModuleUnload(cu_module_));
      cu_module_ = nullptr;
    }
  }
  return initialized_;
}

void CudaOperationResampleP::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  DataSize4 data_size;
  DataSize4 resample_size;

  GET_PARAM_OR_RETURN(params, DataSize4, data_size,     "data_size");
  GET_PARAM_OR_RETURN(params, DataSize4, resample_size, "resample_size");

  Data3D* input_ptr;
  Data3D* output_ptr;

  GET_PARAM_PTR_OR_RETURN(params, Data3D, input_ptr,  "input");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, output_ptr, "output");

  Data3D& input = *input_ptr;
  Data3D& output = *output_ptr;

  /* Check container's dimensions, data must fit to the memory */
  bool input_check =
    data_size.width <= input.Width() &&
    data_size.height <= input.Height() &&
    data_size.depth <= input.Depth();

  bool output_check = 
    resample_size.width <= output.Width() &&
    resample_size.height <= output.Height() &&
    resample_size.depth <= output.Depth();
  
  bool downsample_check =
    output.Width() >= input.Width() &&
    output.Height() >= input.Height() &&
    output.Depth() >= input.Depth();


  if (!input_check || !output_check || !downsample_check) {
    std::printf("Error: Operation '%s'. Wrong dimensions.\n", GetName());
    return;
  }

  /* Check available memory on the cuda device */
  size_t total_memory;
  CheckCudaError(cuMemGetInfo(&free_memory_, &total_memory));

  DataSize4 output_size = data_size;
  output_size.width = resample_size.width;
  ResampleX(input, output, data_size, output_size);

  data_size = output_size;
  output_size.height = resample_size.height;
  ResampleY(output, output, data_size, output_size);

  data_size = output_size;
  output_size.depth = resample_size.depth;
  ResampleZ(output, output, data_size, output_size);
 
}

void CudaOperationResampleP::ResampleX(Data3D& input, Data3D& output, DataSize4& input_size, DataSize4& output_size) const
{
  input_size.pitch = ((input_size.width % alignment_ == 0) ?
                      (input_size.width) :
                      (input_size.width + alignment_ - (input_size.width % alignment_))) *
                      sizeof(float);

 output_size.pitch = ((output_size.width % alignment_ == 0) ?
                      (output_size.width) :
                      (output_size.width + alignment_ - (output_size.width % alignment_))) *
                      sizeof(float);

  size_t i_mem = input_size.pitch * input_size.height * input_size.depth;
  size_t o_mem = output_size.pitch * output_size.height * output_size.depth;

  size_t parts = (i_mem + o_mem) / free_memory_ + 1;
  size_t slice_height = input_size.height / parts;

  /* Allocate CUDA memory on the device */
  CUdeviceptr dev_input;
  CUdeviceptr dev_output;

  size_t i_chunk_size = input_size.pitch * slice_height * input_size.depth;
  size_t o_chunk_size = output_size.pitch * slice_height * output_size.depth;

  bool error = true;
  if ((i_chunk_size + o_chunk_size) <= free_memory_) {
    error = false;
    error |= CheckCudaError(cuMemAlloc(&dev_input, i_chunk_size));
    error |= CheckCudaError(cuMemAlloc(&dev_output, o_chunk_size));
  }

  if (error) {
    std::printf("Error: Operation '%s'. Cannot allocate memory on the device.\n", GetName());
    return;
  }

  /* Initialize kernel constants with sizes of input and output buffers */
  DataSize4 dev_i_size = { input_size.width, slice_height, input_size.depth, input_size.pitch };
  DataSize4 dev_o_size = { output_size.width, slice_height, output_size.depth, output_size.pitch };

  CUdeviceptr dev_const;
  size_t const_size;
  error = false;
  error |= CheckCudaError(cuModuleGetGlobal(&dev_const, &const_size, cu_module_, "input_size"));
  error |= CheckCudaError(     cuMemcpyHtoD(dev_const, &dev_i_size, const_size));
  error |= CheckCudaError(cuModuleGetGlobal(&dev_const, &const_size, cu_module_, "output_size"));
  error |= CheckCudaError(     cuMemcpyHtoD(dev_const, &dev_o_size, const_size));

  if (error) {
    std::printf("Error: Operation '%s'. Cannot initialize constatn memory on the device.\n", GetName());
    return;
  }

  /* Iterate over chunks and perform resampling */
  size_t y_start = 0;
  while (y_start < input_size.height) {
    size_t y_end = std::min(input_size.height, y_start + slice_height);
    size_t chunk_height = y_end - y_start;

    /* Copy a chunk of input to the device */
    CUDA_MEMCPY3D cu_copy3d;
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.srcHost = input.DataPtr();
    cu_copy3d.srcPitch = input.Width() * sizeof(float);
    cu_copy3d.srcHeight = input.Height();

    cu_copy3d.srcXInBytes = 0;
    cu_copy3d.srcY = y_start; /**/
    cu_copy3d.srcZ = 0;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.dstDevice = dev_input;
    cu_copy3d.dstPitch = input_size.pitch;
    cu_copy3d.dstHeight = slice_height;

    cu_copy3d.WidthInBytes = input_size.width * sizeof(float);
    cu_copy3d.Height = chunk_height; /**/
    cu_copy3d.Depth = input_size.depth;

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Perform resampling */
    dim3 block_dim = { 16, 8, 8 };
    dim3 grid_dim =  { static_cast<unsigned int>((dev_o_size.width + block_dim.x - 1) / block_dim.x),
                       static_cast<unsigned int>((chunk_height + block_dim.y - 1) / block_dim.y),
                       static_cast<unsigned int>((dev_o_size.depth + block_dim.z - 1) / block_dim.z) };
  
    void* args[6] = { &dev_input, 
                      &dev_output,
                      &dev_o_size.width,
                      &chunk_height,
                      &dev_o_size.depth,
                      &dev_i_size.width };
  
    CheckCudaError(cuLaunchKernel(cuf_resample_x_,
                                  grid_dim.x, grid_dim.y, grid_dim.z,
                                  block_dim.x, block_dim.y, block_dim.z,
                                  0,
                                  NULL,
                                  args,
                                  NULL));

    /* Copy a resampled chunk of output to the host */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));
    
    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_output;
    cu_copy3d.srcPitch = output_size.pitch;
    cu_copy3d.srcHeight = slice_height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = output.DataPtr();
    cu_copy3d.dstPitch = output.Width() * sizeof(float);
    cu_copy3d.dstHeight = output.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = y_start; /**/
    cu_copy3d.dstZ = 0;
    
    cu_copy3d.WidthInBytes = output_size.width * sizeof(float);
    cu_copy3d.Height = chunk_height; /**/
    cu_copy3d.Depth = output_size.depth;

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    y_start += slice_height;
  }

  CheckCudaError(cuMemFree(dev_input));
  CheckCudaError(cuMemFree(dev_output));
}

void CudaOperationResampleP::ResampleY(Data3D& input, Data3D& output, DataSize4& input_size, DataSize4& output_size) const
{
  input_size.pitch = ((input_size.width % alignment_ == 0) ?
                      (input_size.width) :
                      (input_size.width + alignment_ - (input_size.width % alignment_))) *
                      sizeof(float);

  output_size.pitch = ((output_size.width % alignment_ == 0) ?
                       (output_size.width) :
                       (output_size.width + alignment_ - (output_size.width % alignment_))) *
                       sizeof(float);

  size_t i_mem = input_size.pitch * input_size.height * input_size.depth;
  size_t o_mem = output_size.pitch * output_size.height * output_size.depth;

  size_t parts = (i_mem + o_mem) / free_memory_ + 1;
  size_t slice_depth = input_size.depth / parts;

  /* Allocate CUDA memory on the device */
  CUdeviceptr dev_input;
  CUdeviceptr dev_output;

  size_t i_chunk_size = input_size.pitch * input_size.height * slice_depth;
  size_t o_chunk_size = output_size.pitch * output_size.height * slice_depth;

  bool error = true;
  if ((i_chunk_size + o_chunk_size) <= free_memory_) {
    error = false;
    error |= CheckCudaError(cuMemAlloc(&dev_input, i_chunk_size));
    error |= CheckCudaError(cuMemAlloc(&dev_output, o_chunk_size));
  }

  if (error) {
    std::printf("Error: Operation '%s'. Cannot allocate memory on the device.\n", GetName());
    return;
  }

  /* Initialize kernel constants with sizes of input and output buffers */
  DataSize4 dev_i_size = { input_size.width, input_size.height, slice_depth, input_size.pitch };
  DataSize4 dev_o_size = { output_size.width, output_size.height, slice_depth, output_size.pitch };

  CUdeviceptr dev_const;
  size_t const_size;
  error = false;
  error |= CheckCudaError(cuModuleGetGlobal(&dev_const, &const_size, cu_module_, "input_size"));
  error |= CheckCudaError(     cuMemcpyHtoD(dev_const, &dev_i_size, const_size));
  error |= CheckCudaError(cuModuleGetGlobal(&dev_const, &const_size, cu_module_, "output_size"));
  error |= CheckCudaError(     cuMemcpyHtoD(dev_const, &dev_o_size, const_size));

  if (error) {
    std::printf("Error: Operation '%s'. Cannot initialize constatn memory on the device.\n", GetName());
    return;
  }

  /* Iterate over chunks and perform resampling */
  size_t z_start = 0;
  while (z_start < input_size.depth) {
    size_t z_end = std::min(input_size.depth, z_start + slice_depth);
    size_t chunk_depth = z_end - z_start;

    /* Copy a chunk of input to the device */
    CUDA_MEMCPY3D cu_copy3d;
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.srcHost = input.DataPtr();
    cu_copy3d.srcPitch = input.Width() * sizeof(float);
    cu_copy3d.srcHeight = input.Height();

    cu_copy3d.srcXInBytes = 0;
    cu_copy3d.srcY = 0;
    cu_copy3d.srcZ = z_start; /**/

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.dstDevice = dev_input;
    cu_copy3d.dstPitch = input_size.pitch;
    cu_copy3d.dstHeight = input_size.height;

    cu_copy3d.WidthInBytes = input_size.width * sizeof(float);
    cu_copy3d.Height = input_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Perform resampling */
    dim3 block_dim = { 16, 8, 8 };
    dim3 grid_dim =  { static_cast<unsigned int>((dev_o_size.width + block_dim.x - 1) / block_dim.x),
                       static_cast<unsigned int>((dev_o_size.height + block_dim.y - 1) / block_dim.y),
                       static_cast<unsigned int>((chunk_depth + block_dim.z - 1) / block_dim.z) };
  
    void* args[6] = { &dev_input, 
                      &dev_output,
                      &dev_o_size.width,
                      &dev_o_size.height,
                      &chunk_depth,
                      &dev_i_size.height };
  
    CheckCudaError(cuLaunchKernel(cuf_resample_y_,
                                  grid_dim.x, grid_dim.y, grid_dim.z,
                                  block_dim.x, block_dim.y, block_dim.z,
                                  0,
                                  NULL,
                                  args,
                                  NULL));

    /* Copy a resampled chunk of output to the host */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));
    
    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_output;
    cu_copy3d.srcPitch = output_size.pitch;
    cu_copy3d.srcHeight = output_size.height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = output.DataPtr();
    cu_copy3d.dstPitch = output.Width() * sizeof(float);
    cu_copy3d.dstHeight = output.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = 0;
    cu_copy3d.dstZ = z_start; /**/
    
    cu_copy3d.WidthInBytes = output_size.width * sizeof(float);
    cu_copy3d.Height = output_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    z_start += slice_depth;
  }

  CheckCudaError(cuMemFree(dev_input));
  CheckCudaError(cuMemFree(dev_output));
}

void CudaOperationResampleP::ResampleZ(Data3D& input, Data3D& output, DataSize4& input_size, DataSize4& output_size) const
{
  input_size.pitch = ((input_size.width % alignment_ == 0) ?
                      (input_size.width) :
                      (input_size.width + alignment_ - (input_size.width % alignment_))) *
                      sizeof(float);

  output_size.pitch = ((output_size.width % alignment_ == 0) ?
                       (output_size.width) :
                       (output_size.width + alignment_ - (output_size.width % alignment_))) *
                       sizeof(float);

  size_t i_mem = input_size.pitch * input_size.height * input_size.depth;
  size_t o_mem = output_size.pitch * output_size.height * output_size.depth;

  size_t parts = (i_mem + o_mem) / free_memory_ + 1;
  size_t slice_height = input_size.height / parts;

  /* Allocate CUDA memory on the device */
  CUdeviceptr dev_input;
  CUdeviceptr dev_output;

  size_t i_chunk_size = input_size.pitch * slice_height * input_size.depth;
  size_t o_chunk_size = output_size.pitch * slice_height * output_size.depth;

  bool error = true;
  if ((i_chunk_size + o_chunk_size) <= free_memory_) {
    error = false;
    error |= CheckCudaError(cuMemAlloc(&dev_input, i_chunk_size));
    error |= CheckCudaError(cuMemAlloc(&dev_output, o_chunk_size));
  }

  if (error) {
    std::printf("Error: Operation '%s'. Cannot allocate memory on the device.\n", GetName());
    return;
  }

  /* Initialize kernel constants with sizes of input and output buffers */
  DataSize4 dev_i_size = { input_size.width, slice_height, input_size.depth, input_size.pitch };
  DataSize4 dev_o_size = { output_size.width, slice_height, output_size.depth, output_size.pitch };

  CUdeviceptr dev_const;
  size_t const_size;
  error = false;
  error |= CheckCudaError(cuModuleGetGlobal(&dev_const, &const_size, cu_module_, "input_size"));
  error |= CheckCudaError(     cuMemcpyHtoD(dev_const, &dev_i_size, const_size));
  error |= CheckCudaError(cuModuleGetGlobal(&dev_const, &const_size, cu_module_, "output_size"));
  error |= CheckCudaError(     cuMemcpyHtoD(dev_const, &dev_o_size, const_size));

  if (error) {
    std::printf("Error: Operation '%s'. Cannot initialize constatn memory on the device.\n", GetName());
    return;
  }

  /* Iterate over chunks and perform resampling */
  size_t y_start = 0;
  while (y_start < input_size.height) {
    size_t y_end = std::min(input_size.height, y_start + slice_height);
    size_t chunk_height = y_end - y_start;

    /* Copy a chunk of input to the device */
    CUDA_MEMCPY3D cu_copy3d;
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.srcHost = input.DataPtr();
    cu_copy3d.srcPitch = input.Width() * sizeof(float);
    cu_copy3d.srcHeight = input.Height();

    cu_copy3d.srcXInBytes = 0;
    cu_copy3d.srcY = y_start; /**/
    cu_copy3d.srcZ = 0;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.dstDevice = dev_input;
    cu_copy3d.dstPitch = input_size.pitch;
    cu_copy3d.dstHeight = slice_height;

    cu_copy3d.WidthInBytes = input_size.width * sizeof(float);
    cu_copy3d.Height = chunk_height; /**/
    cu_copy3d.Depth = input_size.depth;

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Perform resampling */
    dim3 block_dim = { 16, 8, 8 };
    dim3 grid_dim =  { static_cast<unsigned int>((dev_o_size.width + block_dim.x - 1) / block_dim.x),
                       static_cast<unsigned int>((chunk_height + block_dim.y - 1) / block_dim.y),
                       static_cast<unsigned int>((dev_o_size.depth + block_dim.z - 1) / block_dim.z) };

    void* args[6] = { &dev_input, 
                      &dev_output,
                      &dev_o_size.width,
                      &chunk_height,
                      &dev_o_size.depth,
                      &dev_i_size.depth };

    CheckCudaError(cuLaunchKernel(cuf_resample_z_,
                                  grid_dim.x, grid_dim.y, grid_dim.z,
                                  block_dim.x, block_dim.y, block_dim.z,
                                  0,
                                  NULL,
                                  args,
                                  NULL));

    /* Copy a resampled chunk of output to the host */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));
    
    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_output;
    cu_copy3d.srcPitch = output_size.pitch;
    cu_copy3d.srcHeight = slice_height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = output.DataPtr();
    cu_copy3d.dstPitch = output.Width() * sizeof(float);
    cu_copy3d.dstHeight = output.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = y_start; /**/
    cu_copy3d.dstZ = 0;
    
    cu_copy3d.WidthInBytes = output_size.width * sizeof(float);
    cu_copy3d.Height = chunk_height; /**/
    cu_copy3d.Depth = output_size.depth;

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    y_start += slice_height;
  }

  CheckCudaError(cuMemFree(dev_input));
  CheckCudaError(cuMemFree(dev_output));
}