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

#include "src/cuda_operations/partial_data/cuda_operation_add_p.h"

#include <algorithm>

#include "src/data_types/data3d.h"
#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationAddP::CudaOperationAddP()
  : CudaOperationBase("CUDA Add Piecemeal")
{
}

bool CudaOperationAddP::Initialize(const OperationParameters* params)
{
  initialized_ = false;

  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/add_3d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_add_, cu_module_, "add_3d"))) {
      initialized_ = true;
    } else {
      CheckCudaError(cuModuleUnload(cu_module_));
      cu_module_ = nullptr;
    }
  }
  return initialized_;
}

void CudaOperationAddP::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  Data3D* p_operand_0;
  Data3D* p_operand_1;

  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_operand_0, "operand_0");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_operand_1, "operand_1");

  Data3D& operand_0 = *p_operand_0;
  Data3D& operand_1 = *p_operand_1;

  DataSize4 data_size;

  GET_PARAM_OR_RETURN(params, DataSize4, data_size, "data_size");

    /* Check available memory on the cuda device */
  size_t total_memory;
  CheckCudaError(cuMemGetInfo(&free_memory_, &total_memory));

  /* Debug */
  //free_memory_ = 16.f * 1024 * 1024;

  data_size.pitch = ((data_size.width % alignment_ == 0) ?
                     (data_size.width) :
                     (data_size.width + alignment_ - (data_size.width % alignment_))) *
                      sizeof(float);

  size_t mem = data_size.pitch * data_size.height * data_size.depth;

  const unsigned int containers_count = 2;
  size_t max_slice_depth = free_memory_ / (containers_count * data_size.pitch * data_size.height);
  size_t total_slice_memory = containers_count * data_size.pitch * data_size.height * max_slice_depth;

  //std::printf("\nADD\n");
  //std::printf("Max slice depth: %d Total slice memory: %.2f MB\n", max_slice_depth, total_slice_memory / (1024.f * 1024.f));
  //std::printf("Free memory: %.2f MB Container size: %.2f MB Total size: %.2f MB\n",
  //  free_memory_ / (1024.f * 1024.f), mem / (1024.f * 1024.f), (containers_count * mem) / (1024.f * 1024.f));

  //std::printf("Max slice depth: %d\n", max_slice_depth);

  /* Allocate CUDA memory on the device (CUDA Arrays for the input (textures), Pitched memory for the output) */
  DataSize4 dev_data_size = data_size;
  dev_data_size.depth = std::min(data_size.depth, max_slice_depth);

  CUdeviceptr dev_operand_0;
  CUdeviceptr dev_operand_1;

  CheckCudaError(cuMemAlloc(&dev_operand_0, dev_data_size.pitch * dev_data_size.height * dev_data_size.depth));
  CheckCudaError(cuMemAlloc(&dev_operand_1, dev_data_size.pitch * dev_data_size.height * dev_data_size.depth));

  /* Initialize kernel constants with sizes of a chunk */
  size_t const_size;
  CheckCudaError(cuModuleGetGlobal(&dev_constants_, &const_size, cu_module_, "container_size"));
  CheckCudaError(     cuMemcpyHtoD(dev_constants_, &dev_data_size, const_size));

  /* Iterate over chunks */
  size_t z_start = 0;
  while (z_start < data_size.depth) {
    size_t z_end = std::min(data_size.depth, z_start + max_slice_depth);
    size_t chunk_depth = z_end - z_start;

    size_t mem_usage = data_size.pitch * data_size.height * chunk_depth;
    //std::printf("ADD Processing chunk: %3d - %3d Memory: %.2f MB\n", z_start, z_end, containers_count * mem_usage / (1024.f * 1024.f));

   /* Copy a chunk of input data to the device */
    CUDA_MEMCPY3D cu_copy3d;
    /* Operand 0 */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.srcHost = operand_0.DataPtr();
    cu_copy3d.srcPitch = operand_0.Width() * sizeof(float);
    cu_copy3d.srcHeight = operand_0.Height();

    cu_copy3d.srcXInBytes = 0;
    cu_copy3d.srcY = 0;
    cu_copy3d.srcZ = z_start; /**/

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.dstDevice = dev_operand_0;
    cu_copy3d.dstPitch = dev_data_size.pitch;
    cu_copy3d.dstHeight = dev_data_size.height;

    cu_copy3d.WidthInBytes = dev_data_size.width * sizeof(float);
    cu_copy3d.Height = dev_data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Operand 1 */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.srcHost = operand_1.DataPtr();
    cu_copy3d.srcPitch = operand_1.Width() * sizeof(float);
    cu_copy3d.srcHeight = operand_1.Height();

    cu_copy3d.srcXInBytes = 0;
    cu_copy3d.srcY = 0;
    cu_copy3d.srcZ = z_start; /**/

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.dstDevice = dev_operand_1;
    cu_copy3d.dstPitch = dev_data_size.pitch;
    cu_copy3d.dstHeight = dev_data_size.height;

    cu_copy3d.WidthInBytes = dev_data_size.width * sizeof(float);
    cu_copy3d.Height = dev_data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Launch kernel */
    size_t block_dim3[3] = { 16, 8, 4 };
    size_t grid_dim3[3] = { (dev_data_size.width  + block_dim3[0] - 1) / block_dim3[0],
                            (dev_data_size.height + block_dim3[1] - 1) / block_dim3[1],
                            (chunk_depth          + block_dim3[2] - 1) / block_dim3[2] };

    void* args[5] = { 
      &dev_operand_0, 
      &dev_operand_1,
      &dev_data_size.width, 
      &dev_data_size.height, 
      &chunk_depth };

    CheckCudaError(cuLaunchKernel(cuf_add_,
                                  grid_dim3[0], grid_dim3[1], grid_dim3[2],
                                  block_dim3[0], block_dim3[1], block_dim3[2],
                                  0,
                                  NULL,
                                  args,
                                  NULL));

    /* Copy a chunk of output to the host */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));
    
    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_operand_0;
    cu_copy3d.srcPitch = dev_data_size.pitch;
    cu_copy3d.srcHeight = dev_data_size.height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = operand_0.DataPtr();
    cu_copy3d.dstPitch = operand_0.Width() * sizeof(float);
    cu_copy3d.dstHeight = operand_0.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = 0;
    cu_copy3d.dstZ = z_start; /**/
    
    cu_copy3d.WidthInBytes = dev_data_size.width * sizeof(float);
    cu_copy3d.Height = dev_data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    z_start += max_slice_depth;
  }

  CheckCudaError(cuMemFree(dev_operand_0));
  CheckCudaError(cuMemFree(dev_operand_1));
}