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

#include "src/cuda_operations/partial_data/cuda_operation_register_p.h"

#include <algorithm>

#include <vector_types.h>

#include "src/data_types/data3d.h"
#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationRegistrationP::CudaOperationRegistrationP()
  : CudaOperationBase("CUDA Registration")
{
}

bool CudaOperationRegistrationP::Initialize(const OperationParameters* params)
{
  initialized_ = false;
 
  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/registration_p.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_registration_p_, cu_module_, "registration_p"))) {
            initialized_ = true;
    } else {
      CheckCudaError(cuModuleUnload(cu_module_));
      cu_module_ = nullptr;
    }
  }
  return initialized_;
}

void CudaOperationRegistrationP::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  Data3D* p_frame_0;
  Data3D* p_frame_1;
  Data3D* p_flow_u;
  Data3D* p_flow_v;
  Data3D* p_flow_w;
  Data3D* p_temp;

  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_frame_0, "frame_0");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_frame_1, "frame_1");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_u,  "flow_u");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_v,  "flow_v");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_w,  "flow_w");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_temp,    "temp");

  /* Use references just for convenience */
  Data3D& frame_0 = *p_frame_0;
  Data3D& frame_1 = *p_frame_1;
  Data3D& flow_u  = *p_flow_u;
  Data3D& flow_v  = *p_flow_v;
  Data3D& flow_w  = *p_flow_w;
  Data3D& temp  = *p_temp;

  float hx;
  float hy;
  float hz;
  DataSize4 data_size;
  size_t max_mag;

  GET_PARAM_OR_RETURN(params, float,     hx,        "hx");
  GET_PARAM_OR_RETURN(params, float,     hy,        "hy");
  GET_PARAM_OR_RETURN(params, float,     hz,        "hz");
  GET_PARAM_OR_RETURN(params, DataSize4, data_size, "data_size");
  GET_PARAM_OR_RETURN(params, size_t,    max_mag,   "max_mag");

  //std::printf("Backward Registration\n");
  /* CPU Version*/
  {
    for (size_t z = 0; z < data_size.depth; ++z) {
      for (size_t y = 0; y < data_size.height; ++y) {
        for (size_t x = 0; x < data_size.width; ++x) {
          float x_f = x + flow_u.Data(x, y, z) * (1.f / hx);
          float y_f = y + flow_v.Data(x, y, z) * (1.f / hy);
          float z_f = z + flow_w.Data(x, y, z) * (1.f / hz);

          if ((x_f < 0.) || (x_f > data_size.width - 1) || (y_f < 0.) || (y_f > data_size.height - 1) || (z_f < 0.) || (z_f > data_size.depth - 1) ||
            std::isnan(x_f) || std::isnan(y_f) || std::isnan(z_f)) {
            temp.Data(x, y, z) = frame_0.Data(x, y, z);
          } else {
            size_t xx = static_cast<size_t>(std::floor(x_f)); 
            size_t yy = static_cast<size_t>(std::floor(y_f)); 
            size_t zz = static_cast<size_t>(std::floor(z_f)); 
            float delta_x = x_f - static_cast<float>(xx);
            float delta_y = y_f - static_cast<float>(yy);
            float delta_z = z_f - static_cast<float>(zz);

            size_t x_1 = std::min(data_size.width -1,   xx + 1);
            size_t y_1 = std::min(data_size.height - 1, yy + 1);
            size_t z_1 = std::min(data_size.depth - 1,  zz + 1);

            float value_0 =
              (1.f - delta_x) * (1.f - delta_y) * frame_1.Data(xx , yy , zz ) +
              (      delta_x) * (1.f - delta_y) * frame_1.Data(x_1, yy , zz ) +
              (1.f - delta_x) * (      delta_y) * frame_1.Data(xx , y_1, zz ) +
              (      delta_x) * (      delta_y) * frame_1.Data(x_1, y_1, zz );

            float value_1 =
              (1.f - delta_x) * (1.f - delta_y) * frame_1.Data(xx , yy , z_1) +
              (      delta_x) * (1.f - delta_y) * frame_1.Data(x_1, yy , z_1) +
              (1.f - delta_x) * (      delta_y) * frame_1.Data(xx , y_1, z_1) +
              (      delta_x) * (      delta_y) * frame_1.Data(x_1, y_1, z_1);

           temp.Data(x, y, z) =
              (1.f - delta_z) * value_0 + delta_z * value_1;
          }
        }
      }
    }

    frame_1.Swap(temp);
  }

  ///* Check available memory on the cuda device and texture memory alignment */
  //size_t total_memory;
  //CheckCudaError(cuMemGetInfo(&free_memory_, &total_memory));

  //CUdevice cu_device;
  //CheckCudaError(cuCtxGetDevice(&cu_device));
  //CheckCudaError(cuDeviceGetAttribute(&alignment_, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, cu_device));

  //alignment_ /= sizeof(float);

  ///* Debug */
  ////free_memory_ = 16.f * 1024 * 1024;

  //data_size.pitch = ((data_size.width % alignment_ == 0) ?
  //                   (data_size.width) :
  //                   (data_size.width + alignment_ - (data_size.width % alignment_))) *
  //                    sizeof(float);

  //size_t mem = data_size.pitch * data_size.height * data_size.depth;

  //const unsigned int containers_count = 6;
  //size_t max_slice_depth = free_memory_ / (containers_count * data_size.pitch * data_size.height);
  //size_t total_slice_memory = containers_count * data_size.pitch * data_size.height * max_slice_depth;

  //max_mag = std::min(max_mag, static_cast<size_t>(std::sqrt(data_size.width * data_size.width + data_size.height * data_size.height + data_size.depth * data_size.depth)));

  //std::printf("Max magnitude: %d\n", max_mag);

  //const size_t min_slice_depth = 1 + 2 * max_mag;
  //if (max_slice_depth < min_slice_depth) {
  //  std::printf("Error. Low GPU memory. Data cannot be partitioned properly. Try to reduce the input size of dataset.\n");
  //  std::printf("Max magnitude: %d\n", max_mag);
  //  //return;
  //}
  //size_t processing_slice_depth = std::min(data_size.depth, max_slice_depth - 2 * max_mag);

  ///* Allocate CUDA memory on the device (CUDA Arrays for the input (textures), Pitched memory for the output) */
  //DataSize4 dev_data_size = data_size;
  //dev_data_size.depth = processing_slice_depth + 2 * max_mag;

  //CUarray dev_ar_frame_0;
  //CUarray dev_ar_frame_1;
  //CUarray dev_ar_flow_u;
  //CUarray dev_ar_flow_v;
  //CUarray dev_ar_flow_w;

  //CUdeviceptr dev_output;

  //CUDA_ARRAY3D_DESCRIPTOR cu_array_descriptor;

  //cu_array_descriptor.Format = CU_AD_FORMAT_FLOAT;
  //cu_array_descriptor.NumChannels = 1;
  //cu_array_descriptor.Width = dev_data_size.width;
  //cu_array_descriptor.Height = dev_data_size.height;
  //cu_array_descriptor.Depth = dev_data_size.depth;
  //cu_array_descriptor.Flags = 0;

  //CheckCudaError(cuArray3DCreate(&dev_ar_frame_0, &cu_array_descriptor));
  //CheckCudaError(cuArray3DCreate(&dev_ar_frame_1, &cu_array_descriptor));
  //CheckCudaError(cuArray3DCreate(&dev_ar_flow_u,  &cu_array_descriptor));
  //CheckCudaError(cuArray3DCreate(&dev_ar_flow_v,  &cu_array_descriptor));
  //CheckCudaError(cuArray3DCreate(&dev_ar_flow_w,  &cu_array_descriptor));

  //CheckCudaError(cuMemAlloc(&dev_output, dev_data_size.pitch * dev_data_size.height * (dev_data_size.depth - 2 * max_mag)));

  ///* Bind textures */
  //CUtexref cu_tr_frame_0;
  //CUtexref cu_tr_frame_1;
  //CUtexref cu_tr_flow_u;
  //CUtexref cu_tr_flow_v;
  //CUtexref cu_tr_flow_w;

  //CheckCudaError(cuModuleGetTexRef(   &cu_tr_frame_0, cu_module_, "t_frame_0"));
  //CheckCudaError(cuTexRefSetFilterMode(cu_tr_frame_0, CU_TR_FILTER_MODE_POINT));
  //CheckCudaError(cuTexRefSetFormat(    cu_tr_frame_0, CU_AD_FORMAT_FLOAT, 1));
  //CheckCudaError(cuTexRefSetArray(     cu_tr_frame_0, dev_ar_frame_0, CU_TRSA_OVERRIDE_FORMAT));

  //CheckCudaError(cuModuleGetTexRef(   &cu_tr_frame_1, cu_module_, "t_frame_1"));
  //CheckCudaError(cuTexRefSetFilterMode(cu_tr_frame_1, CU_TR_FILTER_MODE_LINEAR)); // Linear interpolation only for frame_1
  //CheckCudaError(cuTexRefSetFormat(    cu_tr_frame_1, CU_AD_FORMAT_FLOAT, 1));
  //CheckCudaError(cuTexRefSetArray(     cu_tr_frame_1, dev_ar_frame_1, CU_TRSA_OVERRIDE_FORMAT));

  //CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_u, cu_module_, "t_flow_u"));
  //CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_u, CU_TR_FILTER_MODE_POINT));
  //CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_u, CU_AD_FORMAT_FLOAT, 1));
  //CheckCudaError(cuTexRefSetArray(     cu_tr_flow_u, dev_ar_flow_u, CU_TRSA_OVERRIDE_FORMAT));

  //CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_v, cu_module_, "t_flow_v"));
  //CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_v, CU_TR_FILTER_MODE_POINT));
  //CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_v, CU_AD_FORMAT_FLOAT, 1));
  //CheckCudaError(cuTexRefSetArray(     cu_tr_flow_v, dev_ar_flow_v, CU_TRSA_OVERRIDE_FORMAT));

  //CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_w, cu_module_, "t_flow_w"));
  //CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_w, CU_TR_FILTER_MODE_POINT));
  //CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_w, CU_AD_FORMAT_FLOAT, 1));
  //CheckCudaError(cuTexRefSetArray(     cu_tr_flow_w, dev_ar_flow_w, CU_TRSA_OVERRIDE_FORMAT));

  ///* Initialize kernel constants with sizes of a chunk */
  //size_t const_size;
  //CheckCudaError(cuModuleGetGlobal(&dev_constants_, &const_size, cu_module_, "container_size"));
  //CheckCudaError(     cuMemcpyHtoD(dev_constants_, &dev_data_size, const_size));

  ///* Use arrays for simplification */
  //const unsigned int ptrs_count = 5;

  //CUarray dev_ar[ptrs_count] = {
  //  dev_ar_frame_0,
  //  dev_ar_frame_1,
  //  dev_ar_flow_u,
  //  dev_ar_flow_v,
  //  dev_ar_flow_w
  //};

  //Data3D* d3d_ar[ptrs_count] = {
  //  &frame_0,
  //  &frame_1,
  //  &flow_u,
  //  &flow_v,
  //  &flow_w
  //};

  // /* Iterate over chunks and perform computations */
  //size_t z_start = 0;
  //size_t chunk_num = 0;
  //while (z_start < data_size.depth) {
  //  size_t z_end = std::min(data_size.depth, z_start + processing_slice_depth);
  //  size_t chunk_depth = z_end - z_start;

  //  size_t mem_usage = dev_data_size.pitch * dev_data_size.height * (chunk_depth + 2 * max_mag);
  //  std::printf("BRG Processing chunk: %3d - %3d Memory: %.2f MB\n", z_start, z_end, containers_count * mem_usage / (1024.f * 1024.f));

  //  CUDA_MEMCPY3D cu_copy3d;

  //  /* Copy input chunks to the device */
  //  for (unsigned int i = 0; i < ptrs_count; i++) {
  //    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

  //    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
  //    cu_copy3d.srcHost = d3d_ar[i]->DataPtr();
  //    cu_copy3d.srcPitch = d3d_ar[i]->Width() * sizeof(float);
  //    cu_copy3d.srcHeight = d3d_ar[i]->Height();

  //    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  //    cu_copy3d.dstArray = dev_ar[i];

  //    /* Processing data */
  //    cu_copy3d.dstZ = max_mag;

  //    cu_copy3d.srcZ = z_start; /**/

  //    cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
  //    cu_copy3d.Height = data_size.height;
  //    cu_copy3d.Depth = chunk_depth; /**/

  //    CheckCudaError(cuMemcpy3D(&cu_copy3d));

  //    /* Front border */
  //    size_t front_border_depth = std::min(max_mag, z_start);

  //    if (front_border_depth > 0) {
  //      cu_copy3d.dstZ = max_mag - front_border_depth;

  //      size_t front_z = z_start - front_border_depth;

  //      cu_copy3d.srcZ = front_z; /**/

  //      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
  //      cu_copy3d.Height = data_size.height;
  //      cu_copy3d.Depth = front_border_depth; /**/

  //      CheckCudaError(cuMemcpy3D(&cu_copy3d));
  //    }
  //    /* Rear border */
  //    size_t rear_border_depth = std::min(max_mag, data_size.depth - z_end);

  //    if (rear_border_depth > 0) {
  //      cu_copy3d.dstZ = max_mag + chunk_depth;

  //      size_t rear_z = z_end;

  //      cu_copy3d.srcZ = rear_z; /**/

  //      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
  //      cu_copy3d.Height = data_size.height;
  //      cu_copy3d.Depth = rear_border_depth; /**/

  //      CheckCudaError(cuMemcpy3D(&cu_copy3d));
  //    }

  //    //if (i == 0) {
  //    //  std::printf("Front border: %d  Rear border: %d\n", front_border_depth, rear_border_depth);
  //    //}
  //  }

  //  /* Launch kernel */
  //  dim3 block_dim = { 16, 8, 8 };
  //  dim3 grid_dim = { static_cast<unsigned int>((dev_data_size.width  + block_dim.x - 1) / block_dim.x),
  //                    static_cast<unsigned int>((dev_data_size.height + block_dim.y - 1) / block_dim.y),
  //                    static_cast<unsigned int>((chunk_depth          + block_dim.z - 1) / block_dim.z) };

  //  void* args[] = {
  //    &data_size.width,
  //    &data_size.height,
  //    &data_size.depth,
  //    &processing_slice_depth,
  //    &chunk_depth,
  //    &chunk_num,
  //    &max_mag,
  //    &hx,
  //    &hy,
  //    &hz,
  //    &dev_output
  //  };

  //CheckCudaError(cuLaunchKernel(cuf_registration_p_,
  //                              grid_dim.x, grid_dim.y, grid_dim.z,
  //                              block_dim.x, block_dim.y, block_dim.z,
  //                              0,
  //                              NULL,
  //                              args,
  //                              NULL));

  //  /* Copy results back to the host */
  //  std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

  //  cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  //  cu_copy3d.srcDevice = dev_output;
  //  cu_copy3d.srcPitch = dev_data_size.pitch;
  //  cu_copy3d.srcHeight = dev_data_size.height;

  //  cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
  //  cu_copy3d.dstHost = temp.DataPtr();
  //  cu_copy3d.dstPitch = temp.Width() * sizeof(float);
  //  cu_copy3d.dstHeight = temp.Height();

  //  cu_copy3d.dstXInBytes = 0;
  //  cu_copy3d.dstY = 0;
  //  cu_copy3d.dstZ = z_start; /**/
  //  
  //  cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
  //  cu_copy3d.Height = data_size.height;
  //  cu_copy3d.Depth = chunk_depth; /**/

  //  CheckCudaError(cuMemcpy3D(&cu_copy3d));

  //  z_start += processing_slice_depth;
  //  chunk_num++;
  //}

  ///* After registration is done swap two containers */
  //frame_1.Swap(temp);
  ////frame_1.WriteRAWToFileF32(std::string("./data/output/debug/frame_1_br-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());

  //CheckCudaError(cuMemFree(dev_output));

  //CheckCudaError(cuArrayDestroy(dev_ar_frame_0));
  //CheckCudaError(cuArrayDestroy(dev_ar_frame_1));
  //CheckCudaError(cuArrayDestroy(dev_ar_flow_u));
  //CheckCudaError(cuArrayDestroy(dev_ar_flow_v));
  //CheckCudaError(cuArrayDestroy(dev_ar_flow_w));
}