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

#include "src/cuda_operations/partial_data/cuda_operation_solve_p.h"

#include <cstring>

#include <algorithm>
#include <vector_types.h>

#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationSolveP::CudaOperationSolveP()
  : CudaOperationBase("CUDA Sove Piecemeal")
{
}

bool CudaOperationSolveP::Initialize(const OperationParameters* params)
{
  initialized_ = false;

  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/solve_p_3d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_compute_phi_ksi_p_, cu_module_, "compute_phi_ksi_p_3d")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_solve_p_, cu_module_, "solve_p_3d"))
        ) {
      initialized_ = true;
    } else {
      CheckCudaError(cuModuleUnload(cu_module_));
      cu_module_ = nullptr;
    }
  }
  return initialized_;
}

void CudaOperationSolveP::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  Data3D* p_frame_0;
  Data3D* p_frame_1;
  Data3D* p_flow_u;
  Data3D* p_flow_v;
  Data3D* p_flow_w;
  Data3D* p_flow_du;
  Data3D* p_flow_dv;
  Data3D* p_flow_dw;
  Data3D* p_phi;
  Data3D* p_ksi;
  Data3D* p_temp_du;
  Data3D* p_temp_dv;
  Data3D* p_temp_dw;


  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_frame_0, "frame_0");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_frame_1, "frame_1");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_u,  "flow_u");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_v,  "flow_v");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_w,  "flow_w");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_du, "flow_du");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_dv, "flow_dv");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_flow_dw, "flow_dw");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_phi,     "phi");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_ksi,     "ksi");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_temp_du, "temp_du");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_temp_dv, "temp_dv");
  GET_PARAM_PTR_OR_RETURN(params, Data3D, p_temp_dw, "temp_dw");

  /* Use references just for convenience */
  Data3D& frame_0 = *p_frame_0;
  Data3D& frame_1 = *p_frame_1;
  Data3D& flow_u  = *p_flow_u;
  Data3D& flow_v  = *p_flow_v;
  Data3D& flow_w  = *p_flow_w;
  Data3D& flow_du = *p_flow_du;
  Data3D& flow_dv = *p_flow_dv;
  Data3D& flow_dw = *p_flow_dw;
  Data3D& phi     = *p_phi;
  Data3D& ksi     = *p_ksi;
  Data3D& temp_du = *p_temp_du;
  Data3D& temp_dv = *p_temp_dv;
  Data3D& temp_dw = *p_temp_dw;

  size_t outer_iterations_count;
  size_t inner_iterations_count;
  float equation_alpha;
  float equation_smoothness;
  float equation_data;
  float hx;
  float hy;
  float hz;
  DataSize4 data_size;

  GET_PARAM_OR_RETURN(params, size_t,    outer_iterations_count, "outer_iterations_count");
  GET_PARAM_OR_RETURN(params, size_t,    inner_iterations_count, "inner_iterations_count");
  GET_PARAM_OR_RETURN(params, float,     equation_alpha,         "equation_alpha");
  GET_PARAM_OR_RETURN(params, float,     equation_smoothness,    "equation_smoothness");
  GET_PARAM_OR_RETURN(params, float,     equation_data,          "equation_data");
  GET_PARAM_OR_RETURN(params, float,     hx,                     "hx");
  GET_PARAM_OR_RETURN(params, float,     hy,                     "hy");
  GET_PARAM_OR_RETURN(params, float,     hz,                     "hz");
  GET_PARAM_OR_RETURN(params, DataSize4, data_size,              "data_size");

  /* Check available memory on the cuda device and texture memory alignment */
  size_t total_memory;
  CheckCudaError(cuMemGetInfo(&free_memory_, &total_memory));
  std::printf("Free GPU memory: %.2f MB\n", free_memory_ / (1024.f * 1024.f));


  CUdevice cu_device;
  CheckCudaError(cuCtxGetDevice(&cu_device));
  CheckCudaError(cuDeviceGetAttribute(&alignment_, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, cu_device));

  alignment_ /= sizeof(float);

  /* Run solver kernels */

  /* Mesure execution time */
  size_t total_iterations_count = outer_iterations_count * inner_iterations_count;

  CUevent cu_event_start;
  CUevent cu_event_stop;

  CheckCudaError(cuEventCreate(&cu_event_start, CU_EVENT_DEFAULT));
  CheckCudaError(cuEventCreate(&cu_event_stop, CU_EVENT_DEFAULT));

  CheckCudaError(cuEventRecord(cu_event_start, NULL));

  /* Display computation status */
  if (!silent) {
      Utils::PrintProgressBar(0.f);
      std::printf(" % 3.0f%%", 0.f);
  }
  
  /* Initialize flow increment buffers with 0 */
  flow_du.ZeroData();
  flow_dv.ZeroData();
  flow_dw.ZeroData();


  if (!silent)
    std::printf("\n");

  for (size_t i = 0; i < outer_iterations_count; ++i) {
    ComputePhiKsi(frame_0, frame_1, flow_u, flow_v, flow_w, flow_du, flow_dv, flow_dw, data_size, hx, hy, hz, equation_smoothness, equation_data, phi, ksi);
    //phi.WriteRAWToFileF32(std::string("./data/output/debug/p/phi_slv-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());
    //ksi.WriteRAWToFileF32(std::string("./data/output/debug/p/ksi_slv-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());

    //frame_0.WriteRAWToFileF32(std::string("./data/output/debug/p/frame_0-pk_slv-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());
    //frame_1.WriteRAWToFileF32(std::string("./data/output/debug/p/frame_1-pk_slv-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());
    for (size_t j = 0; j < inner_iterations_count; ++j) {
      /* Solve */
      Solve(frame_0, frame_1, flow_u, flow_v, flow_w, flow_du, flow_dv, flow_dw, phi, ksi, data_size, hx, hy, hz, equation_alpha, temp_du, temp_dv, temp_dw);

      flow_du.Swap(temp_du);
      flow_dv.Swap(temp_dv);
      flow_dw.Swap(temp_dw);

      //flow_du.WriteRAWToFileF32(std::string("./data/output/debug/p/flow_du_slv-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());
      //flow_dv.WriteRAWToFileF32(std::string("./data/output/debug/p/flow_dv_slv-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());
      //flow_dw.WriteRAWToFileF32(std::string("./data/output/debug/p/flow_dw_slv-180-180-151-" + std::to_string(data_size.width) + std::string(".raw")).c_str());

      /* Display computation status */
      if (!silent) {
          float complete = (i * inner_iterations_count + j) / static_cast<float>(total_iterations_count);
          Utils::PrintProgressBar(complete);
          std::printf(" % 3.0f%%", complete * 100);
      }
    }
  }
  /* Estimate GPU computation time */
  CheckCudaError(cuEventRecord(cu_event_stop, NULL));
  CheckCudaError(cuEventSynchronize(cu_event_stop));

  float elapsed_time;
  CheckCudaError(cuEventElapsedTime(&elapsed_time, cu_event_start, cu_event_stop));
  
  if (!silent) {
      Utils::PrintProgressBar(1.f);
      std::printf(" %8.4fs\n", elapsed_time / 1000.);
  }
}

void CudaOperationSolveP::ComputePhiKsi(
  Data3D& frame_0, Data3D& frame_1,
  Data3D& flow_u, Data3D& flow_v, Data3D& flow_w,
  Data3D& flow_du, Data3D& flow_dv, Data3D& flow_dw,
  DataSize4 data_size,
  float hx, float hy, float hz,
  float equation_smoothness, float equation_data,
  Data3D& phi, Data3D& ksi)
{
  //std::printf("\nCompute Phi Ksi\n");

  /* Debug */
  //free_memory_ = 64.f * 1024 * 1024;
  /* Estimate a slice depth for partitioning */
  data_size.pitch = ((data_size.width % alignment_ == 0) ?
                     (data_size.width) :
                     (data_size.width + alignment_ - (data_size.width % alignment_))) *
                     sizeof(float);

  size_t mem = data_size.pitch * data_size.height * data_size.depth;

  const unsigned int containers_count = 10;
  unsigned int max_slice_depth = free_memory_ / (containers_count * data_size.pitch * data_size.height);
  unsigned int total_slice_memory = containers_count * data_size.pitch * data_size.height * max_slice_depth;

  const unsigned int min_slice_depth = 3;
  if (max_slice_depth < min_slice_depth) {
    std::printf("Error. Low GPU memory. Data cannot be partitioned properly. Try to reduce the input size of dataset.\n");
    return;
  }
  unsigned int processing_slice_depth = std::min(static_cast<unsigned int>(data_size.depth), max_slice_depth - 2);

  //std::printf("Max slice depth: %d Total slice memory: %.2f MB\n", max_slice_depth, total_slice_memory / (1024.f * 1024.f));
  //std::printf("Free memory: %.2f MB Container size: %.2f MB Total size: %.2f MB\n",
  //  free_memory_ / (1024.f * 1024.f), mem / (1024.f * 1024.f), (containers_count * mem) / (1024.f * 1024.f));

  //std::printf("Max slice depth: %d Processing slice depth: %d\n\n", max_slice_depth, processing_slice_depth);
 
  /* Allocate CUDA memory on the device (CUDA Arrays for the input (textures), Pitched memory for the output) */
  DataSize4 dev_data_size = data_size;
  dev_data_size.depth = processing_slice_depth + 2;
  
  CUarray dev_ar_frame_0;
  CUarray dev_ar_frame_1;
  CUarray dev_ar_flow_u;
  CUarray dev_ar_flow_v;
  CUarray dev_ar_flow_w;
  CUarray dev_ar_flow_du;
  CUarray dev_ar_flow_dv;
  CUarray dev_ar_flow_dw;

  CUdeviceptr dev_phi;
  CUdeviceptr dev_ksi;

  CUDA_ARRAY3D_DESCRIPTOR cu_array_descriptor;

  cu_array_descriptor.Format = CU_AD_FORMAT_FLOAT;
  cu_array_descriptor.NumChannels = 1;
  cu_array_descriptor.Width = dev_data_size.width;
  cu_array_descriptor.Height = dev_data_size.height;
  cu_array_descriptor.Depth = dev_data_size.depth;
  cu_array_descriptor.Flags = 0;

  CheckCudaError(cuArray3DCreate(&dev_ar_frame_0, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_frame_1, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_u,  &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_v,  &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_w,  &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_du, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_dv, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_dw, &cu_array_descriptor));

  CheckCudaError(cuMemAlloc(&dev_phi, dev_data_size.pitch * dev_data_size.height * (dev_data_size.depth - 2)));
  CheckCudaError(cuMemAlloc(&dev_ksi, dev_data_size.pitch * dev_data_size.height * (dev_data_size.depth - 2)));

  /* Bind textures */
  CUtexref cu_tr_frame_0;
  CUtexref cu_tr_frame_1;
  CUtexref cu_tr_flow_u;
  CUtexref cu_tr_flow_v;
  CUtexref cu_tr_flow_w;
  CUtexref cu_tr_flow_du;
  CUtexref cu_tr_flow_dv;
  CUtexref cu_tr_flow_dw;

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_frame_0, cu_module_, "t_frame_0"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_frame_0, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_frame_0, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_frame_0, dev_ar_frame_0, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_frame_1, cu_module_, "t_frame_1"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_frame_1, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_frame_1, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_frame_1, dev_ar_frame_1, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_u, cu_module_, "t_flow_u"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_u, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_u, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_u, dev_ar_flow_u, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_v, cu_module_, "t_flow_v"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_v, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_v, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_v, dev_ar_flow_v, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_w, cu_module_, "t_flow_w"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_w, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_w, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_w, dev_ar_flow_w, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_du, cu_module_, "t_flow_du"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_du, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_du, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_du, dev_ar_flow_du, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_dv, cu_module_, "t_flow_dv"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_dv, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_dv, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_dv, dev_ar_flow_dv, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_dw, cu_module_, "t_flow_dw"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_dw, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_dw, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_dw, dev_ar_flow_dw, CU_TRSA_OVERRIDE_FORMAT));

  /* Initialize kernel constants with sizes of a chunk */
  size_t const_size;
  CheckCudaError(cuModuleGetGlobal(&dev_constants_, &const_size, cu_module_, "container_size"));
  CheckCudaError(     cuMemcpyHtoD(dev_constants_, &dev_data_size, const_size));

  /* Use arrays for simplification */
  const unsigned int ptrs_count = 8;

  CUarray dev_ar[ptrs_count] = {
    dev_ar_frame_0,
    dev_ar_frame_1,
    dev_ar_flow_u,
    dev_ar_flow_v,
    dev_ar_flow_w,
    dev_ar_flow_du,
    dev_ar_flow_dv,
    dev_ar_flow_dw
  };

  Data3D* d3d_ar[ptrs_count] = {
    &frame_0,
    &frame_1,
    &flow_u,
    &flow_v,
    &flow_w,
    &flow_du,
    &flow_dv,
    &flow_dw
  };

   /* Iterate over chunks and perform computations */
  unsigned int z_start = 0;
  while (z_start < data_size.depth) {
    unsigned int z_end = std::min(static_cast<unsigned int>(data_size.depth), z_start + processing_slice_depth);
    size_t chunk_depth = z_end - z_start;

    size_t mem_usage = dev_data_size.pitch * dev_data_size.height * (chunk_depth + 2);
    //std::printf("CPK Processing chunk: %3d - %3d Memory: %.2f MB\n", z_start, z_end, containers_count * mem_usage / (1024.f * 1024.f));

    CUDA_MEMCPY3D cu_copy3d;

    /* Copy input chunks to the device */
    for (unsigned int i = 0; i < ptrs_count; i++) {
      std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

      cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
      cu_copy3d.srcHost = d3d_ar[i]->DataPtr();
      cu_copy3d.srcPitch = d3d_ar[i]->Width() * sizeof(float);
      cu_copy3d.srcHeight = d3d_ar[i]->Height();

      cu_copy3d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      cu_copy3d.dstArray = dev_ar[i];

      /* Processing data */
      cu_copy3d.dstZ = 1;

      cu_copy3d.srcZ = z_start; /**/

      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
      cu_copy3d.Height = data_size.height;
      cu_copy3d.Depth = chunk_depth; /**/

      CheckCudaError(cuMemcpy3D(&cu_copy3d));

      /* Front border */
      cu_copy3d.dstZ = 0;

      unsigned int front_z = (z_start == 0) ? 1 : z_start - 1;

      cu_copy3d.srcZ = front_z; /**/

      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
      cu_copy3d.Height = data_size.height;
      cu_copy3d.Depth = 1; /**/
      
      CheckCudaError(cuMemcpy3D(&cu_copy3d));

      /* Rear border */
      cu_copy3d.dstZ = 1 + chunk_depth;

      unsigned int rear_z = (z_end == data_size.depth) ? data_size.depth - 2 : z_end;

      cu_copy3d.srcZ = rear_z; /**/

      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
      cu_copy3d.Height = data_size.height;
      cu_copy3d.Depth = 1; /**/
      
      CheckCudaError(cuMemcpy3D(&cu_copy3d));
    }

    /* Launch kernel */
    dim3 block_dim = { 16, 8, 8 };
    dim3 grid_dim = { static_cast<unsigned int>((dev_data_size.width  + block_dim.x - 1) / block_dim.x),
                      static_cast<unsigned int>((dev_data_size.height + block_dim.y - 1) / block_dim.y),
                      static_cast<unsigned int>((chunk_depth          + block_dim.z - 1) / block_dim.z) };

    void* args[] = {
      &dev_data_size.width,
      &dev_data_size.height,
      &chunk_depth,
      &hx,
      &hy,
      &hz,
      &equation_smoothness,
      &equation_data,
      &dev_phi,
      &dev_ksi
    };

    CheckCudaError(cuLaunchKernel(cuf_compute_phi_ksi_p_,
                                  grid_dim.x, grid_dim.y, grid_dim.z,
                                  block_dim.x, block_dim.y, block_dim.z,
                                  0,
                                  NULL,
                                  args,
                                  NULL));

    /* Copy results back to the host */
    /* Phi */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_phi;
    cu_copy3d.srcPitch = dev_data_size.pitch;
    cu_copy3d.srcHeight = dev_data_size.height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = phi.DataPtr();
    cu_copy3d.dstPitch = phi.Width() * sizeof(float);
    cu_copy3d.dstHeight = phi.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = 0;
    cu_copy3d.dstZ = z_start; /**/
    
    cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
    cu_copy3d.Height = data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Ksi */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_ksi;
    cu_copy3d.srcPitch = dev_data_size.pitch;
    cu_copy3d.srcHeight = dev_data_size.height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = ksi.DataPtr();
    cu_copy3d.dstPitch = ksi.Width() * sizeof(float);
    cu_copy3d.dstHeight = ksi.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = 0;
    cu_copy3d.dstZ = z_start; /**/
    
    cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
    cu_copy3d.Height = data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    z_start += processing_slice_depth;
  }

  CheckCudaError(cuMemFree(dev_phi));
  CheckCudaError(cuMemFree(dev_ksi));

  CheckCudaError(cuArrayDestroy(dev_ar_frame_0));
  CheckCudaError(cuArrayDestroy(dev_ar_frame_1));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_u));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_v));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_w));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_du));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_dv));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_dw));
}

void CudaOperationSolveP::Solve(
  Data3D& frame_0, Data3D& frame_1,
  Data3D& flow_u, Data3D& flow_v, Data3D& flow_w,
  Data3D& flow_du, Data3D& flow_dv, Data3D& flow_dw,
  Data3D& phi, Data3D& ksi,
  DataSize4 data_size,
  float hx, float hy, float hz,
  float equation_alpha,
  Data3D& flow_output_du, Data3D& flow_output_dv, Data3D& flow_output_dw)
{ 
  //std::printf("\nSolve\n");

  /* Debug */
  //free_memory_ = 64.f * 1024 * 1024;
  /* Estimate a slice depth for partitioning */
  data_size.pitch = ((data_size.width % alignment_ == 0) ?
                     (data_size.width) :
                     (data_size.width + alignment_ - (data_size.width % alignment_))) *
                     sizeof(float);

  size_t mem = data_size.pitch * data_size.height * data_size.depth;

  const size_t containers_count = 13;
  size_t max_slice_depth = free_memory_ / (containers_count * data_size.pitch * data_size.height);
  size_t total_slice_memory = containers_count * data_size.pitch * data_size.height * max_slice_depth;

  const size_t min_slice_depth = 3;
  if (max_slice_depth < min_slice_depth) {
    std::printf("Error. Low GPU memory. Data cannot be partitioned properly. Try to reduce the input size of dataset.\n");
    return;
  }
  size_t processing_slice_depth = std::min(data_size.depth, max_slice_depth - 2);

  //std::printf("Max slice depth: %d Total slice memory: %.2f MB\n", max_slice_depth, total_slice_memory / (1024.f * 1024.f));
  //std::printf("Free memory: %.2f MB Container size: %.2f MB Total size: %.2f MB\n",
  //  free_memory_ / (1024.f * 1024.f), mem / (1024.f * 1024.f), (containers_count * mem) / (1024.f * 1024.f));

  //std::printf("Max slice depth: %d Processing slice depth: %d\n", max_slice_depth, processing_slice_depth);

  /* Allocate CUDA memory on the device (CUDA Arrays for the input (textures), Pitched memory for the output) */
  DataSize4 dev_data_size = data_size;
  dev_data_size.depth = processing_slice_depth + 2;
  
  CUarray dev_ar_frame_0;
  CUarray dev_ar_frame_1;
  CUarray dev_ar_flow_u;
  CUarray dev_ar_flow_v;
  CUarray dev_ar_flow_w;
  CUarray dev_ar_flow_du;
  CUarray dev_ar_flow_dv;
  CUarray dev_ar_flow_dw;
  CUarray dev_ar_phi;
  CUarray dev_ar_ksi;

  CUdeviceptr dev_output_du;
  CUdeviceptr dev_output_dv;
  CUdeviceptr dev_output_dw;

  CUDA_ARRAY3D_DESCRIPTOR cu_array_descriptor;

  cu_array_descriptor.Format = CU_AD_FORMAT_FLOAT;
  cu_array_descriptor.NumChannels = 1;
  cu_array_descriptor.Width = dev_data_size.width;
  cu_array_descriptor.Height = dev_data_size.height;
  cu_array_descriptor.Depth = dev_data_size.depth;
  cu_array_descriptor.Flags = 0;

  CheckCudaError(cuArray3DCreate(&dev_ar_frame_0, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_frame_1, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_u,  &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_v,  &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_w,  &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_du, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_dv, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_flow_dw, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_phi, &cu_array_descriptor));
  CheckCudaError(cuArray3DCreate(&dev_ar_ksi, &cu_array_descriptor));

  CheckCudaError(cuMemAlloc(&dev_output_du, dev_data_size.pitch * dev_data_size.height * (dev_data_size.depth - 2)));
  CheckCudaError(cuMemAlloc(&dev_output_dv, dev_data_size.pitch * dev_data_size.height * (dev_data_size.depth - 2)));
  CheckCudaError(cuMemAlloc(&dev_output_dw, dev_data_size.pitch * dev_data_size.height * (dev_data_size.depth - 2)));

/* Bind textures */
  CUtexref cu_tr_frame_0;
  CUtexref cu_tr_frame_1;
  CUtexref cu_tr_flow_u;
  CUtexref cu_tr_flow_v;
  CUtexref cu_tr_flow_w;
  CUtexref cu_tr_flow_du;
  CUtexref cu_tr_flow_dv;
  CUtexref cu_tr_flow_dw;
  CUtexref cu_tr_phi;
  CUtexref cu_tr_ksi;

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_frame_0, cu_module_, "t_frame_0"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_frame_0, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_frame_0, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_frame_0, dev_ar_frame_0, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_frame_1, cu_module_, "t_frame_1"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_frame_1, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_frame_1, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_frame_1, dev_ar_frame_1, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_u, cu_module_, "t_flow_u"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_u, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_u, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_u, dev_ar_flow_u, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_v, cu_module_, "t_flow_v"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_v, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_v, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_v, dev_ar_flow_v, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_w, cu_module_, "t_flow_w"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_w, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_w, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_w, dev_ar_flow_w, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_du, cu_module_, "t_flow_du"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_du, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_du, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_du, dev_ar_flow_du, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_dv, cu_module_, "t_flow_dv"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_dv, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_dv, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_dv, dev_ar_flow_dv, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_flow_dw, cu_module_, "t_flow_dw"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_flow_dw, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_flow_dw, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_flow_dw, dev_ar_flow_dw, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_phi, cu_module_, "t_phi"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_phi, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_phi, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_phi, dev_ar_phi, CU_TRSA_OVERRIDE_FORMAT));

  CheckCudaError(cuModuleGetTexRef(   &cu_tr_ksi, cu_module_, "t_ksi"));
  CheckCudaError(cuTexRefSetFilterMode(cu_tr_ksi, CU_TR_FILTER_MODE_POINT));
  CheckCudaError(cuTexRefSetFormat(    cu_tr_ksi, CU_AD_FORMAT_FLOAT, 1));
  CheckCudaError(cuTexRefSetArray(     cu_tr_ksi, dev_ar_ksi, CU_TRSA_OVERRIDE_FORMAT));

  /* Initialize kernel constants with sizes of a chunk */
  size_t const_size;
  CheckCudaError(cuModuleGetGlobal(&dev_constants_, &const_size, cu_module_, "container_size"));
  CheckCudaError(     cuMemcpyHtoD(dev_constants_, &dev_data_size, const_size));

  /* Use arrays for simplification */
  const unsigned int ptrs_count = 10;

  CUarray dev_ar[ptrs_count] = {
    dev_ar_frame_0,
    dev_ar_frame_1,
    dev_ar_flow_u,
    dev_ar_flow_v,
    dev_ar_flow_w,
    dev_ar_flow_du,
    dev_ar_flow_dv,
    dev_ar_flow_dw,
    dev_ar_phi,
    dev_ar_ksi
  };

  Data3D* d3d_ar[ptrs_count] = {
    &frame_0,
    &frame_1,
    &flow_u,
    &flow_v,
    &flow_w,
    &flow_du,
    &flow_dv,
    &flow_dw,
    &phi,
    &ksi
  };

  /* Iterate over chunks and perform computations */
  size_t z_start = 0;
  while (z_start < data_size.depth) {
    size_t z_end = std::min(data_size.depth, z_start + processing_slice_depth);
    size_t chunk_depth = z_end - z_start;

    size_t mem_usage = dev_data_size.pitch * dev_data_size.height * (chunk_depth + 2);
    //std::printf("SLV Processing chunk: %3d - %3d Memory: %.2f MB\n", z_start, z_end, containers_count * mem_usage / (1024.f * 1024.f));

    CUDA_MEMCPY3D cu_copy3d;

    /* Copy input chunks to the device */
    for (unsigned int i = 0; i < ptrs_count; i++) {
      std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

      cu_copy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
      cu_copy3d.srcHost = d3d_ar[i]->DataPtr();
      cu_copy3d.srcPitch = d3d_ar[i]->Width() * sizeof(float);
      cu_copy3d.srcHeight = d3d_ar[i]->Height();

      cu_copy3d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      cu_copy3d.dstArray = dev_ar[i];

      /* Processing data */
      cu_copy3d.dstZ = 1;

      cu_copy3d.srcZ = z_start; /**/

      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
      cu_copy3d.Height = data_size.height;
      cu_copy3d.Depth = chunk_depth; /**/

      CheckCudaError(cuMemcpy3D(&cu_copy3d));

      /* Front border */
      cu_copy3d.dstZ = 0;

      size_t front_z = (z_start == 0) ? 1 : z_start - 1;

      cu_copy3d.srcZ = front_z; /**/

      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
      cu_copy3d.Height = data_size.height;
      cu_copy3d.Depth = 1; /**/
      
      CheckCudaError(cuMemcpy3D(&cu_copy3d));

      /* Rear border */
      cu_copy3d.dstZ = 1 + chunk_depth;

      size_t rear_z = (z_end == data_size.depth) ? data_size.depth - 2 : z_end;

      cu_copy3d.srcZ = rear_z; /**/

      cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
      cu_copy3d.Height = data_size.height;
      cu_copy3d.Depth = 1; /**/
      
      CheckCudaError(cuMemcpy3D(&cu_copy3d));
    }

    /* Launch kernel */
    dim3 block_dim = { 16, 8, 8 };
    dim3 grid_dim = { static_cast<unsigned int>((dev_data_size.width  + block_dim.x - 1) / block_dim.x),
                      static_cast<unsigned int>((dev_data_size.height + block_dim.y - 1) / block_dim.y),
                      static_cast<unsigned int>((chunk_depth          + block_dim.z - 1) / block_dim.z) };

    void* args[] = {
      &data_size.width,
      &data_size.height,
      &data_size.depth,
      &chunk_depth,
      &z_start,
      &hx,
      &hy,
      &hz,
      &equation_alpha,
      &dev_output_du,
      &dev_output_dv,
      &dev_output_dw
    };

    CheckCudaError(cuLaunchKernel(cuf_solve_p_,
                                  grid_dim.x, grid_dim.y, grid_dim.z,
                                  block_dim.x, block_dim.y, block_dim.z,
                                  0,
                                  NULL,
                                  args,
                                  NULL));

    /* Copy results back to the host */
    /* Ouput DU */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_output_du;
    cu_copy3d.srcPitch = dev_data_size.pitch;
    cu_copy3d.srcHeight = dev_data_size.height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = flow_output_du.DataPtr();
    cu_copy3d.dstPitch = flow_output_du.Width() * sizeof(float);
    cu_copy3d.dstHeight = flow_output_du.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = 0;
    cu_copy3d.dstZ = z_start; /**/
    
    cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
    cu_copy3d.Height = data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Ouput DV */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_output_dv;
    cu_copy3d.srcPitch = dev_data_size.pitch;
    cu_copy3d.srcHeight = dev_data_size.height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = flow_output_dv.DataPtr();
    cu_copy3d.dstPitch = flow_output_dv.Width() * sizeof(float);
    cu_copy3d.dstHeight = flow_output_dv.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = 0;
    cu_copy3d.dstZ = z_start; /**/
    
    cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
    cu_copy3d.Height = data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    /* Ouput DW */
    std::memset(&cu_copy3d, 0, sizeof(CUDA_MEMCPY3D));

    cu_copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy3d.srcDevice = dev_output_dw;
    cu_copy3d.srcPitch = dev_data_size.pitch;
    cu_copy3d.srcHeight = dev_data_size.height;

    cu_copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy3d.dstHost = flow_output_dw.DataPtr();
    cu_copy3d.dstPitch = flow_output_dw.Width() * sizeof(float);
    cu_copy3d.dstHeight = flow_output_dw.Height();

    cu_copy3d.dstXInBytes = 0;
    cu_copy3d.dstY = 0;
    cu_copy3d.dstZ = z_start; /**/
    
    cu_copy3d.WidthInBytes = data_size.width * sizeof(float);
    cu_copy3d.Height = data_size.height;
    cu_copy3d.Depth = chunk_depth; /**/

    CheckCudaError(cuMemcpy3D(&cu_copy3d));

    z_start += processing_slice_depth;
  }

  CheckCudaError(cuMemFree(dev_output_du));
  CheckCudaError(cuMemFree(dev_output_dv));
  CheckCudaError(cuMemFree(dev_output_dw));

  CheckCudaError(cuArrayDestroy(dev_ar_frame_0));
  CheckCudaError(cuArrayDestroy(dev_ar_frame_1));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_u));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_v));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_w));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_du));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_dv));
  CheckCudaError(cuArrayDestroy(dev_ar_flow_dw));
  CheckCudaError(cuArrayDestroy(dev_ar_phi));
  CheckCudaError(cuArrayDestroy(dev_ar_ksi));
}