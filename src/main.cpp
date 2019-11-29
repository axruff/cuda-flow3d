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

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda.h>

#include "src/optical_flow/optical_flow_e.h"
#include "src/optical_flow/optical_flow_p.h"

#include "src/data_types/data3d.h"
#include "src/data_types/data_structs.h"
#include "src/data_types/operation_parameters.h"
#include "src/utils/cuda_utils.h"

#ifndef NO_VISUALIZATION
    #include "src/visualization.h"
#endif


using namespace std;

std::string ZeroPadNumber(int num, int pad_num)
{
    std::ostringstream ss;

    ss.fill('0');

    ss << setw(pad_num) << num;
    return ss.str(); 
}


int main(int argc, char** argv)
{

  const bool key_press = false;
  const bool use_visualization = false;
  const bool silent_mode = false;

  const bool use_entire_gpu = true;
  const bool use_partial_gpu = false;

  /* Dataset variables */
  //const size_t width = 128;
  //const size_t height = 128;
  //const size_t depth = 128;

  //const size_t width = 584;
  //const size_t height = 388;
  //const size_t depth = 5;

  const size_t width = 450;
  const size_t height = 180;
  const size_t depth = 450;

  /* Optical flow variables */
  size_t  warp_levels_count       = 40;
  float   warp_scale_factor       = 0.95f;
  size_t  outer_iterations_count  = 40;
  size_t  inner_iterations_count  = 5;
  float   equation_alpha          = 7.5f;
  float   equation_smoothness     = 0.001f;
  float   equation_data           = 0.001f;
  size_t  median_radius           = 5;
  float   gaussian_sigma          = 2.0f;

  OpticalFlowE optical_flow_e;
  OpticalFlowP optical_flow_p;

  CUcontext cu_context;
  Data3D frame_0;
  Data3D frame_1;
  DataSize4 data_size = {width, height, depth, 0};

  std::printf("//----------------------------------------------------------------------//\n");
  std::printf("//            3D Optical flow using NVIDIA CUDA. Version 0.6.0	         //\n");
  std::printf("//                                                                      //\n");
  std::printf("//            Karlsruhe Institute of Technology. 2015 - 2018            //\n");
  std::printf("//----------------------------------------------------------------------//\n");

  int pd = 3;
  int start = 0;
  int end = 0;

  //string path = "./data/"; 
  std::string path = "/mnt/LSDF/tomo/ershov/cork/2019-11-InSitu-MotionAnalysis/Spray_61a/";
  
  std::string file_name = "_scaled2_450-450-180.raw";
  
  
  if (argc > 2)  {
        start = atoi(argv[1]);
        end = atoi(argv[2]);
  }
        

  /* Initialize CUDA */
  if (!InitCudaContextWithFirstAvailableDevice(&cu_context)) {
    return 1;
  }
  /* Load input data */
  //if (!frame_0.ReadRAWFromFileU8("./data/frame_0_128-128-128.raw", data_size.width, data_size.height, data_size.depth) ||
  //    !frame_1.ReadRAWFromFileU8("./data/frame_1_128-128-128.raw", data_size.width, data_size.height, data_size.depth)) {
  //  return 2;
  //}

  //if (!frame_0.ReadRAWFromFileU8("./data/rub1-584-388-5.raw", data_size.width, data_size.height, data_size.depth) ||
  //    !frame_1.ReadRAWFromFileU8("./data/rub2-584-388-5.raw", data_size.width, data_size.height, data_size.depth)) {
  //    return 2;
  //}
  
for (int i=start; i<end; i++) {


  if (!frame_0.ReadRAWFromFileU8((path + "input/" + ZeroPadNumber(i, pd) + "_scaled2_450-450-180.raw").c_str(), data_size.width, data_size.height, data_size.depth) ||
      !frame_1.ReadRAWFromFileU8((path + "input/" + ZeroPadNumber(i + 1, pd) + "_scaled2_450-450-180.raw").c_str(), data_size.width, data_size.height, data_size.depth)) {
      return 2;
  }

#ifndef NO_VISUALIZATION
  Visualization& visualization = Visualization::GetInstance();

  if (use_visualization) {
    visualization.RunInSeparateThread();
    visualization.WaitForInitialization();
  }
#endif

  /* OpticalFlow that stores all data on the one gpu */
  if (use_entire_gpu && optical_flow_e.Initialize(data_size)) {

    Data3D flow_u(data_size.width, data_size.height, data_size.depth);
    Data3D flow_v(data_size.width, data_size.height, data_size.depth);
    Data3D flow_w(data_size.width, data_size.height, data_size.depth);

    OperationParameters params;
    params.PushValuePtr("warp_levels_count",      &warp_levels_count);
    params.PushValuePtr("warp_scale_factor",      &warp_scale_factor);
    params.PushValuePtr("outer_iterations_count", &outer_iterations_count);
    params.PushValuePtr("inner_iterations_count", &inner_iterations_count);
    params.PushValuePtr("equation_alpha",         &equation_alpha);
    params.PushValuePtr("equation_smoothness",    &equation_smoothness);
    params.PushValuePtr("equation_data",          &equation_data);
    params.PushValuePtr("median_radius",          &median_radius);
    params.PushValuePtr("gaussian_sigma",         &gaussian_sigma);

    std::printf("Mode: Full GPU mode \n");

    optical_flow_e.silent = silent_mode;

    optical_flow_e.ComputeFlow(frame_0, frame_1, flow_u, flow_v, flow_w, params);
    
    std::string filename =
      "-" + std::to_string(width) +
      "-" + std::to_string(height) +
      "-" + std::to_string(depth) + ".raw";

    flow_u.WriteRAWToFileF32(std::string(path + "output/" + ZeroPadNumber(i, pd) + "_flow-u" + filename).c_str());
    //flow_v.WriteRAWToFileF32(std::string(path + "output/flow-v" + filename).c_str());
    //flow_w.WriteRAWToFileF32(std::string(path + "output/flow-w" + filename).c_str());
    flow_v.WriteRAWToFileF32(std::string(path + "output/" + ZeroPadNumber(i, pd) + "_flow-v" + filename).c_str());
    flow_w.WriteRAWToFileF32(std::string(path + "output/" + ZeroPadNumber(i, pd) + "_flow-w" + filename).c_str());
    
    optical_flow_e.Destroy();
  }

  if (use_partial_gpu && optical_flow_p.Initialize(data_size)) {

    Data3D flow_u(data_size.width, data_size.height, data_size.depth);
    Data3D flow_v(data_size.width, data_size.height, data_size.depth);
    Data3D flow_w(data_size.width, data_size.height, data_size.depth);

    OperationParameters params;
    params.PushValuePtr("warp_levels_count",      &warp_levels_count);
    params.PushValuePtr("warp_scale_factor",      &warp_scale_factor);
    params.PushValuePtr("outer_iterations_count", &outer_iterations_count);
    params.PushValuePtr("inner_iterations_count", &inner_iterations_count);
    params.PushValuePtr("equation_alpha",         &equation_alpha);
    params.PushValuePtr("equation_smoothness",    &equation_smoothness);
    params.PushValuePtr("equation_data",          &equation_data);
    params.PushValuePtr("median_radius",          &median_radius);
    params.PushValuePtr("gaussian_sigma",         &gaussian_sigma);

    std::printf("Mode: Partial processing mode \n");

    optical_flow_p.silent = silent_mode;

    optical_flow_p.ComputeFlow(frame_0, frame_1, flow_u, flow_v, flow_w, params);
    
    std::string filename =
      "-" + std::to_string(width) +
      "-" + std::to_string(height) +
      "-" + std::to_string(depth) + "-partial.raw";

    flow_u.WriteRAWToFileF32(std::string("./data/output/flow-u" + filename).c_str());
    flow_v.WriteRAWToFileF32(std::string("./data/output/flow-v" + filename).c_str());
    flow_w.WriteRAWToFileF32(std::string("./data/output/flow-w" + filename).c_str());

    optical_flow_p.Destroy();
  }
#ifndef NO_VISUALIZATION
  if (use_visualization) {
    visualization.WaitForFinalization();
  }
#endif


}

  if (key_press) {
    std::printf("Press enter to continue...");
    std::getchar();
  }

  /* Release resources */
  cuCtxDestroy(cu_context);

  return 0;
}
