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

#ifndef VIEWFLOW3D_VISUALIZATION_H_
#define VIEWFLOW3D_VISUALIZATION_H_

#include <thread>
#include <mutex>
#include <condition_variable>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <AntTweakBar.h>

#include "src/data_types/data3d.h"
#include "src/utils/gl/gl_program.h"

class Visualization {
private:
  enum VisualizationMethod {
    VM_LINES, VM_GLYPHS, VM_CONES
  };

  bool initialized_ = false;
  bool has_texture_ = false;
  bool do_next_step_ = true;

  /* Dataset variables */
  size_t ds_width_;
  size_t ds_height_;
  size_t ds_depth_;

  size_t ds_sample_x_ = 10;
  size_t ds_sample_y_ = 10;
  size_t ds_sample_z_ = 10;
  
  float ds_min_;
  float ds_max_;
  float ds_avg_;

  float ds_scale_ = 1.f;
  float ds_threshold_ = 1.f;

  /* GLFW variables */
  size_t window_width_ = 1024;
  size_t window_height_ = 768;
  GLFWwindow* glfw_window_;
  GLFWwindow* glfw_thread_window_;

  /* OpenGL variables */
  VisualizationMethod gl_vis_method_ = VM_GLYPHS;

  glm::mat4 gl_m_projection_p_;
  glm::mat4 gl_m_projection_o_;
  glm::mat4 gl_m_view_;

  glm::quat gl_q_rotation_;

  glm::vec4 gl_clear_color_;

  GLProgram gl_prog_bbox_;
  GLProgram gl_prog_lines_;
  GLProgram gl_prog_r_lines_;

  GLProgram gl_prog_glyphs_;
  GLProgram gl_prog_r_glyphs_;

  GLProgram gl_prog_sprite_;

  GLuint gl_vao_bbox_;
  GLuint gl_vao_lines_;
  GLuint gl_vao_glyphs_;
  GLuint gl_vao_sprite_;
  GLuint gl_vao_cones_;

  GLuint gl_tex_flow_field_;
  GLuint gl_tex_colormap_;

  float gl_f_r_scale_ = 1.f;
  float gl_f_r_seed_ = 1.f;
  bool gl_b_jittered_ = false;

  float gl_f_zoom_ = 0.f;

  int gl_i_mag_dir_ = 0;
  
  bool gl_b_show_grid_ = false;

  int gl_i_cone_indices_ = 0;

  /* AntTweakBar variables */
  TwBar* ant_bar_;

  /* Thread variables */
  std::thread             thr_thread_;
  std::condition_variable thr_cv_initialized_;
  std::mutex              thr_m_lock_;

  /* Singletone */
  Visualization() {};
  Visualization(Visualization const&) = delete;
  void operator=(Visualization const&) = delete;

  /* Initialization routines */
  bool InitializeGLWindow();
  bool InitializeGLResources();
  bool InitializeGLGUI();

  void Render(GLFWwindow* window);

  /* Auxiliaru funciton */
  void UpdateFPSCounter(GLFWwindow* window);
  void UpdateDatasetSize(size_t width, size_t height, size_t depth);

  void GenerateCone(size_t face_count, GLuint gl_vao_cones, int* cone_vertices);

  /* Callback functions */
#ifdef _DEBUG
  static void GLFWErrorCallback(int error, const char* message);
  static void APIENTRY OpenGLDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
#endif

  static void GLFWWindowSizeCallback(GLFWwindow* window, int width, int height);
  static void GLFWCursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
  static void GLFWMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  static void GLFWScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
  static void GLFWKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void GLFWCharCallback(GLFWwindow* window, unsigned int codepoint);

  /* Threading */
  void ThreadBody();

public:
  /* The Singletone is used to allow access from static callbacks to private member variables */
  static Visualization& GetInstance()
  {
    static Visualization instance;
    return instance;
  }

  bool Initialize();
  void Show();
  void Destroy();

  inline bool IsInitialized() const { return initialized_; };
  
  /* Reads 3D flow field from 3 different raw files with U, V and W values; each file has width x height x depth dimensionality */
  bool Load3DFlowTextureFromRAWFileUVW(const char* filename_u, const char* filename_v, const char* filename_w, size_t width, size_t height, size_t depth);
  /* Reads 3D flow field from a raw file containing width, height and depth values at the beginning */
  bool Load3DFlowTextureFromRAWFileWHD(const char* filename);
  bool Load3DFlowTextureFromData3DUVW(Data3D& flow_u, Data3D& flow_v, Data3D&flow_w);

  /* Threading */
  void RunInSeparateThread();
  void WaitForInitialization();
  void WaitForFinalization();

  void DoNextStep();

  void SaveToFile();
};

#endif // !VIEWFLOW3D_VISUALIZATION_H_
