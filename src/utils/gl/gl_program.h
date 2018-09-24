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

#ifndef VIEWFLOW3D_UTILS_GL_GLPROGRAM_H_
#define VIEWFLOW3D_UTILS_GL_GLPROGRAM_H_

#include <string>
#include <unordered_map>

#include <GL/glew.h>

class GLProgram {
private:
  GLuint program_id_ = 0;
  std::unordered_map<std::string, GLint> uniforms_;

public:
  GLProgram() {};
  bool CreateAndLinkProgramFromSourceFiles(const char* vertex_shader_filename, const char* fragment_shader_filename);
  void Use();
  void Delete();
  GLuint GetProgramId() { return program_id_; };
  bool FindUniformLocation(const char* name);
  inline GLint GetUniformLocation(const char* name) { return uniforms_.at(name); };

};

#endif //!VIEWFLOW3D_UTILS_GL_GLPROGRAM_H_

