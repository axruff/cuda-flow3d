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

#ifndef VIEWFLOW3D_UTILS_GL_GLSHADER_H_
#define VIEWFLOW3D_UTILS_GL_GLSHADER_H_

#include <GL/glew.h>

class GLShader {
private:
  GLuint shader_id_ = 0;

public:
  GLShader() {};
  bool LoadFromFileAndCompile(const char* filename, GLenum shader_type);
  void Delete();
  GLuint GetShaderId() { return shader_id_; };
};

#endif // !VIEWFLOW3D_UTILS_GL_GLSHADER_H_
