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

#include "src/utils/gl/gl_program.h"

#include <cstdio>

#include "src/utils/gl/gl_shader.h"

bool GLProgram::CreateAndLinkProgramFromSourceFiles(const char* vertex_shader_filename, const char* fragment_shader_filename)
{
  GLShader vertex_shader;
  GLShader fragment_shader;

  if (!vertex_shader.LoadFromFileAndCompile(vertex_shader_filename, GL_VERTEX_SHADER)) {
    return false;
  }

  if (!fragment_shader.LoadFromFileAndCompile(fragment_shader_filename, GL_FRAGMENT_SHADER)) {
    return false;
  }

  program_id_ = glCreateProgram();
  if (program_id_ == 0) {
    std::printf("Error. Program creation failed:\n%s\n%s\n", vertex_shader_filename, fragment_shader_filename);
    return false;
  }

  glAttachShader(program_id_, vertex_shader.GetShaderId());
  glAttachShader(program_id_, fragment_shader.GetShaderId());
  glLinkProgram(program_id_);

  glDetachShader(program_id_, vertex_shader.GetShaderId());
  glDetachShader(program_id_, fragment_shader.GetShaderId());

  vertex_shader.Delete();
  fragment_shader.Delete();

  GLint link_status;
  glGetProgramiv(program_id_, GL_LINK_STATUS, &link_status);

  if (link_status == GL_FALSE) {
    GLchar info_log[512];
    glGetProgramInfoLog(program_id_, 512, NULL, info_log);
    std::printf("Error. Program linking failed:\n%s\n%s\n", vertex_shader_filename, fragment_shader_filename);
    std::printf("%s\n", info_log);

    Delete();
    return false;
  }

  return true;
}

void GLProgram::Use()
{
  glUseProgram(program_id_);
}

bool GLProgram::FindUniformLocation(const char* name)
{
  GLint location = glGetUniformLocation(program_id_, name);

  if (location != -1) {
    uniforms_[name] = location;
    return true;
  }

  return false;
}

void GLProgram::Delete()
{
  glDeleteProgram(program_id_);
  program_id_ = 0;
}