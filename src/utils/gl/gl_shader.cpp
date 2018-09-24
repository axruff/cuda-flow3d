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

#include "src/utils/gl/gl_shader.h"

#include <cstdio>

bool GLShader::LoadFromFileAndCompile(const char* filename, GLenum shader_type)
{
  std::FILE* file = std::fopen(filename, "rb");
  if (!file) {
    std::printf("Error. Cannot open the shader file: '%s'\n", filename);
    return false;
  }

  std::fseek(file, 0, SEEK_END);
  long int size = std::ftell(file);
  std::rewind(file);

  if (size == 0) {
    std::printf("Error. The shader source file is empty: '%s'\n", filename);
    std::fclose(file);
    return false;
  }

  char* source = new char[size + 1];

  long int readed = std::fread(source, sizeof(char), size, file);
  source[size] = '\0';

  std::fclose(file);

  if (readed != size) {
    std::printf("Error. Reading the shader source file failed: '%s'\n", filename);
    delete[] source;
    return false;
  }

  shader_id_ = glCreateShader(shader_type);

  if (shader_id_ == 0) {
    std::printf("Error. Shader creation failed: '%s'\n", filename);
    delete[] source;
    return false;
  }

  glShaderSource(shader_id_, 1, &source, NULL);
  delete[] source;

  glCompileShader(shader_id_);

  GLint compile_status;
  glGetShaderiv(shader_id_, GL_COMPILE_STATUS, &compile_status);

  if (compile_status == GL_FALSE) {
    GLchar info_log[512];
    glGetShaderInfoLog(shader_id_, 512, NULL, info_log);
    std::printf("Error. Shader compilation failed: '%s'\n", filename);
    std::printf("%s\n", info_log);
    return false;
  }

  return true;
}

void GLShader::Delete()
{
  glDeleteShader(shader_id_);
  shader_id_ = 0;
}