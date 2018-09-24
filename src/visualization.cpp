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

#include "src/visualization.h"

#include <cstdio>
#include <ctime>

#include <iostream>
#include <chrono>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <FreeImage.h>

#include "src/data_types/data3d.h"

bool Visualization::Initialize()
{
  Destroy();

  /* DEBUG: set visualization properties for the thesis */
  {
    //gl_q_rotation_.x = 0.26f;
    //gl_q_rotation_.y = -0.50f;
    //gl_q_rotation_.z = -0.27f;
    //gl_q_rotation_.w = 0.77f;

    //gl_q_rotation_.x = 0.55f;
    //gl_q_rotation_.y = -0.55f;
    //gl_q_rotation_.z = 0.45f;
    //gl_q_rotation_.w = 0.50f;

    //gl_q_rotation_.x = 0.31f;
    //gl_q_rotation_.y = 0.66f;
    //gl_q_rotation_.z = -0.61f;
    //gl_q_rotation_.w = 0.29f;

    //gl_i_mag_dir_ = 1;

    //gl_f_zoom_ = -0.15f;

    //ds_sample_x_ = 30;
    //ds_sample_y_ = 30;
    //ds_sample_z_ = 10;

    //gl_vis_method_ = VM_CONES;

    //gl_b_jittered_ = true;
    //gl_f_r_seed_ = 142.f;
    //gl_f_r_scale_ = 5.f;

    //ds_scale_ = 3.4f;
    //ds_threshold_ = 1.5f;

    //gl_clear_color_ = { .75f, 0.75f, 0.75f, 1.f };

    gl_clear_color_ ={ .0f, .0f, .0f, 1.f };

    //window_width_ = 768;
    //window_height_ = 768;
  }



  initialized_ = InitializeGLWindow();
  if (!initialized_) {
     return initialized_;
  }

  initialized_ = InitializeGLResources();
  if (!initialized_) {
     return initialized_;
  }

  initialized_ = InitializeGLGUI();
  if (!initialized_) {
     return initialized_;
  }
  
  return initialized_;
}

bool Visualization::InitializeGLWindow()
{
   if (!glfwInit()) {
    return false;
  }

#ifdef _DEBUG
  glfwSetErrorCallback(Visualization::GLFWErrorCallback);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_SAMPLES, 16);

  glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

  glfw_thread_window_ = glfwCreateWindow(window_width_, window_height_, "", NULL, NULL);
    if (!glfw_thread_window_) {
    return false;
  }
  
  glfw_window_ = glfwCreateWindow(window_width_, window_height_, "", NULL, glfw_thread_window_);
  if (!glfw_window_) {
    return false;
  }

  /* Set input callback functions */
  glfwSetWindowSizeCallback (glfw_window_, Visualization::GLFWWindowSizeCallback);
  glfwSetCursorPosCallback  (glfw_window_, Visualization::GLFWCursorPositionCallback);
  glfwSetMouseButtonCallback(glfw_window_, Visualization::GLFWMouseButtonCallback);
  glfwSetScrollCallback     (glfw_window_, Visualization::GLFWScrollCallback);
  glfwSetKeyCallback        (glfw_window_, Visualization::GLFWKeyCallback);
  glfwSetCharCallback       (glfw_window_, Visualization::GLFWCharCallback);

  /* Initialize OpenGL context and extension functions */
  glfwMakeContextCurrent(glfw_window_);
  
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    glfwTerminate();
    return false;
  }

  std::printf("OpenGL Vendor: %s\n", glGetString(GL_VENDOR));

#ifdef _DEBUG
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageCallback(OpenGLDebugCallback, NULL);
  glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 0, GL_DEBUG_SEVERITY_NOTIFICATION, -1, "Debug output is enabled.");
#endif

  return true;
}

bool Visualization::InitializeGLResources()
{
  GLuint buffer;
  /* Textures */
  {
    glActiveTexture(GL_TEXTURE0);

    glGenTextures(1, &gl_tex_flow_field_);
    glBindTexture(GL_TEXTURE_3D, gl_tex_flow_field_);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_3D, 0);

    size_t jet_len = 300;
    unsigned char jet[300] = {
      0x00, 0x00, 0x83, 0x00, 0x00, 0x8E, 0x00, 0x00, 0x9A, 0x00, 0x00,
      0xA5, 0x00, 0x00, 0xB1, 0x00, 0x00, 0xBD, 0x00, 0x00, 0xC8, 0x00,
      0x00, 0xD4, 0x00, 0x00, 0xE0, 0x00, 0x00, 0xEB, 0x00, 0x00, 0xF7,
      0x00, 0x00, 0xFE, 0x00, 0x00, 0xFF, 0x00, 0x08, 0xFF, 0x00, 0x12,
      0xFF, 0x00, 0x1D, 0xFF, 0x00, 0x26, 0xFF, 0x00, 0x31, 0xFF, 0x00,
      0x3C, 0xFF, 0x00, 0x46, 0xFF, 0x00, 0x50, 0xFF, 0x00, 0x5A, 0xFF,
      0x00, 0x64, 0xFF, 0x00, 0x6E, 0xFF, 0x00, 0x79, 0xFF, 0x00, 0x83,
      0xFF, 0x00, 0x8E, 0xFF, 0x00, 0x97, 0xFF, 0x00, 0xA2, 0xFF, 0x00,
      0xAC, 0xFF, 0x00, 0xB6, 0xFF, 0x00, 0xC1, 0xFF, 0x00, 0xCB, 0xFF,
      0x00, 0xD5, 0xFF, 0x00, 0xDF, 0xFB, 0x03, 0xE9, 0xF3, 0x0B, 0xF4,
      0xEB, 0x13, 0xFD, 0xE2, 0x1C, 0xFF, 0xDA, 0x24, 0xFF, 0xD1, 0x2D,
      0xFF, 0xC9, 0x35, 0xFF, 0xC1, 0x3D, 0xFF, 0xB9, 0x45, 0xFF, 0xB1,
      0x4D, 0xFF, 0xA8, 0x56, 0xFF, 0xA0, 0x5E, 0xFF, 0x98, 0x66, 0xFF,
      0x8F, 0x6E, 0xFF, 0x87, 0x77, 0xFF, 0x7F, 0x7F, 0xFF, 0x77, 0x87,
      0xFF, 0x6E, 0x8F, 0xFF, 0x66, 0x98, 0xFF, 0x5E, 0xA0, 0xFF, 0x56,
      0xA8, 0xFF, 0x4D, 0xB1, 0xFF, 0x45, 0xB9, 0xFF, 0x3D, 0xC1, 0xFF,
      0x35, 0xC9, 0xFF, 0x2D, 0xD1, 0xFF, 0x24, 0xDA, 0xFF, 0x1C, 0xE2,
      0xFF, 0x13, 0xEB, 0xFF, 0x0B, 0xF3, 0xFA, 0x03, 0xFB, 0xF0, 0x00,
      0xFF, 0xE6, 0x00, 0xFF, 0xDC, 0x00, 0xFF, 0xD4, 0x00, 0xFF, 0xCA,
      0x00, 0xFF, 0xC1, 0x00, 0xFF, 0xB7, 0x00, 0xFF, 0xAD, 0x00, 0xFF,
      0xA4, 0x00, 0xFF, 0x9A, 0x00, 0xFF, 0x92, 0x00, 0xFF, 0x87, 0x00,
      0xFF, 0x7E, 0x00, 0xFF, 0x75, 0x00, 0xFF, 0x6B, 0x00, 0xFF, 0x61,
      0x00, 0xFF, 0x58, 0x00, 0xFF, 0x4E, 0x00, 0xFF, 0x44, 0x00, 0xFF,
      0x3C, 0x00, 0xFF, 0x32, 0x00, 0xFF, 0x29, 0x00, 0xFF, 0x1F, 0x00,
      0xFE, 0x16, 0x00, 0xF7, 0x0C, 0x00, 0xEB, 0x03, 0x00, 0xE0, 0x00,
      0x00, 0xD4, 0x00, 0x00, 0xC8, 0x00, 0x00, 0xBD, 0x00, 0x00, 0xB1,
      0x00, 0x00, 0xA5, 0x00, 0x00, 0x9A, 0x00, 0x00, 0x8E, 0x00, 0x00,
      0x82, 0x00, 0x00
    };

    glGenTextures(1, &gl_tex_colormap_);
    glBindTexture(GL_TEXTURE_1D, gl_tex_colormap_);

    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, 100, 0, GL_RGB, GL_UNSIGNED_BYTE, jet);
    
    glBindTexture(GL_TEXTURE_1D, 0);
  }

  /* Bounding box */
  {
    float bbox_vertices[] = {
      -0.5f, -0.5f,  0.5f,
       0.5f, -0.5f,  0.5f,
       0.5f,  0.5f,  0.5f,
      -0.5f,  0.5f,  0.5f,
      -0.5f,  0.5f, -0.5f,
      -0.5f, -0.5f, -0.5f,
       0.5f, -0.5f, -0.5f,
       0.5f,  0.5f, -0.5f,
    };

    glGenVertexArrays(1, &gl_vao_bbox_);
    glBindVertexArray(gl_vao_bbox_);

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 8 * 3 * sizeof(float), bbox_vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    unsigned char bbox_indices[] = {
      0, 1,
      1, 2,
      2, 3,
      3, 0,
      3, 4,
      4, 5,
      5, 0,
      5, 6,
      6, 7,
      7, 4,
      1, 6,
      2, 7
    };

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 12 * 2 * sizeof(unsigned char), bbox_indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    gl_prog_bbox_.CreateAndLinkProgramFromSourceFiles("./shaders/bbox_v.glsl", "./shaders/bbox_f.glsl");
    gl_prog_bbox_.FindUniformLocation("MVP");
  }

  /* Lines */
  {
    float line_vertices[] = {
      0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 1.f
    };

    glGenVertexArrays(1, &gl_vao_lines_);
    glBindVertexArray(gl_vao_lines_);

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 2 * 4 * sizeof(float), line_vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    gl_prog_lines_.CreateAndLinkProgramFromSourceFiles("./shaders/lines_v.glsl", "./shaders/lines_f.glsl");
    gl_prog_lines_.FindUniformLocation("MVP");
    gl_prog_lines_.FindUniformLocation("dim");
    gl_prog_lines_.FindUniformLocation("sam");
    gl_prog_lines_.FindUniformLocation("mag_threshold");
    gl_prog_lines_.FindUniformLocation("mag_scale");
    gl_prog_lines_.FindUniformLocation("flow_sampler");

    glProgramUniform1i(gl_prog_lines_.GetProgramId(), gl_prog_lines_.GetUniformLocation("flow_sampler"), 0);

    gl_prog_r_lines_.CreateAndLinkProgramFromSourceFiles("./shaders/lines_r_v.glsl", "./shaders/lines_f.glsl");
    gl_prog_r_lines_.FindUniformLocation("MVP");
    gl_prog_r_lines_.FindUniformLocation("dim");
    gl_prog_r_lines_.FindUniformLocation("sam");
    gl_prog_r_lines_.FindUniformLocation("mag_threshold");
    gl_prog_r_lines_.FindUniformLocation("mag_scale");
    gl_prog_r_lines_.FindUniformLocation("flow_sampler");

    gl_prog_r_lines_.FindUniformLocation("r_scale");
    gl_prog_r_lines_.FindUniformLocation("r_seed");


    glProgramUniform1i(gl_prog_r_lines_.GetProgramId(), gl_prog_r_lines_.GetUniformLocation("flow_sampler"), 0);
  }

  /* Glyphs */
  {
    float glyph_height = 1.f;
    float glyph_base2 = 0.2f;

    glm::vec3 glyph_data[18] = {
      glm::vec3(         0.f, glyph_height,          0.f),
      glm::vec3(-glyph_base2,          0.f,  glyph_base2),
      glm::vec3( glyph_base2,          0.f,  glyph_base2),
      glm::vec3(         0.f, glyph_height,          0.f),
      glm::vec3(-glyph_base2,          0.f, -glyph_base2),
      glm::vec3(-glyph_base2,          0.f,  glyph_base2),
      glm::vec3(         0.f, glyph_height,          0.f),
      glm::vec3( glyph_base2,          0.f, -glyph_base2),
      glm::vec3(-glyph_base2,          0.f, -glyph_base2),
      glm::vec3(         0.f, glyph_height,          0.f),
      glm::vec3( glyph_base2,          0.f,  glyph_base2),
      glm::vec3( glyph_base2,          0.f, -glyph_base2),
      /* Base */
      glm::vec3( glyph_base2, 0.f, -glyph_base2),
      glm::vec3( glyph_base2, 0.f,  glyph_base2),
      glm::vec3(-glyph_base2, 0.f,  glyph_base2),
      glm::vec3(-glyph_base2, 0.f,  glyph_base2),
      glm::vec3(-glyph_base2, 0.f, -glyph_base2),
      glm::vec3( glyph_base2, 0.f, -glyph_base2),
    };

    glm::vec3 glyph_normals[18];
    for (size_t i = 0; i < 6; i++) {
      glyph_normals[i * 3 + 0] = glm::normalize(glm::cross(glyph_data[i * 3] - glyph_data[i * 3 + 1], glyph_data[i * 3] - glyph_data[i * 3 + 2]));
      glyph_normals[i * 3 + 1] = glm::normalize(glm::cross(glyph_data[i * 3] - glyph_data[i * 3 + 1], glyph_data[i * 3] - glyph_data[i * 3 + 2]));
      glyph_normals[i * 3 + 2] = glm::normalize(glm::cross(glyph_data[i * 3] - glyph_data[i * 3 + 1], glyph_data[i * 3] - glyph_data[i * 3 + 2]));
    }

    glGenVertexArrays(1, &gl_vao_glyphs_);
    glBindVertexArray(gl_vao_glyphs_);
    /* Vertices */
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 18 * 3 * sizeof(float), glyph_data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    /* Normals */
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 18 * 3 * sizeof(float), glyph_normals, GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    gl_prog_glyphs_.CreateAndLinkProgramFromSourceFiles("./shaders/glyphs_v.glsl", "./shaders/glyphs_f.glsl");
    gl_prog_glyphs_.FindUniformLocation("m_model");
    gl_prog_glyphs_.FindUniformLocation("m_view");
    gl_prog_glyphs_.FindUniformLocation("m_projection");
    gl_prog_glyphs_.FindUniformLocation("m_normal");

    gl_prog_glyphs_.FindUniformLocation("dim");
    gl_prog_glyphs_.FindUniformLocation("sam");
    gl_prog_glyphs_.FindUniformLocation("mag_threshold");
    gl_prog_glyphs_.FindUniformLocation("mag_scale");
    gl_prog_glyphs_.FindUniformLocation("mag_max");
    gl_prog_glyphs_.FindUniformLocation("mag_dir");

    gl_prog_glyphs_.FindUniformLocation("flow_sampler");
    gl_prog_glyphs_.FindUniformLocation("colormap_sampler");

    glProgramUniform1i(gl_prog_glyphs_.GetProgramId(), gl_prog_glyphs_.GetUniformLocation("flow_sampler"), 0);
    glProgramUniform1i(gl_prog_glyphs_.GetProgramId(), gl_prog_glyphs_.GetUniformLocation("colormap_sampler"), 1);

    gl_prog_r_glyphs_.CreateAndLinkProgramFromSourceFiles("./shaders/glyphs_r_v.glsl", "./shaders/glyphs_f.glsl");
    gl_prog_r_glyphs_.FindUniformLocation("m_model");
    gl_prog_r_glyphs_.FindUniformLocation("m_view");
    gl_prog_r_glyphs_.FindUniformLocation("m_projection");
    gl_prog_r_glyphs_.FindUniformLocation("m_normal");

    gl_prog_r_glyphs_.FindUniformLocation("dim");
    gl_prog_r_glyphs_.FindUniformLocation("sam");
    gl_prog_r_glyphs_.FindUniformLocation("mag_threshold");
    gl_prog_r_glyphs_.FindUniformLocation("mag_scale");
    gl_prog_r_glyphs_.FindUniformLocation("mag_max");
    gl_prog_r_glyphs_.FindUniformLocation("mag_dir");

    gl_prog_r_glyphs_.FindUniformLocation("flow_sampler");
    gl_prog_r_glyphs_.FindUniformLocation("colormap_sampler");

    gl_prog_r_glyphs_.FindUniformLocation("r_scale");
    gl_prog_r_glyphs_.FindUniformLocation("r_seed");

    glProgramUniform1i(gl_prog_r_glyphs_.GetProgramId(), gl_prog_r_glyphs_.GetUniformLocation("flow_sampler"), 0);
    glProgramUniform1i(gl_prog_r_glyphs_.GetProgramId(), gl_prog_r_glyphs_.GetUniformLocation("colormap_sampler"), 1);

  }

  /* Cones */
  {
    glGenVertexArrays(1, &gl_vao_cones_); 
    GenerateCone(16, gl_vao_cones_, &gl_i_cone_indices_);
  }

  /* Sprite */
  {
    float sprite[4 * 4] = {
      0.f, 0.f, 0.f, 0.f,
      1.f, 0.f, 1.f, 0.f,
      0.f, 1.f, 0.f, 1.f,
      1.f, 1.f, 1.f, 1.f
    };

    unsigned char sprite_indices[] = {
      0, 2, 1,
      2, 3, 1
    };

    glGenVertexArrays(1, &gl_vao_sprite_);
    glBindVertexArray(gl_vao_sprite_);

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 4 * 4 * sizeof(float), sprite, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * 2 * sizeof(unsigned char), sprite_indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    gl_prog_sprite_.CreateAndLinkProgramFromSourceFiles("./shaders/sprite_v.glsl", "./shaders/sprite_f.glsl");
    gl_prog_sprite_.FindUniformLocation("MVP");
    gl_prog_sprite_.FindUniformLocation("colormap_sampler");

    glProgramUniform1i(gl_prog_sprite_.GetProgramId(), gl_prog_sprite_.GetUniformLocation("colormap_sampler"), 1);
  }
  /* Setup matrices */
  gl_m_view_= glm::lookAt(glm::vec3(0.f, 0.f, 3.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));

  return true;
}

void Visualization::GenerateCone(size_t face_count, GLuint gl_vao_cones, int* indices_count)
{
  if (face_count * 3 > std::numeric_limits<unsigned char>::max()) {
    printf("Error. Cannot generate cone with more than %d faces.\n", std::numeric_limits<unsigned char>::max() / 3);
    std::terminate();
  }

  float cone_height = 1.f;
  float cone_base2 = 0.2f;

  float angle = 2.f * glm::pi<float>() / static_cast<float>(face_count);

  int vertices = 3 * face_count;

  /* Indexed cone */
  glm::vec3* cone_vertices_ptr = new glm::vec3[vertices];
  
  for (size_t i = 0; i < face_count; i++) {
    /* Upper vertex */
    cone_vertices_ptr[i + 0 * face_count] = glm::vec3(0.f, cone_height, 0.f);
    /* Bottom vertex */
    cone_vertices_ptr[i + 1 * face_count] = glm::vec3(cone_base2 * glm::cos(angle * (i)), 0.f, cone_base2 * glm::sin(angle * (i)));
    /* Base vertex, the same as a bottom vertex*/
    cone_vertices_ptr[i + 2 * face_count] = cone_vertices_ptr[i + face_count];
  }

  *indices_count = 3 * (2 * face_count - 2);
  unsigned char* cone_indices_ptr = new unsigned char[*indices_count];
  /* Cone */
  for (size_t i = 0; i < face_count; i++) {
    cone_indices_ptr[3 * i + 0] = i;
    cone_indices_ptr[3 * i + 1] = face_count + (i + 1) % face_count;
    cone_indices_ptr[3 * i + 2] = face_count + (i + 0);
  }
  /* Base */
  for (size_t i = 0; i < face_count - 2; i++) {
    cone_indices_ptr[3 * (face_count + i) + 0] = 2 * face_count + 0;
    cone_indices_ptr[3 * (face_count + i) + 1] = 2 * face_count + i + 1;
    cone_indices_ptr[3 * (face_count + i) + 2] = 2 * face_count + i + 2;
  }

  glm::vec3* cone_normals_ptr = new glm::vec3[3 * face_count];

  for (size_t i = 0; i < face_count; i++) {
    glm::vec3& v0  = cone_vertices_ptr[cone_indices_ptr[3 * i + 0]];
    glm::vec3& v1  = cone_vertices_ptr[cone_indices_ptr[3 * i + 1]];
    glm::vec3& v2  = cone_vertices_ptr[cone_indices_ptr[3 * i + 2]];

    /* Upper vertex */
    cone_normals_ptr[i] = glm::normalize(glm::cross(v0 - v1, v0 - v2));
    /* Bottom vertex */
    cone_normals_ptr[i + face_count] = glm::vec3(cone_base2 * glm::cos(angle * (i)), 0.f, cone_base2 * glm::sin(angle * (i)));
    /* Base vertex, the same as a bottom vertex*/
    cone_normals_ptr[i + 2 * face_count] = glm::vec3(0.f, -1.f, 0.f);
  }

  GLuint buffer;
  glBindVertexArray(gl_vao_cones);
  /* Vertices */
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, vertices * 3 * sizeof(float), cone_vertices_ptr, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  /* Normals */
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, vertices * 3 * sizeof(float), cone_normals_ptr, GL_STATIC_DRAW);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  /* Indices */
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, *indices_count * sizeof(unsigned char), cone_indices_ptr, GL_STATIC_DRAW);

  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] cone_normals_ptr;
  delete[] cone_indices_ptr;
  delete[] cone_vertices_ptr;
}

bool Visualization::InitializeGLGUI()
{
  TwInit(TW_OPENGL_CORE, NULL);
  
  ant_bar_ = TwNewBar("Options");
  TwDefine("'Options' size='250 500'");
  //TwDefine("'Options' iconified=true");

  TwAddSeparator(ant_bar_, "Dataset info", NULL);

  TwAddVarRO(ant_bar_, "Width", TW_TYPE_INT32, &ds_width_, NULL);
  TwAddVarRO(ant_bar_, "Height", TW_TYPE_INT32, &ds_height_, NULL);
  TwAddVarRO(ant_bar_, "Depth", TW_TYPE_INT32, &ds_depth_, NULL);
  TwAddVarRO(ant_bar_, "Max", TW_TYPE_FLOAT, &ds_max_, "precision=4");
  TwAddVarRO(ant_bar_, "Avg", TW_TYPE_FLOAT, &ds_avg_, "precision=4");
  TwAddVarRO(ant_bar_, "Min", TW_TYPE_FLOAT, &ds_min_, "precision=4");

  TwAddSeparator(ant_bar_, "Orientation", NULL);

  TwAddVarRW(ant_bar_, "Rotation", TW_TYPE_QUAT4F, glm::value_ptr(gl_q_rotation_), NULL);

  TwAddSeparator(ant_bar_, "Sampling", NULL);

  TwAddVarRW(ant_bar_, "X Samples", TW_TYPE_INT32, &ds_sample_x_, "min=1");
  TwAddVarRW(ant_bar_, "Y Samples", TW_TYPE_INT32, &ds_sample_y_, "min=1");
  TwAddVarRW(ant_bar_, "Z Samples", TW_TYPE_INT32, &ds_sample_z_, "min=1");
  TwAddVarRW(ant_bar_, "Show grid", TW_TYPE_BOOLCPP, &gl_b_show_grid_, NULL);
  TwAddSeparator(ant_bar_, "Rendering", NULL);
  TwAddVarRW(ant_bar_, "Threshold", TW_TYPE_FLOAT, &ds_threshold_, " min=0 step=0.01");
  TwAddVarRW(ant_bar_, "Scale", TW_TYPE_FLOAT, &ds_scale_, " min=0.1 step=0.05");
  {
    TwEnumVal vis_method_ev[3] = { { VM_LINES, "Lines" }, { VM_GLYPHS, "Glyphs" }, { VM_CONES, "Cones" } };
    TwType vis_method_type = TwDefineEnum("VisType", vis_method_ev, 3);
    TwAddVarRW(ant_bar_, "Visualisation", vis_method_type, &gl_vis_method_, "help='Change visualisation method.'");
  }
  TwAddVarRW(ant_bar_, "Magnitute / Direction", TW_TYPE_BOOL32, &gl_i_mag_dir_, NULL);
  
  TwAddSeparator(ant_bar_, "Computation", NULL);
  TwAddVarRW(ant_bar_, "Next step", TW_TYPE_BOOLCPP, &do_next_step_, NULL);

  TwAddSeparator(ant_bar_, "Random", NULL);
  TwAddVarRW(ant_bar_, "Jittered", TW_TYPE_BOOLCPP, &gl_b_jittered_, NULL);
  TwAddVarRW(ant_bar_, "Random Seed", TW_TYPE_FLOAT, &gl_f_r_seed_, " min=0.1 step=0.05");
  TwAddVarRW(ant_bar_, "Random Scale", TW_TYPE_FLOAT, &gl_f_r_scale_, " min=0.0 step=0.05");

  return true;
}

void Visualization::Render(GLFWwindow* window)
{
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);

  //glClearColor(0.6f, 0.6f, 0.6f, 1.f);
  glClearColor(gl_clear_color_.r, gl_clear_color_.g, gl_clear_color_.b, gl_clear_color_.a);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  /* Bind textures */
  glActiveTexture(GL_TEXTURE0 + 0);
  glBindTexture(GL_TEXTURE_3D, gl_tex_flow_field_);
  glBindSampler(0, 0);

  glActiveTexture(GL_TEXTURE0 + 1);
  glBindTexture(GL_TEXTURE_1D, gl_tex_colormap_);
  glBindSampler(1, 0);

  /* Scale data set to fit into the screen */
  float scale = 0.01f;

  glm::mat4 l_model = glm::scale(glm::mat4(), glm::vec3(scale)) * glm::mat4_cast(gl_q_rotation_);
  glm::mat4 l_view = glm::translate(gl_m_view_, glm::vec3(0.f, 0.f, gl_f_zoom_));
  
  glm::mat4 MVP = gl_m_projection_p_ * l_view * l_model;

  /* Draw flow field */
  switch (gl_vis_method_) {
  case VM_LINES: {
    if (gl_b_jittered_) {
      gl_prog_r_lines_.Use();
      glUniformMatrix4fv(gl_prog_r_lines_.GetUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
      glUniform3uiv(gl_prog_r_lines_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));
      glUniform1f(gl_prog_r_lines_.GetUniformLocation("mag_threshold"), ds_threshold_);
      glUniform1f(gl_prog_r_lines_.GetUniformLocation("mag_scale"), ds_scale_);

      glUniform1f(gl_prog_r_lines_.GetUniformLocation("r_scale"), gl_f_r_scale_);
      glUniform1f(gl_prog_r_lines_.GetUniformLocation("r_seed"), gl_f_r_seed_);
    } else {
      gl_prog_lines_.Use();
      glUniformMatrix4fv(gl_prog_lines_.GetUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
      glUniform3uiv(gl_prog_lines_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));
      glUniform1f(gl_prog_lines_.GetUniformLocation("mag_threshold"), ds_threshold_);
      glUniform1f(gl_prog_lines_.GetUniformLocation("mag_scale"), ds_scale_);
    }
    glBindVertexArray(gl_vao_lines_);
    glDrawArraysInstanced(GL_LINES, 0, 2, ds_sample_x_ * ds_sample_y_ * ds_sample_z_);
    break;
  }
  case VM_GLYPHS: {
    glm::mat4 l_normal = glm::transpose(glm::inverse(l_view * l_model));

    if (gl_b_jittered_) {
      gl_prog_r_glyphs_.Use();
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_model"), 1, GL_FALSE, glm::value_ptr(l_model));
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_view"), 1, GL_FALSE, glm::value_ptr(l_view));
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_projection"), 1, GL_FALSE, glm::value_ptr(gl_m_projection_p_));
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_normal"), 1, GL_FALSE, glm::value_ptr(l_normal));

      glUniform3uiv(gl_prog_r_glyphs_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));
      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("mag_threshold"), ds_threshold_);
      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("mag_scale"), ds_scale_);
      glUniform1i(gl_prog_r_glyphs_.GetUniformLocation("mag_dir"), gl_i_mag_dir_);

      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("r_scale"), gl_f_r_scale_);
      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("r_seed"), gl_f_r_seed_);
    } else {
      gl_prog_glyphs_.Use();
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_model"), 1, GL_FALSE, glm::value_ptr(l_model));
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_view"), 1, GL_FALSE, glm::value_ptr(l_view));
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_projection"), 1, GL_FALSE, glm::value_ptr(gl_m_projection_p_));
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_normal"), 1, GL_FALSE, glm::value_ptr(l_normal));

      glUniform3uiv(gl_prog_glyphs_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));
      glUniform1f(gl_prog_glyphs_.GetUniformLocation("mag_threshold"), ds_threshold_);
      glUniform1f(gl_prog_glyphs_.GetUniformLocation("mag_scale"), ds_scale_);
      glUniform1i(gl_prog_glyphs_.GetUniformLocation("mag_dir"), gl_i_mag_dir_);
    }
    glBindVertexArray(gl_vao_glyphs_);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 18, ds_sample_x_ * ds_sample_y_ * ds_sample_z_);
    break;
  }
  case VM_CONES: {
    glm::mat4 l_normal = glm::transpose(glm::inverse(l_view * l_model));

    if (gl_b_jittered_) {
      gl_prog_r_glyphs_.Use();
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_model"), 1, GL_FALSE, glm::value_ptr(l_model));
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_view"), 1, GL_FALSE, glm::value_ptr(l_view));
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_projection"), 1, GL_FALSE, glm::value_ptr(gl_m_projection_p_));
      glUniformMatrix4fv(gl_prog_r_glyphs_.GetUniformLocation("m_normal"), 1, GL_FALSE, glm::value_ptr(l_normal));

      glUniform3uiv(gl_prog_r_glyphs_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));
      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("mag_threshold"), ds_threshold_);
      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("mag_scale"), ds_scale_);
      glUniform1i(gl_prog_r_glyphs_.GetUniformLocation("mag_dir"), gl_i_mag_dir_);

      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("r_scale"), gl_f_r_scale_);
      glUniform1f(gl_prog_r_glyphs_.GetUniformLocation("r_seed"), gl_f_r_seed_);
    } else {
      gl_prog_glyphs_.Use();
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_model"), 1, GL_FALSE, glm::value_ptr(l_model));
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_view"), 1, GL_FALSE, glm::value_ptr(l_view));
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_projection"), 1, GL_FALSE, glm::value_ptr(gl_m_projection_p_));
      glUniformMatrix4fv(gl_prog_glyphs_.GetUniformLocation("m_normal"), 1, GL_FALSE, glm::value_ptr(l_normal));

      glUniform3uiv(gl_prog_glyphs_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));
      glUniform1f(gl_prog_glyphs_.GetUniformLocation("mag_threshold"), ds_threshold_);
      glUniform1f(gl_prog_glyphs_.GetUniformLocation("mag_scale"), ds_scale_);
      glUniform1i(gl_prog_glyphs_.GetUniformLocation("mag_dir"), gl_i_mag_dir_);
    }

    glBindVertexArray(gl_vao_cones_);
    glDrawElementsInstanced(GL_TRIANGLES, gl_i_cone_indices_, GL_UNSIGNED_BYTE, 0, ds_sample_x_ * ds_sample_y_ * ds_sample_z_);
    break;
  }

  default:
    break;
  }

  /* Draw sampling grid */
  if (gl_b_show_grid_) {
    if (gl_b_jittered_) {
      gl_prog_r_lines_.Use();
      glUniformMatrix4fv(gl_prog_r_lines_.GetUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
      glUniform3uiv(gl_prog_r_lines_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));

      glUniform1f(gl_prog_r_lines_.GetUniformLocation("r_scale"), gl_f_r_scale_);
      glUniform1f(gl_prog_r_lines_.GetUniformLocation("r_seed"), gl_f_r_seed_);
    } else {
      gl_prog_lines_.Use();
      glUniformMatrix4fv(gl_prog_lines_.GetUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
      glUniform3uiv(gl_prog_lines_.GetUniformLocation("sam"), 1, glm::value_ptr(glm::uvec3(ds_sample_x_, ds_sample_y_, ds_sample_z_)));
    }

    glPointSize(1.5f);
    glBindVertexArray(gl_vao_lines_);
    glDrawArraysInstanced(GL_POINTS, 0, 1, ds_sample_x_ * ds_sample_y_ * ds_sample_z_);
  }

  /* Draw the bounding box */
  MVP = gl_m_projection_p_ * l_view * glm::mat4_cast(gl_q_rotation_) * glm::scale(glm::mat4(), scale * glm::vec3(ds_width_, ds_height_, ds_depth_));

  gl_prog_bbox_.Use();
  glUniformMatrix4fv(gl_prog_bbox_.GetUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));

  glBindVertexArray(gl_vao_bbox_);
  glDrawElements(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0);


  /* Draw the color map */
  if (gl_i_mag_dir_ == 0 && (gl_vis_method_== VM_GLYPHS || gl_vis_method_ == VM_CONES)) {
    glm::mat4 s_model;

    s_model = glm::translate(s_model, glm::vec3(window_width_ / 4.f, window_height_ - 20.f, 0.f));
    s_model = glm::scale(s_model, glm::vec3(window_width_ / 2.f, 15.f, 0.f));

    MVP = gl_m_projection_o_ * s_model;

    gl_prog_sprite_.Use();
    glUniformMatrix4fv(gl_prog_sprite_.GetUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));

    glBindVertexArray(gl_vao_sprite_);

    glDisable(GL_DEPTH_TEST);
    glDrawElements(GL_TRIANGLES, 3 * 2, GL_UNSIGNED_BYTE, 0);
    glEnable(GL_DEPTH_TEST);
  }
}

void Visualization::Show()
{
  if (!initialized_) {
    return;
  }

  glfwShowWindow(glfw_window_);

  /* Main loop */
  while (!glfwWindowShouldClose(glfw_window_)) {
    UpdateFPSCounter(glfw_window_);
    Render(glfw_window_);

    TwDraw();

    glfwSwapBuffers(glfw_window_);
    glfwPollEvents();
  }
}

void Visualization::UpdateFPSCounter(GLFWwindow* window)
{
  static double previous_seconds = glfwGetTime();
  static int frame_count;

  double current_seconds = glfwGetTime();
  double elapsed_seconds = current_seconds - previous_seconds;
  
  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    double fps = static_cast<double>(frame_count) / elapsed_seconds;
    char title[128];
    sprintf (title, "3D Flow Field Visualization @ FPS: %.2f", fps);
    glfwSetWindowTitle (window, title);
    frame_count = 0;
  }
  frame_count++;
}

void Visualization::Destroy()
{
  if (initialized_) {
    TwTerminate();
    glfwTerminate();
  }
}

void Visualization::UpdateDatasetSize(size_t width, size_t height, size_t depth)
{
      ds_width_ = width;
      ds_height_ = height;
      ds_depth_ = depth;

      glProgramUniform3uiv(gl_prog_lines_.GetProgramId(), gl_prog_lines_.GetUniformLocation("dim") , 1, glm::value_ptr(glm::uvec3(width, height, depth)));

      glProgramUniform3uiv(gl_prog_r_lines_.GetProgramId(), gl_prog_r_lines_.GetUniformLocation("dim") , 1, glm::value_ptr(glm::uvec3(width, height, depth)));

      glProgramUniform3uiv(gl_prog_glyphs_.GetProgramId(), gl_prog_glyphs_.GetUniformLocation("dim") , 1, glm::value_ptr(glm::uvec3(width, height, depth)));
      glProgramUniform1f(  gl_prog_glyphs_.GetProgramId(), gl_prog_glyphs_.GetUniformLocation("mag_max"), ds_max_);

      glProgramUniform3uiv(gl_prog_r_glyphs_.GetProgramId(), gl_prog_r_glyphs_.GetUniformLocation("dim") , 1, glm::value_ptr(glm::uvec3(width, height, depth)));
      glProgramUniform1f(  gl_prog_r_glyphs_.GetProgramId(), gl_prog_r_glyphs_.GetUniformLocation("mag_max"), ds_max_);


      TwSetParam(ant_bar_, "X Samples", "max", TW_PARAM_INT32, 1, &width);
      TwSetParam(ant_bar_, "Y Samples", "max", TW_PARAM_INT32, 1, &height);
      TwSetParam(ant_bar_, "Z Samples", "max", TW_PARAM_INT32, 1, &depth);
}

/* Texture loading */

bool Visualization::Load3DFlowTextureFromRAWFileUVW(const char* filename_u, const char* filename_v, const char* filename_w, size_t width, size_t height, size_t depth)
{
  has_texture_ = false;
  if (initialized_) {
    /* When this function is called from another thread use OpenGL context sharing */
    if (glfwGetCurrentContext() != glfw_window_) {
      glfwMakeContextCurrent(glfw_thread_window_);
    }

    Data3D flow_u(width, height, depth);
    Data3D flow_v(width, height, depth);
    Data3D flow_w(width, height, depth);

    std::printf("Loading a flow field from files\n");

    has_texture_  = flow_u.ReadRAWFromFileF32(filename_u, width, height, depth);
    has_texture_ &= flow_v.ReadRAWFromFileF32(filename_v, width, height, depth);
    has_texture_ &= flow_w.ReadRAWFromFileF32(filename_w, width, height, depth);

    if (has_texture_) {
      glm::vec3 *data = new glm::vec3[width * height * depth];


      ds_min_ = std::numeric_limits<float>::max();
      ds_max_ = std::numeric_limits<float>::min();
      ds_avg_ = 0.f;

      for (size_t z = 0; z < depth; ++z) {
        for (size_t y = 0; y < height; ++y) {
          for (size_t x = 0; x < width; ++x) {
            glm::vec3 vector(flow_u.Data(x, y, z), flow_v.Data(x, y, z), flow_w.Data(x, y, z));

            data[width * height * z + width * y + x] = vector;

            float magnitude = glm::length(vector);

            ds_min_ = std::fmin(ds_min_, magnitude);
            ds_max_ = std::fmax(ds_max_, magnitude);
            ds_avg_ += magnitude;
          }
        }
      }
      ds_avg_ /= static_cast<float>(width * height * depth);

      /* Update private variables and shader uniforms */
      UpdateDatasetSize(width, height, depth);

      /* Load the texture into OpenGL memory */
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, gl_tex_flow_field_);
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, data);
      glBindTexture(GL_TEXTURE_3D, 0);

      has_texture_ = true;

      delete[] data;
    }
  }
  return has_texture_;
}

bool Visualization::Load3DFlowTextureFromRAWFileWHD(const char* filename)
{
  has_texture_ = false;
  if (initialized_) {
    /* When this function is called from another thread use OpenGL context sharing */
    if (glfwGetCurrentContext() != glfw_window_) {
      glfwMakeContextCurrent(glfw_thread_window_);
    }

    std::FILE *file = std::fopen(filename, "rb");
    if (file) {
      size_t readed = 0;

      glm::vec3 vector;
      readed = std::fread(glm::value_ptr(vector), sizeof(glm::vec3), 1, file);

      size_t width = static_cast<size_t>(vector.x);
      size_t height = static_cast<size_t>(vector.y);
      size_t depth = static_cast<size_t>(vector.z);

      has_texture_ = readed && width && height && depth;

      if (has_texture_) {
        glm::vec3 *data = new glm::vec3[width * height * depth];

        ds_min_ = std::numeric_limits<float>::max();
        ds_max_ = std::numeric_limits<float>::min();
        ds_avg_ = 0.f;

        for (size_t z = 0; z < depth; ++z) {
          for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
              readed = std::fread(glm::value_ptr(vector), sizeof(glm::vec3), 1, file);
              if (readed == 0) {
                std::printf("Error reading a raw file: %s\n", filename);
                has_texture_ = false;
                goto loop_break;
              }

              data[width * height * z + width * y + x] = vector;

              float magnitude = glm::length(vector);

              ds_min_ = std::fmin(ds_min_, magnitude);
              ds_max_ = std::fmax(ds_max_, magnitude);
              ds_avg_ += magnitude;
            }
          }
        }
        ds_avg_ /= static_cast<float>(width * height * depth);

        /* Update private variables and shader uniforms */
        UpdateDatasetSize(width, height, depth);

        /* Load the texture into OpenGL memory */
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, gl_tex_flow_field_);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, data);
        glBindTexture(GL_TEXTURE_3D, 0);

        has_texture_ = true;
loop_break:
        delete[] data;
      }
      std::fclose(file);
    } else {
      std::printf("Cannot find a raw file: %s\n", filename);
    }
  }
  return has_texture_;
}

bool Visualization::Load3DFlowTextureFromData3DUVW(Data3D& flow_u, Data3D& flow_v, Data3D&flow_w)
{
  has_texture_ = false;
  if (initialized_) {
    /* When this function is called from another thread use OpenGL context sharing */
    if (glfwGetCurrentContext() != glfw_window_) {
      glfwMakeContextCurrent(glfw_thread_window_);
    }

    size_t width = flow_u.Width();
    size_t height = flow_u.Height();
    size_t depth = flow_u.Depth();

    glm::vec3 *data = new glm::vec3[width * height * depth];


    ds_min_ = std::numeric_limits<float>::max();
    ds_max_ = std::numeric_limits<float>::min();
    ds_avg_ = 0.f;

    for (size_t z = 0; z < depth; ++z) {
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          glm::vec3 vector(flow_u.Data(x, y, z), flow_v.Data(x, y, z), flow_w.Data(x, y, z));

          data[width * height * z + width * y + x] = vector;

          float magnitude = glm::length(vector);

          ds_min_ = std::fmin(ds_min_, magnitude);
          ds_max_ = std::fmax(ds_max_, magnitude);
          ds_avg_ += magnitude;
        }
      }
    }
    ds_avg_ /= static_cast<float>(width * height * depth);

    /* Update private variables and shader uniforms */
    UpdateDatasetSize(width, height, depth);

    /* Load the texture into OpenGL memory */
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, gl_tex_flow_field_);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_3D, 0);

    has_texture_ = true;

    delete[] data;
  }
  return has_texture_;
}

/* Threading */

void Visualization::ThreadBody()
{
  Initialize();
  {
    std::unique_lock<std::mutex> locker(thr_m_lock_);
    thr_cv_initialized_.notify_all();
  }
  Show();
  Destroy();
}

void Visualization::RunInSeparateThread()
{
  thr_thread_ = std::thread(&Visualization::ThreadBody, this);
}

void Visualization::WaitForInitialization()
{
  std::unique_lock<std::mutex> locker(thr_m_lock_);
  while (!initialized_) {
    thr_cv_initialized_.wait(locker);
  }
}

void Visualization::WaitForFinalization()
{
  thr_thread_.join();
}

void Visualization::DoNextStep()
{
  while (!do_next_step_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void Visualization::SaveToFile()
{
  int width, height;
  glfwGetFramebufferSize(glfw_window_, &width, &height);

  GLubyte* pixels = new GLubyte[3 * width * height];

  glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);

  // Convert to FreeImage format & save to file
  FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);


  std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  char buffer[32];
  std::strftime(buffer, 32, "%H-%M-%S", std::localtime(&t));

  std::string filename = "./data/output/screenshots/screenshot-";
  filename += buffer;
  filename +=".png";

  std::cout << filename << std::endl;

  FreeImage_Save(FIF_PNG, image, filename.c_str(), 0);

  // Free resources
  FreeImage_Unload(image);
  delete[] pixels;
}

/* Callbacks */

#ifdef _DEBUG

void Visualization::GLFWErrorCallback(int error, const char* message)
{
  std::printf("GLFW: %s\n", message);
}

void Visualization::OpenGLDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
  std::printf("OpenGL: %s\n", message);
}

#endif // _DEBUG

void Visualization::GLFWWindowSizeCallback(GLFWwindow* window, int width, int height)
{
  if (width && height) {
    Visualization& visualisation = GetInstance();
    
    visualisation.window_width_ = width;
    visualisation.window_height_ = height;
    
    visualisation.gl_m_projection_p_ = glm::perspective(45.f, width / static_cast<float>(height), 0.1f, 100.f);
    visualisation.gl_m_projection_o_ = glm::ortho(0.f, static_cast<float>(width), static_cast<float>(height), 0.f, -1.f, 1.f);
    
    glViewport(0, 0, width, height);
    TwWindowSize(width, height);
  }
}

void Visualization::GLFWCursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
  TwEventMousePosGLFW(static_cast<int>(xpos), static_cast<int>(ypos));
}

void Visualization::GLFWMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  TwEventMouseButtonGLFW(button, action);
}

void Visualization::GLFWScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  if (!TwEventMouseWheelGLFW(static_cast<int>(yoffset))) {
    Visualization& visualisation = GetInstance();
    const float zoom_step = 0.1f;
    if (yoffset > 0) {
      visualisation.gl_f_zoom_ += zoom_step;
    } else {
      visualisation.gl_f_zoom_ -= zoom_step;
    }
  }
}

void Visualization::GLFWKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  //TwEventKeyGLFW(key, action);

  if (key == GLFW_KEY_S && action == GLFW_PRESS) {
     Visualization& visualisation = GetInstance();
    visualisation.SaveToFile();
  }
}

void Visualization::GLFWCharCallback(GLFWwindow* window, unsigned int codepoint)
{
  //TwEventCharGLFW(codepoint, GLFW_PRESS);
}
