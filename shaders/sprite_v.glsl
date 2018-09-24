#version 400

uniform mat4 MVP;

layout(location = 0) in vec4 vertex;

out vec2 tex_coords;

void main () {
  tex_coords = vertex.zw;
  gl_Position = MVP * vec4(vertex.xy, 0.f, 1.f);
}