#version 400

uniform mat4 MVP;

layout(location = 0) in vec3 vertex;

out vec4 color;

void main () {
  gl_Position = MVP * vec4 (vertex, 1.f);
  color = vec4(1.5f * normalize(vertex), 1.f);
};