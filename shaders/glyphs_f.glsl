#version 400

in vec4 color;

out vec4 frag_color;

void main () {
  if (color.w == 0.f) {
    discard;
  }
  frag_color = color;
}