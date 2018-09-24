#version 400
in vec2 tex_coords;
out vec4 frag_color;

uniform sampler1D colormap_sampler;

void main () {
  frag_color = texture(colormap_sampler, tex_coords.x);
}