#version 400

uniform mat4 MVP;

uniform uvec3 dim;
uniform uvec3 sam;

uniform float mag_threshold;
uniform float mag_scale;

uniform sampler3D flow_sampler;

layout(location = 0) in vec4 vertex;

out vec4 color;

void main () {
  uint x = gl_InstanceID % sam.x;
  uint y = (gl_InstanceID / sam.x) % sam.y;
  uint z = gl_InstanceID / (sam.x * sam.y);

  vec3 d = dim / vec3(sam + 1);
  
  vec3 position = d * vec3(x, y, z) - dim / 2.f + d;
  vec3 coord = (d * vec3(x, y, z) + d) / dim;

  vec4 flow = texture(flow_sampler, coord);
  
  if (vertex.w == 0.f || length(flow) < mag_threshold) {
    gl_Position = MVP * vec4(vertex.xyz + position, 1.f);
  } else {
    gl_Position = MVP * vec4(flow.xyz * mag_scale + position, 1.f);
  };
  
  color = vec4(vertex.w, 1.f - vertex.w, 0.f, 0.5f);
}