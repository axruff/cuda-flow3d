#version 400

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_projection;
uniform mat4 m_normal;

uniform uvec3 dim;
uniform uvec3 sam;
uniform float mag_threshold;
uniform float mag_scale;
uniform float mag_max;
uniform int mag_dir;

uniform sampler3D flow_sampler;
uniform sampler1D colormap_sampler;

layout(location = 0) in vec4 vertex;
layout(location = 1) in vec3 normal;

out vec4 color;

mat3 get_rotation(in vec3 direction) {
  if (direction.x == 0.f && direction.z == 0.f) {
    if (direction.y < 0) {
      return mat3(-1.f);
    }
    return mat3(1.f);
  } else {
    vec3 new_y = normalize(direction);
    vec3 new_z = normalize(cross(new_y, vec3(0.f, 1.f, 0.f)));
    vec3 new_x = normalize(cross(new_y, new_z));
    
    return mat3(new_x, new_y, new_z);
  }
}

void main () {
  mat4 MVP = m_projection * m_view * m_model;

  uint x = gl_InstanceID % sam.x;
  uint y = (gl_InstanceID / sam.x) % sam.y;
  uint z = gl_InstanceID / (sam.x * sam.y);
  vec3 d = dim / vec3(sam + 1);
  vec3 position = d * vec3(x, y, z) - dim / 2.f + d;
  vec3 coord = (d * vec3(x, y, z) + d) / dim;

  vec4 flow = texture(flow_sampler, coord);
  mat3 rotation = get_rotation(flow.xyz);

  gl_Position = MVP * vec4(rotation * vertex.xyz * mag_scale * length(flow.xyz) + position, 1.f);

  mat3 normal_rotation = transpose(inverse(rotation));

  vec4 position_ms = m_view * m_model * vec4(vertex.xyz, 1.f);
  vec4 normal_ms = m_normal * vec4(normal_rotation * normal, 0.f);
  vec3 light_dir = normalize(position_ms.xyz);

  if (length(flow) >= mag_threshold) {
    if (mag_dir == 0) {
      color = vec4(texture(colormap_sampler, length(flow.xyz) / mag_max).xyz * clamp(dot(-normalize(normal_ms.xyz), light_dir.xyz), 0.f, 1.f), 1.f);
    } else {
      color = vec4(1.5f * normalize(flow.xyz) * clamp(dot(-normalize(normal_ms.xyz), light_dir.xyz), 0.f, 1.f), 1.f);
    }
  } else {
    color = vec4(0.f, 0.f, 0.f, 0.f);
  }
}