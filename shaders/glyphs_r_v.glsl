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

uniform float r_scale;
uniform float r_seed;

uniform sampler3D flow_sampler;
uniform sampler1D colormap_sampler;

layout(location = 0) in vec4 vertex;
layout(location = 1) in vec3 normal;

out vec4 color;

vec3 mod289(vec3 x) {
return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec2 mod289(vec2 x) {
return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec3 permute(vec3 x) {
return mod289(((x*34.0)+1.0)*x);
}
float snoise(vec2 v)
{
const vec4 C = vec4(0.211324865405187, // (3.0-sqrt(3.0))/6.0
0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
-0.577350269189626, // -1.0 + 2.0 * C.x
0.024390243902439); // 1.0 / 41.0
// First corner
vec2 i = floor(v + dot(v, C.yy) );
vec2 x0 = v - i + dot(i, C.xx);
// Other corners
vec2 i1;
//i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
//i1.y = 1.0 - i1.x;
i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
// x0 = x0 - 0.0 + 0.0 * C.xx ;
// x1 = x0 - i1 + 1.0 * C.xx ;
// x2 = x0 - 1.0 + 2.0 * C.xx ;
vec4 x12 = x0.xyxy + C.xxzz;
x12.xy -= i1;
// Permutations
i = mod289(i); // Avoid truncation effects in permutation
vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
+ i.x + vec3(0.0, i1.x, 1.0 ));
vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
m = m*m ;
m = m*m ;
// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
vec3 x = 2.0 * fract(p * C.www) - 1.0;
vec3 h = abs(x) - 0.5;
vec3 ox = floor(x + 0.5);
vec3 a0 = x - ox;
// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
// Compute final noise value at P
vec3 g;
g.x = a0.x * x0.x + h.x * x0.y;
g.yz = a0.yz * x12.xz + h.yz * x12.yw;
return 130.0 * dot(m, g);
}

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
  position += (vec3(
    snoise(position.xy * r_seed), 
    snoise(position.yz * r_seed + 17.f), 
    snoise(position.zx * r_seed - 43.f)) - 0.0f) * r_scale; 

  // vec3 coord = (d * vec3(x, y, z) + d) / dim;
  vec3 coord = position / dim + 0.5f;

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