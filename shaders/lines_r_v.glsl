#version 400

uniform mat4 MVP;

uniform uvec3 dim;
uniform uvec3 sam;

uniform float mag_threshold;
uniform float mag_scale;

uniform float r_scale;
uniform float r_seed;

uniform sampler3D flow_sampler;

layout(location = 0) in vec4 vertex;

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

void main () {
  uint x = gl_InstanceID % sam.x;
  uint y = (gl_InstanceID / sam.x) % sam.y;
  uint z = gl_InstanceID / (sam.x * sam.y);

  vec3 d = dim / vec3(sam + 1);
  
  vec3 position = d * vec3(x, y, z) - dim / 2.f + d;
  position += (vec3(
    snoise(position.xy * r_seed), 
    snoise(position.yz * r_seed + 17.f), 
    snoise(position.zx * r_seed - 43.f)) - 0.0f) * r_scale; 

  //CHECK TEXTURE COORDINATES
  // vec3 coord = (d * vec3(x, y, z) + d) / dim;
  vec3 coord = position / dim + 0.5f;


  vec4 flow = texture(flow_sampler, coord);
  
  if (vertex.w == 0.f || length(flow) < mag_threshold) {
    gl_Position = MVP * vec4(vertex.xyz + position, 1.f);
  } else {
    gl_Position = MVP * vec4(flow.xyz * mag_scale + position, 1.f);
  };
  
  color = vec4(vertex.w, 1.f - vertex.w, 0.f, 0.5f);
}