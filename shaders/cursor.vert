#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoords;

out VS_OUT {
    vec2 texCoords;
    vec2 pickCoords;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform vec2 pickPosition; 
// uniform mat4 pickMatrix;

void main()
{
    vs_out.texCoords = aTexCoords;
    gl_Position = projection * view * model * vec4(aPos, 1.0); 
    vec2 tmp=gl_Position.xy;
    tmp/=gl_Position.w;
    vs_out.pickCoords=tmp-pickPosition;
}