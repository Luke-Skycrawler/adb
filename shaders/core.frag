#version 330 core
out vec4 color;
in vec2 TexCoords;
float near_plane=1.0,far_plane=100.0;
uniform sampler2D depthMap;
float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
}
void main()
{             
    float depthValue = texture(depthMap, TexCoords).r;
    color = vec4(vec3(LinearizeDepth(depthValue)/far_plane), 1.0);
    // color = vec4(vec3(depthValue), 1.0);
}