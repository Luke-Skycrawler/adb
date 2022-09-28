#version 330 core
out vec4 FragColor;
in vec3 FragPos;

uniform vec3 viewPos;
void main()
{
    // vec3 viewDir= normalize(viewPos-FragPos);
    vec3 depth = vec3(length(viewPos-FragPos));
    FragColor = vec4(depth, 1.0);
}
