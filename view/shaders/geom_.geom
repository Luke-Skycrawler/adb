#version 450 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VS_OUT {
    vec2 texCoords;
} gs_in[];

out int selected_alias;
out vec2 TexCoords; 
out float diffuse;
uniform int alias;

void main(){
    int i;
    // 1 1 1 | 1  
    // x x x | 0  
    // y y y | 0
    float x1=gl_in[0].gl_Position.x/gl_in[0].gl_Position.w;
    float x2=gl_in[1].gl_Position.x/gl_in[1].gl_Position.w;
    float x3=gl_in[2].gl_Position.x/gl_in[2].gl_Position.w;
    float y1=gl_in[0].gl_Position.y/gl_in[0].gl_Position.w;
    float y2=gl_in[1].gl_Position.y/gl_in[1].gl_Position.w;
    float y3=gl_in[2].gl_Position.y/gl_in[2].gl_Position.w;

    float dj[3];
    dj[0]=x2*y3-x3*y2;
    dj[1]=x3*y1-x1*y3;
    dj[2]=x1*y2-x2*y1;
    float D=dj[0]+dj[1]+dj[2];
    for(i=0;i<3;i++){
        dj[i]/=D;
        if(dj[i]<=0.0)break;
    }
    if(i==3){
        selected_alias=alias;
        diffuse=0.0;
    }
    else diffuse=1.0;
    for(i=0;i<3;i++){
        gl_Position = gl_in[i].gl_Position;
        TexCoords = gs_in[i].texCoords;
        EmitVertex();
    }
    EndPrimitive();
}