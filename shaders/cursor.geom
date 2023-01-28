#version 450 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;
layout (binding=0,r32i) uniform iimageBuffer alias_buffer;
in VS_OUT {
    vec2 texCoords;
    vec2 pickCoords;
} gs_in[];

// out int selected_alias;
out vec2 TexCoords; 
out float diffuse;
uniform int alias;

void main(){
    int i;
    // 1 1 1 | 1  
    // x x x | 0  
    // y y y | 0
    float x1=gs_in[0].pickCoords.x;
    float x2=gs_in[1].pickCoords.x;
    float x3=gs_in[2].pickCoords.x;
    float y1=gs_in[0].pickCoords.y;
    float y2=gs_in[1].pickCoords.y;
    float y3=gs_in[2].pickCoords.y;

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
        // selected_alias=alias;
        imageStore(alias_buffer,0 , ivec4(alias));
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