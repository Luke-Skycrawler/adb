#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "../view/shader.h"
#include "../view/camera.h"
#include "../view/mesh.h"
#include "../view/model.h"
#include "utils.h"
#include "../view/light.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
void render_arrow(float len);
unsigned int depthMapFBO,depthMap;
static const int SHADOW_WIDTH=800,SHADOW_HEIGHT=600;
bool model_draw = true, display_corner = true, move_light = false;
// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
// camera
static const float z_camera = 3.0f;
Camera camera(glm::vec3(0.0f, 0.0f, z_camera));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool concentric = true;

// lighting
glm::vec3 LightPositions[]={
    glm::vec3(0.0f, 0.0f, 2.5f),
    glm::vec3(-2.0f, -2.0f, 5.0f),
    glm::vec3(2.0f, 2.0f, 5.0f),
    glm::vec3(2.0f, -2.0f, 5.0f)   
};
glm::vec3 &lightPos(LightPositions[0]);
int main()
{

    // glfw: initialize and configure
    // ------------------------------------------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // ------------------------------------------------------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "glad framework", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCharCallback(window, char_callback);
    //glfwSetMouseButtonCallback(window, click_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ------------------------------------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // ------------------------------------------------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader programs
    // ------------------------------------------------------------------
    //Shader lightingShader("shaders/cursor/cursor.vert", "shaders/cursor/cursor.frag", "shaders/cursor/cursor.geom");
    Shader lightingShader("shaders/shadow/shadow.vert", "shaders/shadow/shadow.frag");
    //Shader lightingShader("shaders/light/multi_light.vert", "shaders/light/multi_light.frag");
    Shader depthShader("shaders/depth/depth.vert","shaders/depth/depth.frag");
    Shader cornerShader("shaders/corner/corner.vert","shaders/corner/corner.frag");
    // select buffers setup
    // ------------------------------------------------------------------
    unsigned int tex, buf;
    // Generate a name for the buffer object, bind it to the
    // GL_TEXTURE_BINDING, and allocate 4K for the buffer
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(int), NULL, GL_DYNAMIC_READ);
    // Generate a new name for our texture
    glGenTextures(1, &tex);
    // Bind it to the buffer texture target to create it
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    // Attach the buffer object to the texture and specify format as
    // single channel floating point
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, buf);
    // Now bind it for read-write to one of the image units
    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    // ------------------------------------------------------------------
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    unsigned int quadVBO, quadVAO;
    float corner[]={
         0.5f,  1.0f,  0.0f, 1.0f,
         0.5f,  0.5f,  0.0f, 0.0f,
         1.0f,  0.5f,  1.0f, 0.0f,
         0.5f,  1.0f,  0.0f, 1.0f,
         1.0f,  0.5f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    float quad[]={
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), &quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int cornerVAO,cornerVBO;
    glGenVertexArrays(1, &cornerVAO);
    glGenBuffers(1, &cornerVBO);
    glBindVertexArray(cornerVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cornerVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(corner), &corner, GL_STATIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // load models 
    Light lights(LightPositions,4);
    // load textures (we now use a utility function to keep the code more organized)
    // ------------------------------------------------------------------
    unsigned int diffuseMap = loadTexture("assets/wood.png");
    unsigned int specularMap = loadTexture("assets/wood_specular.png");

    cornerShader.setInt("screenTexture",0);
    
    unsigned int texColorBuffer;
    glGenTextures(1,&texColorBuffer);
    glBindTexture(GL_TEXTURE_2D,texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,SCR_WIDTH,SCR_HEIGHT,0,GL_RGB,GL_UNSIGNED_BYTE,NULL);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texColorBuffer,0);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_TEXTURE_2D,texColorBuffer,0);

    unsigned int rbo;
    glGenRenderbuffers(1,&rbo);
    glBindRenderbuffer(GL_RENDERBUFFER,rbo);
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,SCR_WIDTH,SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_RENDERBUFFER,rbo);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
        std::cout<<"error: framebuffer\n";
    glBindFramebuffer(GL_FRAMEBUFFER,0);

    gen_preview_framebuffer();
    // ------------------------------------------------------------------
    // shader configuration
    // ------------------------------------------------------------------
    lightingShader.use();
    lightingShader.setInt("material.diffuse", 0);
    lightingShader.setInt("material.specular", 1);
    lightingShader.setInt("shadowMap",2);
    
    lightingShader.setFloat("material.shininess",64);

    // ------------------------------------------------------------------
    // render loop
    // ------------------------------------------------------------------

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 130";
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    // io.Fonts->AddFontDefault();
    // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    // ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    // IM_ASSERT(font != NULL);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        // lightingShader.setFloat("time",currentFrame);
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text."); // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // input
        processInput(window);
        // render setup
        glBindFramebuffer(GL_FRAMEBUFFER,0);
        glClearColor(0.1f,0.1f,0.1f,1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
        glStencilOp(GL_KEEP,GL_KEEP,GL_REPLACE);
        glStencilFunc(GL_ALWAYS,1,0XFF);
        glStencilMask(0XFF);
        lightingShader.use();

        // ------------------------------------------------------------------
        // render
        // ------------------------------------------------------------------
        // be sure to activate shader when setting uniforms/drawing objects
        float scale=1.02;
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 tmpmodel=glm::scale(model,glm::vec3(scale,scale,scale));
        glm::vec3 box2Pos(0.3,0.0,1.2);
        glm::mat4 lightSpaceTrans = glm::lookAt(lightPos,glm::vec3(0.0f),camera.WorldUp);
        
        const auto renderScene = [&](Shader &shader)
        {
            renderPlane();

            // translate and render another
            model = glm::translate(model, box2Pos);
            shader.setMat4("model", model);
             renderCube();
            // render particle
            // shader.setMat4("model");
            
            //model = glm::translate(glm::mat4(0.1f), glm::vec3(globals.p.pos * 10.0f, 0.0f));
            //shader.setMat4("model", model);


            //shader.setVec3("objectColor", 1.0f, 1.0f, 1.0f);
            //particle.Draw(shader);
            //

            //model = glm::translate(glm::mat4(0.1f), glm::vec3(globals.p.pos * 10.0f, 0.5f));
            //if (u != 0.0f) {
            //model[1][1] = globals.p.f[1] / u;
            //model[1][0] = globals.p.f[0] / u;
            //model[0][1] = globals.p.f[0] / u;
            //model[0][0] = -globals.p.f[1] / u;

            //}
            //u /= 5.0;
            //u = fmax(fmin(1.0, u), 0.0);

            //shader.setMat4("model", model);
            //shader.setVec3("objectColor", u, 0.5f, 1.0f - u);
            //render_arrow(u * 2.0);
            
            // TODO: add rendering code here
            // ----------------------------------------------------





            // ----------------------------------------------------
        };
        if(display_corner){
            glBindFramebuffer(GL_FRAMEBUFFER,depthMapFBO);
            glEnable(GL_DEPTH_TEST);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            depthShader.use();

            // view/projection transformations
            model = glm::mat4(1.0f);

            // depthShader.setMat4("projection",projection);
            depthShader.setMat4("projection",glm::perspective(glm::radians(89.0f),(float)SHADOW_WIDTH/SHADOW_HEIGHT,0.1f,10.0f));
            depthShader.setMat4("view",lightSpaceTrans);
            depthShader.setMat4("model",model);
            depthShader.setVec3("viewPos",lightPos);
            renderScene(depthShader);
            glBindFramebuffer(GL_FRAMEBUFFER,0);
            model = glm::mat4(1.0f);
        }
 
        int viewport[4];
        glGetIntegerv(GL_VIEWPORT,viewport);
        lightingShader.use();
        lightingShader.setVec2("pickPosition",glm::vec2(lastX/viewport[2]*2-1.0f,(1-lastY/viewport[3])*2-1.0f));
        lightingShader.setMat4("lightView",glm::perspective(glm::radians(89.0f),(float)SHADOW_WIDTH/SHADOW_HEIGHT,0.1f,10.0f)*lightSpaceTrans);
        view = camera.GetViewMatrix();
        lightingShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
        lightingShader.setVec3("lightColor",  1.0f, 1.0f, 1.0f);
        lightingShader.setVec3("lightPos",lightPos);
        lightingShader.setVec3("viewPos",camera.Position);
        // view/projection transformations
        lightingShader.setMat4("projection", projection);
        lightingShader.setMat4("view", view);

        // world transformation
        lightingShader.setMat4("model", model);

        // bind diffuse map
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseMap);
        // bind specular map
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, specularMap);
        if(display_corner){
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D,depthMap);
        }
        renderScene(lightingShader);
        // also draw the lamp object
        lights.Draw(camera);

        if(display_corner){
            glBindFramebuffer(GL_FRAMEBUFFER,0);
            glDisable(GL_DEPTH_TEST);
            cornerShader.use();
            glBindVertexArray(cornerVAO);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,depthMap);
            glDrawArrays(GL_TRIANGLES,0,6);
        }
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // ------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}
void gen_preview_framebuffer(){
    glGenFramebuffers(1,&depthMapFBO);
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
                SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if(move_light){
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            lightPos+= 2.5f*deltaTime*camera.Front;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            lightPos-= 2.5f*deltaTime*camera.Front;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            lightPos-= 2.5f*deltaTime*camera.Right;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            lightPos+= 2.5f*deltaTime*camera.Right;
    }
     else{
         if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
             camera.ProcessKeyboard(FORWARD, deltaTime);
         if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
             camera.ProcessKeyboard(BACKWARD, deltaTime);
         if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
             camera.ProcessKeyboard(LEFT, deltaTime);
         if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
             camera.ProcessKeyboard(RIGHT, deltaTime);
     }
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

void char_callback(GLFWwindow* window, unsigned int codepoint)
{
    if (codepoint == 'c')
        display_corner=!display_corner;
    if (codepoint == 'l')
        move_light=!move_light;
    
}
void renderPlane(){

    static unsigned int planeVBO,planeVAO=0;
    static float planeVertices[] = {
        // positions            // normals         // texcoords
         25.0f,  25.0f,-0.5f,  0.0f, 1.0f, 0.0f,  25.0f,  0.0f,
        -25.0f,  25.0f,-0.5f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
        -25.0f, -25.0f,-0.5f,  0.0f, 1.0f, 0.0f,   0.0f, 25.0f,
         25.0f,  25.0f,-0.5f,  0.0f, 1.0f, 0.0f,  25.0f,  0.0f,
        -25.0f, -25.0f,-0.5f,  0.0f, 1.0f, 0.0f,   0.0f, 25.0f,
         25.0f, -25.0f,-0.5f,  0.0f, 1.0f, 0.0f,  25.0f, 25.0f
    };
    if(planeVAO == 0){
        // plane VAO
        glGenVertexArrays(1, &planeVAO);
        glGenBuffers(1, &planeVBO);
        glBindVertexArray(planeVAO);
        glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindVertexArray(0);
    }
    glBindVertexArray(planeVAO);
    glDrawArrays(GL_TRIANGLES,0,6);
}
void renderCube(int light){
    static float vertices[] = {
        // positions          // normals           // texture coords
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f
    };
    // first, configure the cube's VAO (and VBO)
    static unsigned int VBO=-1, cubeVAO=-1,lightCubeVAO;
    if(cubeVAO==-1){
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &VBO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindVertexArray(cubeVAO);

        // position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(6*sizeof(float)));
        glEnableVertexAttribArray(2);

        // second, configure the light's VAO (VBO stays the same; the vertices are the same for the light object which is also a 3D cube)
        glGenVertexArrays(1, &lightCubeVAO);
        glBindVertexArray(lightCubeVAO);

        // we only need to bind to the VBO (to link it with glVertexAttribPointer), no need to fill it; the VBO's data already contains all we need (it's already bound, but we do it again for educational purposes)
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }
    glBindVertexArray(light?lightCubeVAO:cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}
void render_arrow(float len){
    float max_len = 2.0;
    float sqrt3by2 = sqrt(3.0f) / 2, width = 0.5, height = 2.0f, 
        bottom_width = 0.2f, bottom_height = 1.0f, 
        mid_height = 1.0f;
    height = fmax(len, 2.0f);
    mid_height = bottom_height = height - 1.0f; //len < max_len ? len / 2.0: len - 1;    
    width = 0.5f;
    bottom_width = width * 0.4f;
    float vertices[] = {
        0.0, height, 0.0,
        width, mid_height, 0.0,
        -width / 2.0,mid_height, sqrt3by2 * width, 

        0.0, height, 0.0,
        width, mid_height, 0.0,
        -width / 2.0,mid_height, -sqrt3by2 * width,

        
        0.0, height, 0.0,
        -width / 2.0,mid_height, sqrt3by2 * width, 
        -width / 2.0,mid_height, -sqrt3by2 * width,

        width  , mid_height, 0.0,
        -width / 2.0,mid_height, sqrt3by2 * width, 
        -width / 2.0,mid_height, -sqrt3by2 * width,

        bottom_width  , mid_height, 0.0,
        -bottom_width / 2.0,mid_height, sqrt3by2 * bottom_width, 
        -bottom_width / 2.0,mid_height, -sqrt3by2 * bottom_width,
        
        bottom_width  , mid_height - bottom_height, 0.0,
        -bottom_width / 2.0, mid_height - bottom_height, sqrt3by2 * bottom_width, 
        -bottom_width / 2.0, mid_height - bottom_height, -sqrt3by2 * bottom_width,


        bottom_width  , mid_height, 0.0,
        bottom_width  , mid_height - bottom_height, 0.0,
        -bottom_width / 2.0, mid_height - bottom_height, sqrt3by2 * bottom_width, 

        bottom_width  , mid_height, 0.0,
        -bottom_width / 2.0, mid_height, sqrt3by2 * bottom_width, 
        -bottom_width / 2.0, mid_height - bottom_height, sqrt3by2 * bottom_width, 


        -bottom_width / 2.0, mid_height, sqrt3by2 * bottom_width, 
        -bottom_width / 2.0, mid_height, -sqrt3by2 * bottom_width,        
        -bottom_width / 2.0, mid_height - bottom_height, -sqrt3by2 * bottom_width,

        -bottom_width / 2.0, mid_height, sqrt3by2 * bottom_width,        
        -bottom_width / 2.0, mid_height - bottom_height, sqrt3by2 * bottom_width, 
        -bottom_width / 2.0, mid_height - bottom_height, -sqrt3by2 * bottom_width,


        bottom_width  , mid_height, 0.0,
        -bottom_width / 2.0, mid_height, -sqrt3by2 * bottom_width,
        bottom_width  , mid_height - bottom_height, 0.0,
        

        -bottom_width / 2.0, mid_height, -sqrt3by2 * bottom_width,
        bottom_width  , mid_height - bottom_height, 0.0,
        -bottom_width / 2.0, mid_height - bottom_height, -sqrt3by2 * bottom_width,

    };
    // first, configure the cube's VAO (and VBO)
    static unsigned int vbo=-1, vao=-1;
    if(vao==-1){
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

