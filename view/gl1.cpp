#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "shader.h"
#include "camera.h"
#include "mesh.h"
#include "model.h"
#include "gl1.h"
#include "light.h"

unsigned int depthMapFBO, depthMap;
static const int SHADOW_WIDTH = 800, SHADOW_HEIGHT = 600;
unsigned int loadCubemap(std::vector<std::string> faces);
int objectType = 0;
bool postrender = false, edge = false, skybox = false, model_draw = false,
     display_corner = true, Motion = false, feedback = false, cursor_hidden = true;
// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
static const char *varyings[] = {
    "selected_alias"
    // "alias"
    // "TexCoords","selected_alias","a","b","c"
};
// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// lighting
glm::vec3 LightPositions[] = {
    glm::vec3(1.2f, 1.0f, 2.0f),
    glm::vec3(1.2f, 2.0f, 0.0f),
    glm::vec3(-1.2f, 2.0f, 2.0f),
    glm::vec3(-1.2f, 2.0f, 0.0f)};
// glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
glm::vec3 &lightPos(LightPositions[0]);
int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, click_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    // Shader lightingShader("shaders/geom.vert","shaders/geom.frag","shaders/geom_.geom",varyings);
    Shader lightingShader("shaders/cursor.vert", "shaders/cursor.frag", "shaders/cursor.geom");
    unsigned int feedback_vbo = lightingShader.vbo[0], select_xfb = lightingShader.xfb;
    unsigned int select_program = lightingShader.ID;
    // unsigned int select_program=Feedback_Initialize(&feedback_vbo,&select_xfb);
    // Shader lightingShader("shaders/1.color.vert", "shaders/1.color.frag");
    // Shader lightingShader("shaders/1.color_.vert", "shaders/1.color_.frag","shaders/pass_through.geom");
    // Shader lightCubeShader("shaders/1.light_cube.vs", "shaders/1.light_cube.fs");
    Shader simpleShader("shaders/1.color.vs", "shaders/simple.fs");
    Shader screenShader("shaders/view.vs", "shaders/core.fs");
    Shader skyboxShader("shaders/skycube.vs", "shaders/skycube.fs");
    Shader depthShader("shaders/1.color.vs", "shaders/simple.frag");
    Shader cornerShader("shaders/view.vs", "shaders/core.frag");
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
    float corner[] = {
        0.5f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.0f, 0.0f,
        1.0f, 0.5f, 1.0f, 0.0f,
        0.5f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.5f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};
    float quad[] = {
        -1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), &quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int cornerVAO, cornerVBO;
    glGenVertexArrays(1, &cornerVAO);
    glGenBuffers(1, &cornerVBO);
    glBindVertexArray(cornerVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cornerVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(corner), &corner, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // load models
    Model temple("nanosuit/nanosuit.obj");
    Light lights(LightPositions, 4);
    // Model temple("mods/gallery/gallery.obj");
    // load textures (we now use a utility function to keep the code more organized)
    // -----------------------------------------------------------------------------
    unsigned int diffuseMap = loadTexture("assets/container2.png");
    unsigned int specularMap = loadTexture("assets/container2_specular.png");
    //--------cube texture
    std::vector<std::string> faces{
        "assets/skybox/right.jpg",
        "assets/skybox/left.jpg",
        "assets/skybox/top.jpg",
        "assets/skybox/bottom.jpg",
        "assets/skybox/front.jpg",
        "assets/skybox/back.jpg"};
    unsigned int cubemapTexture = loadCubemap(faces);

    screenShader.use();
    screenShader.setInt("screenTexture", 0);
    cornerShader.setInt("screenTexture", 0);
    unsigned int framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    unsigned int texColorBuffer;
    glGenTextures(1, &texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColorBuffer, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, texColorBuffer, 0);

    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "error: framebuffer\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    gen_preview_framebuffer();
    // shader configuration
    // --------------------
    lightingShader.use();
    lightingShader.setInt("material.diffuse", 0);
    lightingShader.setInt("material.specular", 1);
    lightingShader.setInt("shadowMap", 2);

    lightingShader.setFloat("material.shininess", 64);
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        // lightingShader.setFloat("time",currentFrame);
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----

        processInput(window);
        // glBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, postrender ? framebuffer : 0);
        // glBindFramebuffer(GL_FRAMEBUFFER,0);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilFunc(GL_ALWAYS, 1, 0XFF);
        glStencilMask(0XFF);
        lightingShader.use();

        // render
        // ------
        // glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // be sure to activate shader when setting uniforms/drawing objects
        float scale = 1.02;
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 tmpmodel = glm::scale(model, glm::vec3(scale, scale, scale));
        glm::vec3 box2Pos(0.3, 0.0, 1.2);
        glm::mat4 lightSpaceTrans = glm::lookAt(lightPos, glm::vec3(0.0f), camera.WorldUp);
        if (display_corner)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
            glEnable(GL_DEPTH_TEST);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            depthShader.use();

            // view/projection transformations
            model = glm::mat4(1.0f);

            // depthShader.setMat4("projection",projection);
            depthShader.setMat4("projection", glm::perspective(glm::radians(89.0f), (float)SHADOW_WIDTH / SHADOW_HEIGHT, 0.1f, 10.0f));
            depthShader.setMat4("view", lightSpaceTrans);
            depthShader.setMat4("model", model);
            depthShader.setVec3("viewPos", lightPos);
            // bind diffuse map
            renderPlane();
            // render the cube
            renderCube();

            model = glm::translate(model, box2Pos);
            depthShader.setMat4("model", model);
            renderCube();
            if (model_draw)
            {
                depthShader.setMat4("model", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.1f, 0.0f)));
                temple.Draw(depthShader);
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_FRAMEBUFFER, postrender ? framebuffer : 0);
            model = glm::mat4(1.0f);
        }
        // glm::mat4 pick=glm::pickMatrix(glm::vec2(),glm::vec2(),glViewport())

        int viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        lightingShader.use();
        lightingShader.setVec2("pickPosition", glm::vec2(lastX / viewport[2] * 2 - 1.0f, (1 - lastY / viewport[3]) * 2 - 1.0f));
        if (feedback)
        {
            // glEnable(GL_RASTERIZER_DISCARD);
            glUseProgram(select_program);
            glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, select_xfb);
            glBeginTransformFeedback(GL_TRIANGLES);

            // else glResumeTransformFeedback();
            // renderCube();
        }
        lightingShader.setMat4("lightView", glm::perspective(glm::radians(89.0f), (float)SHADOW_WIDTH / SHADOW_HEIGHT, 0.1f, 10.0f) * lightSpaceTrans);
        view = camera.GetViewMatrix();
        lightingShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
        lightingShader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
        lightingShader.setVec3("lightPos", lightPos);
        lightingShader.setVec3("viewPos", camera.Position);
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
        if (display_corner)
        {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, depthMap);
        }
        // FIXME: should do the select pass in reverse order
        lightingShader.setInt("alias", 5);
        renderPlane();
        // render the cube
        lightingShader.setInt("alias", 3);
        renderCube();
        // glBindVertexArray(cubeVAO);
        // glDrawArrays(GL_TRIANGLES, 0, 36);

        glActiveTexture(GL_TEXTURE0);
        lightingShader.use();
        model = glm::translate(model, box2Pos);
        lightingShader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        if (model_draw)
        {
            // lightingShader.use();
            lightingShader.setInt("alias", 2);
            lightingShader.setMat4("model", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.1f, 0.0f)));
            temple.Draw(lightingShader);
        }
        if (!cursor_hidden && objectType)
        {
            model = glm::mat4(glm::mat3(camera.Right, camera.Up, -camera.Front));
            model = glm::translate(model, camera.Position * glm::mat3(model) + glm::vec3(0.0, 0.0, -3.0));
            lightingShader.setMat4("model", model);
            renderCube();
        }
        if (feedback)
        {
            glEndTransformFeedback();
            int obj;
            // glDisable(GL_RASTERIZER_DISCARD);
            glGetNamedBufferSubData(buf, 0, sizeof(int), &obj);
            std::cout << obj << std::endl;
            //     bool b=glUnmapNamedBuffer(feedback_vbo);
            // glPauseTransformFeedback();
            // glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 5 * sizeof(int), NULL, GL_DYNAMIC_READ);
            glBufferData(GL_TEXTURE_BUFFER, sizeof(int), NULL, GL_DYNAMIC_READ);

            glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
        }

        // also draw the lamp object
        lights.Draw(camera);
        if (skybox)
        {
            glStencilMask(0x00);
            // skybox
            // glDepthMask(GL_FALSE);
            glm::mat4 skyview = glm::mat4(glm::mat3(view));
            glDepthFunc(GL_LEQUAL);
            skyboxShader.use();
            skyboxShader.setMat4("projection", projection);
            skyboxShader.setMat4("view", skyview);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
            renderCube();
            // glBindVertexArray(cubeVAO);
            // glDrawArrays(GL_TRIANGLES,0,36);

            // glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);
        }
        if (edge)
        {
            glStencilFunc(GL_NOTEQUAL, 1, 0XFF);
            // glStencilMask(0x00);
            glDisable(GL_DEPTH_TEST);
            simpleShader.use();
            simpleShader.setMat4("projection", projection);
            simpleShader.setMat4("view", view);
            simpleShader.setMat4("model", tmpmodel);
            renderCube();
            // glDrawArrays(GL_TRIANGLES,0,36);
            glStencilMask(0xFF);
            glEnable(GL_DEPTH_TEST);
            glStencilFunc(GL_ALWAYS, 1, 0XFF);
        }

        if (postrender)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDisable(GL_DEPTH_TEST);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            screenShader.use();
            glBindVertexArray(quadVAO);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texColorBuffer);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        if (display_corner)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDisable(GL_DEPTH_TEST);
            cornerShader.use();
            glBindVertexArray(cornerVAO);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, depthMap);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    // glDeleteVertexArrays(1, &cubeVAO);
    // glDeleteVertexArrays(1, &lightCubeVAO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    // glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}
void gen_preview_framebuffer()
{
    glGenFramebuffers(1, &depthMapFBO);
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                 SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    // glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,SCR_WIDTH,SCR_HEIGHT,0,GL_RGB,GL_UNSIGNED_BYTE,NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    // glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthMap, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (Motion)
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            lightPos += 2.5f * deltaTime * camera.Front;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            lightPos -= 2.5f * deltaTime * camera.Front;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            lightPos -= 2.5f * deltaTime * camera.Right;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            lightPos += 2.5f * deltaTime * camera.Right;
    }
    else
    {
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

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        postrender = !postrender;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        edge = !edge;
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
        skybox = !skybox;
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
        model_draw = !model_draw;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        display_corner = !display_corner;
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        Motion = !Motion;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        if (cursor_hidden)
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            firstMouse = true;
            glfwSetCursorPos(window, SCR_WIDTH / 2, SCR_HEIGHT / 2);
        }
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        cursor_hidden = !cursor_hidden;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void click_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (cursor_hidden)
        return;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        feedback = true;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
        feedback = false;
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xpos, double ypos)
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

    if (cursor_hidden)
        camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    if (cursor_hidden)
        camera.ProcessMouseScroll(yoffset);
    else
    {
        if (yoffset > 0)
            objectType += yoffset;
        else
            objectType = 0;
    }
}
unsigned int loadCubemap(std::vector<std::string> faces)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++)
    {
        unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                         0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}
unsigned int Feedback_Initialize(unsigned int *_vbo, unsigned int *_xfb)
{
    static unsigned int xfb, sort_prog, geometry, vert, vbo[2];
    glGenTransformFeedbacks(1, &xfb);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, xfb);

    sort_prog = glCreateProgram();

    static const char *sort_vs_source =
        "#version 410\n"
        "\n"
        "uniform mat4 model_matrix;\n"
        "\n"
        "layout (location = 0) in vec4 position;\n"
        "layout (location = 1) in vec3 normal;\n"
        "\n"
        "out vec3 vs_normal;\n"
        "\n"
        "void main(void)\n"
        "{\n"
        "    vs_normal = (model_matrix * vec4(normal, 0.0)).xyz;\n"
        "    gl_Position = model_matrix * position;\n"
        "}\n";

    static const char *sort_gs_source =
        "#version 410\n"
        "\n"
        "layout (triangles) in;\n"
        "layout (points, max_vertices = 3) out;\n"
        "\n"
        "uniform mat4 projection_matrix;\n"
        "\n"
        "in vec3 vs_normal[];\n"
        "out int selected_alias;\n"
        "layout (stream = 0) out vec4 rf_position;\n"
        "layout (stream = 0) out vec3 rf_normal;\n"
        "\n"
        "layout (stream = 1) out vec4 lf_position;\n"
        "layout (stream = 1) out vec3 lf_normal;\n"
        "\n"
        "void main(void)\n"
        "{\n"
        "    vec4 A = gl_in[0].gl_Position;\n"
        "    vec4 B = gl_in[1].gl_Position;\n"
        "    vec4 C = gl_in[2].gl_Position;\n"
        "    vec3 AB = (B - A).xyz;\n"
        "    vec3 AC = (C - A).xyz;\n"
        "    vec3 face_normal = cross(AB, AC);\n"
        "    int i;\n"
        "    selected_alias=100;\n"
        "    if (face_normal.x < 0.0)\n"
        "    {\n"
        "        for (i = 0; i < gl_in.length(); i++)\n"
        "        {\n"
        "            rf_position = projection_matrix * (gl_in[i].gl_Position - vec4(30.0, 0.0, 0.0, 0.0));\n"
        "            rf_normal = vs_normal[i];\n"
        "            EmitStreamVertex(0);\n"
        "        }\n"
        "        EndStreamPrimitive(0);\n"
        "    }\n"
        "    else\n"
        "    {\n"
        "        for (i = 0; i < gl_in.length(); i++)\n"
        "        {\n"
        "            lf_position = projection_matrix * (gl_in[i].gl_Position + vec4(30.0, 0.0, 0.0, 0.0));\n"
        "            lf_normal = vs_normal[i];\n"
        "            EmitStreamVertex(1);\n"
        "        }\n"
        "        EndStreamPrimitive(1);\n"
        "    }\n"
        "}\n";
    // vglAttachShaderSource(sort_prog, GL_VERTEX_SHADER, sort_vs_source);
    // vglAttachShaderSource(sort_prog, GL_GEOMETRY_SHADER, sort_gs_source);
    geometry = glCreateShader(GL_GEOMETRY_SHADER);
    vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(geometry, 1, &sort_gs_source, NULL);
    glCompileShader(geometry);
    glShaderSource(vert, 1, &sort_vs_source, NULL);
    glCompileShader(vert);
    glAttachShader(sort_prog, vert);
    glAttachShader(sort_prog, geometry);

    static const char *varyings[] =
        {
            "selected_alias"};

    glTransformFeedbackVaryings(sort_prog, 1, varyings, GL_INTERLEAVED_ATTRIBS);

    glLinkProgram(sort_prog);
    glUseProgram(sort_prog);

    // model_matrix_pos = glGetUniformLocation(sort_prog, "model");
    // projection_matrix_pos = glGetUniformLocation(sort_prog, "projection");

    // glGenVertexArrays(2, vao);
    glGenBuffers(2, vbo);

    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, vbo[i]);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 5 * sizeof(GLfloat), NULL, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, vbo[i]);
    }
    if (_xfb)
    {
        *_vbo = vbo[0];
        *_xfb = xfb;
    }
    return sort_prog;
}
