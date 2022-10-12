#include "env.h"
#include "shader.h"
#include "mesh.h"
#include "model.h"
#include "light.h"
#define _MAIN_CPP
#include "global_variables.h"
#include "../test_cases/tests.h"
#include "../model/time_integrator.h"
#include "../model/glue.h"

using namespace std;
//------------------ optional features ----------------------------
// #define FEATURE_MODEL
// #define FEATURE_EDGE
// #define FEATURE_POSTRENDER
//-----------------------------------------------------------------

render_cubes(Shader shader, vector<Cube> cubes){
    for (auto &c: cubes) {
        glm::mat4 A(from_eigen(c.A));
        shader.setMat4("model", A);
        renderCube();
    }
}
int main()
{
    vector<Cube> cubes;
    Cube *cube = spinning_cube();
    cubes.push_back(*cube);

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "adb viewer", NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, click_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCharCallback(window, text_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    // Shader lightingShader("shaders/geom.vert","shaders/geom.frag","shaders/geom_.geom",varyings);
    // cursor shader version 1

    // Shader lightingShader("shaders/cursor.vert", "shaders/cursor.frag", "shaders/cursor.geom");
    // version 2

    Shader lightingShader("shaders/1.color.vert", "shaders/1.color.frag");
    // shadow shader

    // Shader lightingShader("shaders/1.color_.vert", "shaders/1.color_.frag","shaders/pass_through.geom");
    // depth shader? don't use this

    unsigned int feedback_vbo = lightingShader.vbo[0], select_xfb = lightingShader.xfb;
    unsigned int select_program = lightingShader.ID;
    // unsigned int select_program=Feedback_Initialize(&feedback_vbo,&select_xfb);
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
#ifdef FEATURE_POSTRENDER
    unsigned int quadVBO, quadVAO;
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
#endif

    float corner[] = {
        0.5f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.0f, 0.0f,
        1.0f, 0.5f, 1.0f, 0.0f,
        0.5f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.5f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};
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

#ifdef FEATURE_MODEL
    // load models
    Model temple("nanosuit/nanosuit.obj");
    // Model temple("mods/gallery/gallery.obj");
#endif
    Light lights(LightPositions, 4);

    // load textures (we now use a utility function to keep the code more organized)
    // -----------------------------------------------------------------------------
    unsigned int diffuseMap = loadTexture("assets/container2.png");
    unsigned int specularMap = loadTexture("assets/container2_specular.png");
    //--------cube texture
    vector<string> faces{
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

#ifdef FEATURE_POSTRENDER
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
        cout << "error: framebuffer\n";
#endif

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
        globals.deltaTime = currentFrame - globals.lastFrame;
        globals.lastFrame = currentFrame;

        // input
        // -----

        processInput(window);
        // glBindFramebuffer(GL_FRAMEBUFFER,framebuffer);

#ifdef FEATURE_POSTRENDER
        glBindFramebuffer(GL_FRAMEBUFFER, globals.postrender ? framebuffer : 0);
#endif

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
        glm::mat4 projection = glm::perspective(glm::radians(globals.camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 100.0f);
        glm::mat4 view = globals.camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 tmpmodel = glm::scale(model, glm::vec3(scale, scale, scale));
        glm::vec3 box2Pos(0.3, 0.0, 1.2);
        glm::mat4 lightSpaceTrans = glm::lookAt(lightPos, glm::vec3(0.0f), globals.camera.WorldUp);
        if (globals.display_corner)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, globals.depthMapFBO);
            glEnable(GL_DEPTH_TEST);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            depthShader.use();

            // view/projection transformations
            model = glm::mat4(1.0f);

            implicit_euler(globals.deltaTime, cubes);
             

            // depthShader.setMat4("projection",projection);
            depthShader.setMat4("projection", glm::perspective(glm::radians(89.0f), (float)SHADOW_WIDTH / SHADOW_HEIGHT, 0.1f, 10.0f));
            depthShader.setMat4("view", lightSpaceTrans);
            depthShader.setMat4("model", model);
            depthShader.setVec3("viewPos", lightPos);
            // bind diffuse map
            renderPlane();
            render_cubes(depthShader, cubes);

            // model = glm::translate(model, box2Pos);
            // depthShader.setMat4("model", model);
            // renderCube();

#ifdef FEATURE_MODEL
            if (globals.model_draw)
            {
                depthShader.setMat4("model", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.1f, 0.0f)));
                temple.Draw(depthShader);
            }
#endif
#ifdef FEATURE_POSTRENDER
            glBindFramebuffer(GL_FRAMEBUFFER, globals.postrender ? framebuffer : 0);
#endif
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            model = glm::mat4(1.0f);
        }
        // glm::mat4 pick=glm::pickMatrix(glm::vec2(),glm::vec2(),glViewport())

        int viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        lightingShader.use();
        lightingShader.setVec2("pickPosition", glm::vec2(globals.lastX / viewport[2] * 2 - 1.0f, (1 - globals.lastY / viewport[3]) * 2 - 1.0f));
        if (globals.feedback)
        {
            // glEnable(GL_RASTERIZER_DISCARD);
            glUseProgram(select_program);
            glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, select_xfb);
            glBeginTransformFeedback(GL_TRIANGLES);

            // else glResumeTransformFeedback();
            // renderCube();
        }
        lightingShader.setMat4("lightView", glm::perspective(glm::radians(89.0f), (float)SHADOW_WIDTH / SHADOW_HEIGHT, 0.1f, 10.0f) * lightSpaceTrans);
        view = globals.camera.GetViewMatrix();
        lightingShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
        lightingShader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
        lightingShader.setVec3("lightPos", lightPos);
        lightingShader.setVec3("viewPos", globals.camera.Position);
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
        if (globals.display_corner)
        {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, globals.depthMap);
        }
        // FIXME: should do the select pass in reverse order
        lightingShader.setInt("alias", 5);
        renderPlane();
        // render the cube
        // lightingShader.setInt("alias", 3);
        // renderCube();
        // // glBindVertexArray(cubeVAO);
        // // glDrawArrays(GL_TRIANGLES, 0, 36);

        // glActiveTexture(GL_TEXTURE0);
        // lightingShader.use();
        // model = glm::translate(model, box2Pos);
        // lightingShader.setMat4("model", model);
        // glDrawArrays(GL_TRIANGLES, 0, 36);
        render_cubes(lightingShader, cubes);

#ifdef FEATURE_MODEL
        if (globals.model_draw)
        {
            // lightingShader.use();
            lightingShader.setInt("alias", 2);
            lightingShader.setMat4("model", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.1f, 0.0f)));
            temple.Draw(lightingShader);
        }
#endif
        if (!globals.cursor_hidden && globals.objectType)
        {
            model = glm::mat4(glm::mat3(globals.camera.Right, globals.camera.Up, -globals.camera.Front));
            model = glm::translate(model, globals.camera.Position * glm::mat3(model) + glm::vec3(0.0, 0.0, -3.0));
            lightingShader.setMat4("model", model);
            renderCube();
        }
        if (globals.feedback)
        {
            glEndTransformFeedback();
            int obj;
            // glDisable(GL_RASTERIZER_DISCARD);
            glGetNamedBufferSubData(buf, 0, sizeof(int), &obj);
            cout << obj << endl;
            //     bool b=glUnmapNamedBuffer(feedback_vbo);
            // glPauseTransformFeedback();
            // glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 5 * sizeof(int), NULL, GL_DYNAMIC_READ);
            glBufferData(GL_TEXTURE_BUFFER, sizeof(int), NULL, GL_DYNAMIC_READ);

            glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
        }

        // also draw the lamp object
        lights.Draw(globals.camera);
        if (globals.skybox)
        {
            glStencilMask(0x00);
            // globals.skybox
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
#ifdef FEATURE_EDGE
        if (globals.edge)
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
#endif
#ifdef FEATURE_POSTRENDER
        if (globals.postrender)
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
#endif
        if (globals.display_corner)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDisable(GL_DEPTH_TEST);
            cornerShader.use();
            glBindVertexArray(cornerVAO);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, globals.depthMap);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
#ifdef FEATURE_POSTRENDER
    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    // glDeleteVertexArrays(1, &cubeVAO);
    // glDeleteVertexArrays(1, &lightCubeVAO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    // glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
#endif
    glfwTerminate();
    return 0;
}

