#include "env.h"
#include "shader.h"
#include "mesh.h"
#include "model.h"
#include "light.h"
#define _MAIN_CPP
#include "global_variables.h"
#include "cube.h"
#include "abd.h"
#include "../model/time_integrator.h"
#include "../model/glue.h"
#include <glm/gtx/string_cast.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <omp.h>
//#define FEATURE_MODEL
using namespace std;
using namespace Eigen;
//------------------ optional features ----------------------------
// #define FEATURE_MODEL
// #define FEATURE_EDGE
// #define FEATURE_POSTRENDER
//-----------------------------------------------------------------

vector<int> Cube::_edges {}, Cube::_indices {};
void render_cubes(Shader shader, vector<unique_ptr<AffineBody>> &cubes)
{
    for (int i = 0; i < cubes.size(); i++)
    {
        auto& c {*cubes[i]};
        glm::mat4 A(from_eigen(c.A));
        // glm::mat4 A = glm::mat4(1.0f);
        for (int i = 0; i < 3; i++)
            A[3][i] = c.p(i);
        shader.setMat4("model", A);
        c.draw(shader);
    }
}
int main()
{
    Cube::gen_indices();
    int n_proc = omp_get_num_procs();
    omp_set_num_threads(n_proc);
    setNbThreads(n_proc);
    initParallel();
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
     Shader lightingShader("shaders/shadow/shadow.vert", "shaders/shadow/shadow.frag");

    unsigned int feedback_vbo = lightingShader.vbo[0], select_xfb = lightingShader.xfb;
    unsigned int select_program = lightingShader.ID;
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

    Light lights(globals.light_positions, 4);

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

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    gen_preview_framebuffer();
    // shader configuration
    // --------------------
    lightingShader.use();
    lightingShader.setInt("material.diffuse", 0);
    lightingShader.setInt("material.specular", 1);
    lightingShader.setInt("shadowMap", 2);

    lightingShader.setFloat("material.shininess", 64);

    reset(true);
    int n_cubes = globals.cubes.size();
    globals.edges = utils::gen_edge_list(globals.cubes, n_cubes);
    globals.points = utils::gen_point_list(globals.cubes, n_cubes);
    globals.triangles = utils::gen_triangle_list(globals.cubes, n_cubes);
    
    auto &lightPos{globals.light_positions[0]};
    // be sure to call after glfw intiailzation 
    ABD abd(globals);
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
        // glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
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
        std::string trace_folder = globals.trace_folder;
        bool init = globals.ts == 0;
        if (!globals.player) {
            for (int i = 0; i < 1; i++)
                abd.implicit_euler(globals.dt);
            player_save(trace_folder, globals.ts, globals.cubes, init);
            if (globals.ending_ts > 0 && globals.ts >= globals.ending_ts) 
                exit_callback(window);
                // glfwSetWindowShouldClose(window, true);
        }
        else{
            spdlog::info("timestep = {}", globals.ts);
            player_load(trace_folder,globals.ts++, globals.cubes);
        }
        if (globals.display_corner)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, globals.depthMapFBO);
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
            if (globals.ground)
                renderPlane();
            render_cubes(depthShader, globals.cubes);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            model = glm::mat4(1.0f);
        }

        int viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        lightingShader.use();
        lightingShader.setVec2("pickPosition", glm::vec2(globals.lastX / viewport[2] * 2 - 1.0f, (1 - globals.lastY / viewport[3]) * 2 - 1.0f));
        if (globals.feedback) {
            // glEnable(GL_RASTERIZER_DISCARD);
            glUseProgram(select_program);
            glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, select_xfb);
            glBeginTransformFeedback(GL_TRIANGLES);

            // else glResumeTransformFeedback();
            // renderCube();
        }
        lightingShader.setMat4("lightView", glm::perspective(glm::radians(89.0f), (float)SHADOW_WIDTH / SHADOW_HEIGHT, 0.1f, 10.0f) * lightSpaceTrans);
        view = globals.camera.GetViewMatrix();
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
        if (globals.display_corner) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, globals.depthMap);
        }
        // FIXME: should do the select pass in reverse order
        lightingShader.setInt("alias", 5);
        lightingShader.setVec3("objectColor", 0.0f, 0.5f, 1.0f);
        if (globals.ground)
            renderPlane();
        lightingShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);

        render_cubes(lightingShader, globals.cubes);
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
    glfwTerminate();
    return 0;
}
