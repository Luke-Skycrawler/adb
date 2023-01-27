#include "env.h"
#include "global_variables.h"
#include "../test_cases/tests.h"


#include <fstream> 
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#define DURATION_TO_DOUBLE(X) duration_cast<duration<double>>((X)).count()
using json = nlohmann::json;
void reset(bool init)
{
    std::ifstream f("../config.json");
    json data = json::parse(f);
    double dt = data["dt"];
    auto predefined = data["predefined_case"]["enable"];
    auto g = data["gravity"];
    globals.gravity = vec3(0.0, g? -9.8 : 0.0 , 0.0);
    globals.col_set = data["col_set"];
    globals.line_search = data["line_search"];
    globals.upper_bound = data["uppper_bound"];
    globals.sparse = data["solver"] != "dense";
    globals.dense = data["solver"] != "sparse";
    globals.ee = data["ee"];
    globals.pt = data["pt"];
    globals.ground = data["ground"];
    globals.psd = data["psd"];
    globals.damp = data["damping"]["enable"];
    globals.backoff = data["backoff"];
    if (predefined) {
        int id = data["predefined_case"]["id"];
        if (id == 1) {
            globals.cubes.push_back(spinning_cube());
        }
        else {
            globals.cubes.clear();
            cube_blocks(2);
        }
    }
    else {
        globals.cubes.clear();
        customize(data["predefined_case"]["custom"]);
    }
    int n_cubes = globals.cubes.size();
    globals.writelock_cols.resize(n_cubes);
    auto ptr = globals.writelock_cols.data();
    for(int i = 0; i < n_cubes; i++){
        omp_init_lock(ptr + i);
    }
    globals.tot_iter = 0;
    globals.ts = 0;
    glfwSetTime(0.0);
    {
        globals.dt = data["dt"];
        globals.max_iter = data["max_iter"];
        globals.alpha = data["damping"]["alpha"];
        globals.beta = data["damping"]["beta"];
        globals.safe_factor = data["safe_factor"];
        globals.mu = data["mu"];
        globals.eps_x = data["eps_x"];
    }
    if (init) {
        auto sh_args = data["spatial hashing"];
        globals.sh = make_unique<spatial_hashing>(sh_args["xyz_bits"], sh_args["n_buffer"], sh_args["min_xyz"], sh_args["max_xyz"], sh_args["dx"]);
    }
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (globals.motion)
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            lightPos += 2.5f * globals.deltaTime * globals.camera.Front;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            lightPos -= 2.5f * globals.deltaTime * globals.camera.Front;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            lightPos -= 2.5f * globals.deltaTime * globals.camera.Right;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            lightPos += 2.5f * globals.deltaTime * globals.camera.Right;
    }
    else
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            globals.camera.ProcessKeyboard(FORWARD, globals.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            globals.camera.ProcessKeyboard(BACKWARD, globals.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            globals.camera.ProcessKeyboard(LEFT, globals.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            globals.camera.ProcessKeyboard(RIGHT, globals.deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
        double t = glfwGetTime();
        spdlog::error("running time = {}, ts = {}, #iters = {}, fps = {}", t, globals.ts, globals.tot_iter, globals.ts / t);
        glfwSetWindowShouldClose(window, true);
    }

    // if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    //     globals.postrender = !globals.postrender;
    // if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    //     globals.edge = !globals.edge;
    // if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
    //     globals.skybox = !globals.skybox;
    // if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
    //     globals.model_draw = !globals.model_draw;
}

void text_callback(GLFWwindow *window, unsigned int c){
    Cube* p;
    switch (c) {
        case 'c': case 'C':
            globals.display_corner = !globals.display_corner;
            break;
        case 'l': case 'L':
            globals.motion = !globals.motion;
            break;
        case 'r': case 'R':
            //p = &(globals.cubes[0]);
            //globals.cubes.clear();
            //globals.cubes.push_back(*spinning_cube());
            //delete p;
            // globals.cubes = cube_blocks(2);
            reset();
            break;
        case 'q': case 'Q':
            if (globals.cursor_hidden)
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                globals.firstMouse = true;
                glfwSetCursorPos(window, SCR_WIDTH / 2, SCR_HEIGHT / 2);
            }
            else
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            globals.cursor_hidden = !globals.cursor_hidden;
            break;
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
    if (globals.cursor_hidden)
        return;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        globals.feedback = true;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
        globals.feedback = false;
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (globals.firstMouse)
    {
        globals.lastX = xpos;
        globals.lastY = ypos;
        globals.firstMouse = false;
    }

    float xoffset = xpos - globals.lastX;
    float yoffset = globals.lastY - ypos; // reversed since y-coordinates go from bottom to top

    globals.lastX = xpos;
    globals.lastY = ypos;

    if (globals.cursor_hidden)
        globals.camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    if (globals.cursor_hidden)
        globals.camera.ProcessMouseScroll(yoffset);
    else
    {
        if (yoffset > 0)
            globals.objectType += yoffset;
        else
            globals.objectType = 0;
    }
}
