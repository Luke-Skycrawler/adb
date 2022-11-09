#include "env.h"
#include "global_variables.h"
#include "../test_cases/tests.h"
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
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

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
            globals.cubes = cube_blocks(2);
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
