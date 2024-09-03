cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(abd)
# set(IPC_TOOLKIT_BUILD_PYTHON ON)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package (Eigen3 CONFIG REQUIRED)
# find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    GIT_TAG main
)
FetchContent_GetProperties(ipc_toolkit)
FetchContent_MakeAvailable(ipc_toolkit)
set(HEADERS 
    cyCodeBase/cyCore.h
    cyCodeBase/cyPolynomial.h
    model/affine_body.h
    model/affine_obj.h
    model/cube.h
    model/FrictionUtils.h
    model/glue.h
    model/iaabb.h
    # model/IpcCollisionConstraint.h
    # model/IpcFrictionConstraint.h
    model/psd.h
    model/othogonal_energy.h
    model/time_integrator.h
    model/barrier.h
    model/collision.h
    model/spatial_hashing.h
    model/geometry.h
    model/ipc_extension.h
    test_cases/tests.h
    view/camera.h
    view/env.h
    view/global_variables.h
    view/light.h
    view/mesh.h
    view/model.h
    view/shader.h
    view/stb_image.h
)

set(SOURCE 
    model/affine_obj.cpp
    model/col_set.cpp
    model/iaabb.cpp
    # model/IpcCollisionConstraint.cpp
    # model/IpcFrictionConstraint.cpp
    model/othogonal_energy.cpp
    model/barrier.cpp
    model/euler.cpp
    model/collision.cpp
    model/sparse.cpp
    model/spatial_hashing.cpp
    model/geometry.cpp
    model/mutils.cpp
    model/upper_bound.cpp
    model/energy.cpp
    model/core_ipc.cpp
    model/pt_ipc.cpp
    model/ee_ipc.cpp
    model/vg_ipc.cpp
    model/assemble_matrices.cpp
    model/friction.cpp
    model/ipc_extension.cpp
    test_cases/spinning_cube.cpp
    view/control.cpp
    view/light.cpp
    view/utils.cpp
    )

set(PYTHON_SOURCE 
    model/core_ipc.cpp
    model/barrier.cpp
    model/geometry.cpp
    model/collision.cpp
    model/ipc_extension.cpp
)

set(PYTHON_HEADERS
    model/psd.h
    model/affine_body.h
    model/barrier.h
    model/geometry.h
    model/collision.h
    model/ipc_extension.h
    cyCodeBase/cyCore.h
    cyCodeBase/cyPolynomial.h
    )
    
add_executable(abd view/model_view.cpp
    ${HEADERS}
    ${SOURCE}
    )

pybind11_add_module(abdtk ${PYTHON_HEADERS} ${PYTHON_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/python/bindings.cpp)
target_compile_definitions(abdtk PRIVATE ABDTK)
source_group(src FILES view/model_view.cpp ${SOURCE})
source_group(include FILES ${HEADERS})

target_include_directories(abd PRIVATE    
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cyCodeBase
    ${CMAKE_CURRENT_SOURCE_DIR}/iAABB
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases
    ${CMAKE_CURRENT_SOURCE_DIR}/view
)

target_include_directories(abdtk PRIVATE    
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cyCodeBase
    ${CMAKE_CURRENT_SOURCE_DIR}/iAABB
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases
    ${CMAKE_CURRENT_SOURCE_DIR}/view
    ${ipc_toolkit_SOURCE_DIR}/src
)

# Now, link the ipc_toolkit to your target (replace 'your_target' with your actual target)
target_include_directories(abd PRIVATE ${ipc_toolkit_SOURCE_DIR}/src)

target_link_libraries(abd PRIVATE ipc::toolkit glfw assimp::assimp glad::glad glm::glm Eigen3::Eigen nlohmann_json::nlohmann_json)
target_link_libraries(abdtk PRIVATE ipc::toolkit Eigen3::Eigen)
# if(MSVC) 
#     set_property(TARGET abd PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#     set_property(TARGET abd PROPERTY CUDA_ARCHITECTURES 86)
# endif()
