# Multi-Affine Body Dynamics 
- Reproduce Affine Body Dynamics

### INSTALL

### Windows
Make sure you have cmake and vcpkg installed. Detailed instructions can be found [here](https://vcpkg.io/en/getting-started.html).

Navigate to the vcpkg directory, and installed the following packages via vcpkg
`vcpkg install glfw3 eigen3 glm glad assimp nlohmann-json --triplet=x64-windows`

Navigate to the root directory glad_framework, and type in

```
cmake -B "build" -S . -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build
```
