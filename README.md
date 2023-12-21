# Multi-Affine Body Dynamics 
- Reproduce Affine Body Dynamics

### INSTALL

*Windows*
Make sure you have cmake and vcpkg installed. Detailed instructions can be found [here](https://vcpkg.io/en/getting-started.html).

Navigate to the vcpkg directory, and installed the following packages via vcpkg
`vcpkg install glfw3 eigen3 glm glad assimp nlohmann-json --triplet=x64-windows`

Navigate to the root directory glad_framework, and type in

```
cmake -B "build" -S . -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config=Release
```

### Running

The main config file is config.json under the top folder.

The test cases are under test_cases folder. You can change line 10 of config.json to switch the test cases.

For example, to turn off friction, you can change line 52-54 to 
```json
    "vg_friction": false,
    "pt_friction": false,
    "ee_friction": false,
```

The chains.json case requires turning off the ground plane 
```json
    "ground": false,
``` 


The results will be saved to "trace_folder" specified in config.json. It saves 500 frames by default (controlled by "ending_ts" parameter).

You can then view your saved results by changing the "player" option to true.

### Keymap

`w`,`a`,`s`,`d`: navigates around the scene
`r`: restarts the simulation
