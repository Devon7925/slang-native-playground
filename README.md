# Slang Native Playground

A project to allow Slang playground shaders to run outside of a web environment. Your user shader can be set in `shaders/user.slang`. It bundles shaders in compilation for easy deployment/sharing.

Use should be as easy as cloning and running `cargo run`.

Depends on [slang-rs](https://github.com/FloatyMonkey/slang-rs). See that repo if it is having trouble recognizing your version of Slang.

## Default Shaders status

| Name | Status | Reason |
|------|--------|--------|
| Simple Print | ✅ | |
| Simple Image | ✅ | |
| Image From URL | ✅ | |
| Multi-kernel Demo | ✅ | |
| | | |
| ShaderToy: Circle | ✅ | |
| ShaderToy: Ocean | ✅ | |
| 2D Splatter | ✅ | |
| Differentiable 2D Splatter | ❌ | Blocked on atomic reflection not working in slang. See https://github.com/shader-slang/slang/issues/6257. |
| | | |
| Properties | ✅ | |
| Generics & Extensions | ✅ | |
| Operator Overload | ✅ | |
| Variadic generics | ✅ | |
| Automatic Differentiation | ✅ | |
| Graphics Entrypoints | ❌ | Compilation only support not planned |
| Atomics | ✅ | |





## Extra features

This app also has a number of additional features that are not(yet) in the official playground.

### `[playground::DELTA_TIME]`
Controls a float uniform to give the difference in time between frames. Useful for making things not break at high framerates.

### `playground::BLACK_SCREEN`

Acts the same as `BLACK` but is automatically scaled based on screen size.

### `playground::BLACK_3D`

Acts the same as `BLACK` but for 3d textures.

### Draw features

* `playground::DRAW` on a vertex shader allows creating a fixed size draw that will run every frame
* `playground::REBIND_FOR_DRAW` on a texture or buffer allows specifying a resource that will mirror another resource so it becomes accessible in a graphics context. Generally neccesary to make writable resources available.
* `playground::SAMPLER` allows a sampler to be used

### `playground::CALL_INDIRECT`

Takes a buffer and a offset and makes an indirect dispatch using them. The buffer is not accessible from the shader being called.

### `playground::KEY_INPUT`

Allows to control a uniform float based on a key with `1.0` meaning pressed and `0.0` meaning released.

Example:
```slang
[playground::KEY_INPUT("Space")]
uniform float spacePressed;
[playground::KEY_INPUT("W")] 
uniform float wPressed;
```

### `playground::MODEL`

Allows loading *.obj files into a buffer. Element type must be a struct with only `position`, `normal`, and `uv` fields.

Example:

```slang
[playground::MODEL("static/teapot.obj")]
StructuredBuffer<Vertex> verticies;

struct Vertex
{
    float3 position;
    float3 normal;
}
```

### Github Imports

Allows importing slang files directly from github using `github://user_name/repo_name/path/file.slang`.

Example:

```slang
import "github://Devon7925/slang-native-playground/examples/free_flight_camera.slang";
```

## Examples

There are examples of use of the extra features in the `examples` folder. For single files, replace `user.slang` with their contents. For folders, replace the whole contents of `shaders/` with their contents.
| Name | Description |
|------|--------|
| `cube_raster.slang` | Textured cube rasterizer with basic lighting |
| `free_flight_camera.slang` | A rasterizer of the Utah Teacup with keyboard control for the camera |
| `painting.slang` | A simple painting app demonstrating storage textures and indirect dispatch |
| `voxels/` | A voxel rendering engine with simple falling sand physics |

## Web Build

This app supports building for web. To do so run:

```bash
cargo build --target wasm32-unknown-unknown
wasm-bindgen --out-dir target/generated/ --web target/wasm32-unknown-unknown/debug/slang-native-playground.wasm 
```

The web build should then be accessible from `index.html`.


