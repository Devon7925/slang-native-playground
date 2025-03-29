# Slang Native Playground

A project to allow Slang playground shaders to run outside of a web environment. It bundles shaders in compilation for easy deployment/sharing.

This is a library. See [slang-native-playground-example](https://github.com/Devon7925/slang-native-playground-example) for usage. Note `imageMain` and `printMain` are only available from files named `user.slang`.

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
| Differentiable 2D Splatter | ✅ | |
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

If you have rate limit issues you can create a github personal access token and put it in a `GITHUB_TOKEN` file at the root of the repository.

## Examples

There are examples of use of the extra features in the `examples` folder. These can be run using `cargo run --example`. For example `cargo run --example cube_raster`.

| Name | Description |
|------|--------|
| `cube_raster` | Textured cube rasterizer with basic lighting |
| `free_flight_camera` | A rasterizer of the Utah Teacup with keyboard control for the camera |
| `painting` | A simple painting app demonstrating storage textures and indirect dispatch |
| `voxels` | A voxel rendering engine with simple falling sand physics |

