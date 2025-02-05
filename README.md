# Slang Native Playground

A project to allow Slang playground shaders to run outside of a web environment. 

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
| 2D Splatter | ❌ | Currently blocked on https://github.com/gfx-rs/wgpu/issues/4704. |
| Differentiable 2D Splatter | ❌ | Blocked on atomic reflection not working in slang. See https://github.com/shader-slang/slang/issues/6257. |
| | | |
| Properties | ✅ | |
| Generics & Extensions | ✅ | |
| Operator Overload | ✅ | |
| Variadic generics | ✅ | |
| Automatic Differentiation | ✅ | |
| Graphics Entrypoints | ❌ | Compilation only support not planned |
| Atomics | ✅ | |
| | | |
| BLACK Commands | 📐 | Untested |
| Uniform Commands | ❌ | Unimplemented |





