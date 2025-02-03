# Slang Native Playground

A project to allow Slang playground shaders to run outside of a web environment. 

## Default Shaders status

| Name | Status | Reason |
|------|--------|--------|
| All Print Shaders | ❌ | Print shaders not currently supported |
| Simple Image | ✅ | |
| Image From URL | ✅ | |
| Multi-kernel Demo | ✅ | |
| ShaderToy: Circle | ✅ | |
| ShaderToy: Ocean | ✅ | |
| 2D Splatter | ❌ | Currently blocked on https://github.com/gfx-rs/wgpu/issues/4704. |
| Differentiable 2D Splatter | ❌ | Blocked on atomic reflection not working in slang. See https://github.com/shader-slang/slang/issues/6257. |
| Graphics Entrypoints | ❌ | Compilation only support not planned |
| BLACK Commands | 📐 | Untested |
| Uniform Commands | ❌ | Unimplemented |




