use slang_compiler_type_definitions::CompilationResult;
use slang_native_playground::launch;
use slang_shader_macros::compile_shader;

fn main() {
    let compilation: CompilationResult = compile_shader!("cube_raster.slang", ["examples"]);
    launch(compilation);
}