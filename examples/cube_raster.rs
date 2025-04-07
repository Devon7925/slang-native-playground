use slang_playground_compiler::CompilationResult;
use slang_native_playground::launch;
use slang_shader_macros::compile_shader;

fn main() {
    let compilation: CompilationResult = compile_shader!("cube_raster.slang", ["examples"]);
    launch(compilation);
}