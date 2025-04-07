use slang_playground_compiler::CompilationResult;
use slang_native_playground::launch;
use slang_shader_macros::compile_shader;

fn main() {
    let compilation: CompilationResult = compile_shader!("free_flight_camera.slang", ["examples"]);
    launch(compilation);
}