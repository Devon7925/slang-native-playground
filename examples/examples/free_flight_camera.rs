use slang_playground_compiler::CompilationResult;
use examples::launch;
use slang_shader_macros::compile_shader;

fn main() {
    let compilation: CompilationResult = compile_shader!("free_flight_camera.slang", ["examples/examples"]);
    launch(compilation);
}