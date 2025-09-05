use slang_playground_compiler::CompilationResult;
use examples::launch;
use slang_shader_macros::compile_shader;

#[cfg(target_family = "wasm")]
mod wasm_workaround {
    unsafe extern "C" {
        pub(super) fn __wasm_call_ctors();
    }
}

fn main() {
    let compilation: CompilationResult = compile_shader!("painting.slang", ["examples/examples"]);
    launch(compilation);
}