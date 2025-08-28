fn main() {
    // The macro should generate a `AnnotatedUniform` struct from the Slang file.
    // Instantiate it and print to ensure the generated type is available.
    let a = annotation_test::AnnotatedUniform { prop1: 0 };
    println!("Generated AnnotatedUniform: {:?}", a);
}

mod annotation_test {
    slang_shader_macros::shader_module!("annotation.slang", ["examples/examples"]);
}