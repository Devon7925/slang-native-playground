import playground;

// A simple struct annotated with playground_ANNOTATION
[playground::ANNOTATION("doc = \"This is a test annotation for the Rust struct generated from Slang.\"")]
[playground::ANNOTATION("doc = \"This is a second test annotation for the Rust struct generated from Slang.\"")]
struct AnnotatedUniform
{
    uint prop1;
};

[playground::EXTERNAL(1)]
RWStructuredBuffer<AnnotatedUniform> outputBuffer;

// Minimal entry point referencing the struct so it shows up in reflection
void printMain()
{
    outputBuffer[0] = AnnotatedUniform(1);
}
