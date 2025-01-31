struct VertexShaderOutput {
@builtin(position) position: vec4f,
@location(0) texcoord: vec2f,
};

@vertex fn vs(
@builtin(vertex_index) vertexIndex : u32
) -> VertexShaderOutput {
let uv = array(
    // 1st triangle
    vec2f( 0.0,  0.0),  // center
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 0.0,  1.0),  // center, top

    // 2st triangle
    vec2f( 0.0,  1.0),  // center, top
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 1.0,  1.0),  // right, top
);

let pos = array(
    // 1st triangle
    vec2f( -1.0,  -1.0),  // center
    vec2f( 1.0,  -1.0),  // right, center
    vec2f( -1.0,  1.0),  // center, top

    // 2st triangle
    vec2f( -1.0,  1.0),  // center, top
    vec2f( 1.0,  -1.0),  // right, center
    vec2f( 1.0,  1.0),  // right, top
);

var vsOutput: VertexShaderOutput;
let xy = pos[vertexIndex];
vsOutput.position = vec4f(xy, 0.0, 1.0);
vsOutput.texcoord = uv[vertexIndex];
return vsOutput;
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment fn fs(fsInput: VertexShaderOutput) -> @location(0) vec4f {
    let color = textureSample(ourTexture, ourSampler, fsInput.texcoord);
    return color;
}