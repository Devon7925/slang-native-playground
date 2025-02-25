import user;
import playground;

[format("rgba8")]
[playground::BLACK_SCREEN(1.0, 1.0)]
WTexture2D<float4> outputTexture;

[format("rgba8")]
[playground::REBIND_FOR_DRAW("outputTexture")]
Texture2D<float4> drawTexture;

[playground::SAMPLER]
SamplerState sampler;

[shader("compute")]
[numthreads(16, 16, 1)]
[playground::CALL::SIZE_OF("outputTexture")]
void imageMain(uint2 dispatchThreadID: SV_DispatchThreadID)
{
    uint width = 0;
    uint height = 0;
    outputTexture.GetDimensions(width, height);

    float4 color = imageMain(dispatchThreadID.xy, int2(width, height));

    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
        return;

    outputTexture.Store(dispatchThreadID.xy, color);
}

struct Vertex
{
    float2 position;
    float2 uv;
}

struct VOut
{
    float4 position : SV_Position;
    float2 texcoord;
}

const static Vertex[] verticies = {
    Vertex(float2(-1.0, -1.0), float2(0.0, 0.0)),
    Vertex(float2(1.0, -1.0), float2(1.0, 0.0)),
    Vertex(float2(-1.0, 1.0), float2(0.0, 1.0)),
    Vertex(float2(-1.0, 1.0), float2(0.0, 1.0)),
    Vertex(float2(1.0, -1.0), float2(1.0, 0.0)),
    Vertex(float2(1.0, 1.0), float2(1.0, 1.0)),
};

[playground::DRAW(6, "fragmentMain")]
[shader("vertex")]
VOut vertexMain(uint vertexID: SV_VertexID)
{
    Vertex input = verticies[vertexID];
    VOut output;
    output.position = float4(input.position, 0.0, 1.0);
    output.texcoord = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VOut input) : SV_Target
{
    let color = drawTexture.Sample(sampler, input.texcoord);
    return color;
}