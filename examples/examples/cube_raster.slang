import playground;

[playground::MOUSE_POSITION]
float4 mousePosition;

[format("rgba8")]
[playground::BLACK_SCREEN(1.0, 1.0)]
WTexture2D<float4> outputTexture;

[format("rgba8")]
[playground::REBIND_FOR_DRAW("outputTexture")]
Texture2D<float4> drawTexture;

[playground::URL("static/jeep.jpg")]
Texture2D<float4> cubeTexture;

[playground::SAMPLER]
SamplerState sampler;

struct Vertex
{
    float3 position;
    float3 normal;
    float2 uv;
}

struct VOut
{
    float4 position : SV_Position;
    float3 originalPosition;
    float3 normal;
    float2 texcoord;
}

static const Vertex[] cube_verticies = {
    // +Z face
    Vertex(float3(-1.0, -1.0, 1.0), float3(0.0, 0.0, 1.0), float2(0.0, 0.0)),
    Vertex(float3(1.0, -1.0, 1.0), float3(0.0, 0.0, 1.0), float2(1.0, 0.0)),
    Vertex(float3(-1.0, 1.0, 1.0), float3(0.0, 0.0, 1.0), float2(0.0, 1.0)),
    Vertex(float3(-1.0, 1.0, 1.0), float3(0.0, 0.0, 1.0), float2(0.0, 1.0)),
    Vertex(float3(1.0, -1.0, 1.0), float3(0.0, 0.0, 1.0), float2(1.0, 0.0)),
    Vertex(float3(1.0, 1.0, 1.0), float3(0.0, 0.0, 1.0), float2(1.0, 1.0)),

    // -Z face
    Vertex(float3(-1.0, -1.0, -1.0), float3(0.0, 0.0, -1.0), float2(0.0, 0.0)),
    Vertex(float3(1.0, -1.0, -1.0), float3(0.0, 0.0, -1.0), float2(1.0, 0.0)),
    Vertex(float3(-1.0, 1.0, -1.0), float3(0.0, 0.0, -1.0), float2(0.0, 1.0)),
    Vertex(float3(-1.0, 1.0, -1.0), float3(0.0, 0.0, -1.0), float2(0.0, 1.0)),
    Vertex(float3(1.0, -1.0, -1.0), float3(0.0, 0.0, -1.0), float2(1.0, 0.0)),
    Vertex(float3(1.0, 1.0, -1.0), float3(0.0, 0.0, -1.0), float2(1.0, 1.0)),

    // +Y face
    Vertex(float3(-1.0, 1.0, -1.0), float3(0.0, 1.0, 0.0), float2(0.0, 0.0)),
    Vertex(float3(1.0, 1.0, -1.0), float3(0.0, 1.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(-1.0, 1.0, 1.0), float3(0.0, 1.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(-1.0, 1.0, 1.0), float3(0.0, 1.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(1.0, 1.0, -1.0), float3(0.0, 1.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(1.0, 1.0, 1.0), float3(0.0, 1.0, 0.0), float2(1.0, 1.0)),

    // -Y face
    Vertex(float3(-1.0, -1.0, -1.0), float3(0.0, -1.0, 0.0), float2(0.0, 0.0)),
    Vertex(float3(1.0, -1.0, -1.0), float3(0.0, -1.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(-1.0, -1.0, 1.0), float3(0.0, -1.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(-1.0, -1.0, 1.0), float3(0.0, -1.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(1.0, -1.0, -1.0), float3(0.0, -1.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(1.0, -1.0, 1.0), float3(0.0, -1.0, 0.0), float2(1.0, 1.0)),

    // +X face
    Vertex(float3(1.0, -1.0, -1.0), float3(1.0, 0.0, 0.0), float2(0.0, 0.0)),
    Vertex(float3(1.0, 1.0, -1.0), float3(1.0, 0.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(1.0, -1.0, 1.0), float3(1.0, 0.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(1.0, -1.0, 1.0), float3(1.0, 0.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(1.0, 1.0, -1.0), float3(1.0, 0.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(1.0, 1.0, 1.0), float3(1.0, 0.0, 0.0), float2(1.0, 1.0)),

    // -X face
    Vertex(float3(-1.0, -1.0, -1.0), float3(-1.0, 0.0, 0.0), float2(0.0, 0.0)),
    Vertex(float3(-1.0, 1.0, -1.0), float3(-1.0, 0.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(-1.0, -1.0, 1.0), float3(-1.0, 0.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(-1.0, -1.0, 1.0), float3(-1.0, 0.0, 0.0), float2(0.0, 1.0)),
    Vertex(float3(-1.0, 1.0, -1.0), float3(-1.0, 0.0, 0.0), float2(1.0, 0.0)),
    Vertex(float3(-1.0, 1.0, 1.0), float3(-1.0, 0.0, 0.0), float2(1.0, 1.0)),
};
// Function to create a view matrix from camera position, target position, and up vector
float4x4 CreateViewMatrix(float3 eye, float3 target, float3 up)
{
    // Calculate forward (z), right (x), and up (y) vectors
    float3 zaxis = normalize(eye - target);     // Forward vector (reversed from target direction)
    float3 xaxis = normalize(cross(up, zaxis)); // Right vector
    float3 yaxis = cross(zaxis, xaxis);         // Up vector

    // Construct the view matrix
    float4x4 viewMatrix = {
        xaxis.x, yaxis.x, zaxis.x, 0,
        xaxis.y, yaxis.y, zaxis.y, 0,
        xaxis.z, yaxis.z, zaxis.z, 0,
        -dot(xaxis, eye), -dot(yaxis, eye), -dot(zaxis, eye), 1
    };

    return viewMatrix;
}

// Function to create a perspective projection matrix
float4x4 CreatePerspectiveMatrix(float fovY, float aspectRatio, float nearZ, float farZ)
{
    // Calculate dimensions
    float yScale = 1.0f / tan(fovY * 0.5f);
    float xScale = yScale / aspectRatio;
    float zRange = farZ / (nearZ - farZ);

    // Construct the perspective projection matrix
    float4x4 projectionMatrix = {
        xScale, 0, 0, 0,
        0, yScale, 0, 0,
        0, 0, zRange, -1,
        0, 0, nearZ * zRange, 0
    };

    return projectionMatrix;
}

[playground::DRAW(36, "fragmentMain")]
[shader("vertex")]
VOut vertexMain(uint vertexID: SV_VertexID)
{
    uint imageW;
    uint imageH;
    drawTexture.GetDimensions(imageW, imageH);

    float aspectRatio = float(imageW) / float(imageH);

    float2 cameraControl = float2(1.0, -1.0) * -3.141592 / 2.0 * (2.0 * mousePosition.xy / float2(imageW, imageH) - 1.0);
    float3 cameraPos = float3(5.0 * sin(cameraControl.x) * cos(cameraControl.y), 5.0 * sin(cameraControl.y), 5.0 * cos(cameraControl.x) * cos(cameraControl.y));

    Vertex input = cube_verticies[vertexID];
    VOut output;
    float4x4 viewMatrix = CreateViewMatrix(cameraPos, float3(0, 0, 0), float3(0, 1, 0));
    float4x4 perspectiveMatrix = CreatePerspectiveMatrix(3.1415 / 2, aspectRatio, 0.1, 100.0);
    float4 viewPosition = mul(float4(input.position, 1.0), viewMatrix);

    output.position = mul(viewPosition, perspectiveMatrix);
    output.originalPosition = input.position;
    output.normal = input.normal;
    output.texcoord = input.uv;
    return output;
}

static const float3 lightPos = float3(2.0, 3.0, 4.0);

[shader("fragment")]
float4 fragmentMain(VOut input)
    : SV_Target
{
    float4 albedo = cubeTexture.Sample(sampler, input.texcoord);
    float3 diff = lightPos - input.originalPosition;
    float light = 25.0 / dot(diff, diff) * max(dot(normalize(diff), input.normal), 0.0);
    return (light + 0.1) * albedo;
}
