import playground;

[format("rgba8")]
[playground::BLACK_SCREEN(1.0, 1.0)]
WTexture2D<float4> outputTexture;

[format("rgba8")]
[playground::REBIND_FOR_DRAW("outputTexture")]
Texture2D<float4> drawTexture;

[playground::ZEROS(8)]
RWStructuredBuffer<float> gameData;

[playground::REBIND_FOR_DRAW("gameData")]
StructuredBuffer<float> drawPlayerData;

[playground::KEY_INPUT("W")]
uniform float wPressed;
[playground::KEY_INPUT("A")]
uniform float aPressed;
[playground::KEY_INPUT("S")]
uniform float sPressed;
[playground::KEY_INPUT("D")]
uniform float dPressed;
[playground::KEY_INPUT("Space")]
uniform float spacePressed;
[playground::KEY_INPUT("Shift")]
uniform float shiftPressed;

[playground::SLIDER(0.5, 0.0, 2.0)]
uniform float speed;

[playground::MODEL("static/teapot.obj")]
StructuredBuffer<Vertex> verticies;

struct Vertex
{
    float3 position;
    float3 normal;
}

struct VOut
{
    float4 position : SV_Position;
    float3 originalPosition;
    float3 normal;
}

[shader("compute")]
[numthreads(1, 1, 1)]
[playground::CALL(1, 1, 1)]
void update(uint2 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 imageSize;
    outputTexture.GetDimensions(imageSize.x, imageSize.y);

    float3 cameraPos = float3(gameData[0], gameData[1], gameData[2]);

    float3 cameraDir = getCameraDirection(float2(imageSize));
    float3 up = float3(0.0, 1.0, 0.0);
    float3 left = normalize(cross(up, cameraDir));

    float3 inputVector = speed * float3(wPressed - sPressed, aPressed - dPressed, spacePressed - shiftPressed);
    float3x3 inputTransform = float3x3(cameraDir, left, up);

    cameraPos += mul(inputVector, inputTransform);

    gameData[0] = cameraPos.x;
    gameData[1] = cameraPos.y;
    gameData[2] = cameraPos.z;
}

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

float3 getCameraDirection(float2 imageSize)
{
    float2 cameraControl = -3.141592 / 2.0 * (2.0 * getMousePosition().xy / imageSize - 1.0);
    return float3(sin(cameraControl.x) * cos(cameraControl.y), sin(cameraControl.y), cos(cameraControl.x) * cos(cameraControl.y));
}

[playground::DRAW(18960, "fragmentMain")]
[shader("vertex")]
VOut vertexMain(uint vertexID: SV_VertexID)
{
    uint2 imageSize;
    drawTexture.GetDimensions(imageSize.x, imageSize.y);

    float aspectRatio = float(imageSize.x) / float(imageSize.y);

    float2 cameraControl = float2(1.0, -1.0) * -3.141592 / 2.0 * (2.0 * getMousePosition().xy / float2(imageSize) - 1.0);
    float3 cameraPos = float3(drawPlayerData[0], drawPlayerData[1], drawPlayerData[2]);

    Vertex input = verticies[vertexID];
    VOut output;
    float3 cameraDirection = getCameraDirection(float2(imageSize));
    float4x4 viewMatrix = CreateViewMatrix(cameraPos, cameraPos + cameraDirection, float3(0, 1, 0));
    float4x4 perspectiveMatrix = CreatePerspectiveMatrix(3.1415 / 2, aspectRatio, 0.1, 100.0);
    float4 viewPosition = mul(float4(input.position, 1.0), viewMatrix);

    output.position = mul(viewPosition, perspectiveMatrix);
    output.originalPosition = input.position;
    output.normal = input.normal;
    return output;
}

static const float3 lightPos = float3(2.0, 3.0, 4.0);

[shader("fragment")]
float4 fragmentMain(VOut input)
    : SV_Target
{
    float4 albedo = float4(1.0);
    float3 diff = lightPos - input.originalPosition;
    float light = 25.0 / dot(diff, diff) * max(dot(normalize(diff), input.normal), 0.0);
    return (light + 0.1) * albedo;
}
