import playground;

[playground::KEY_INPUT("Space")]
uniform float spacePressed;
[playground::KEY_INPUT("W")]
uniform float wPressed;
[playground::KEY_INPUT("S")]
uniform float sPressed;
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<float4> outputTexture;

[shader("compute")]
[numthreads(16, 16, 1)]
[playground::CALL(1, 1, 1)]
void keyDemo(uint3 threadId: SV_DispatchThreadID)
{
    uint2 pixel = threadId.xy;
    uint2 dimensions;
    outputTexture.GetDimensions(dimensions.x, dimensions.y);

    if (pixel.x >= dimensions.x || pixel.y >= dimensions.y)
        return;

    // Change color based on key presses
    float4 color = float4(0, 0, 0, 1);

    // Space key controls blue channel
    color.b = spacePressed;

    // W key controls red channel
    color.r = wPressed;

    // S key controls green channel
    color.g = sPressed;

    outputTexture[pixel] = color;
    print("Key States - Space: %f, W: %f, S: %f\n", spacePressed, wPressed, sPressed);
}

void printMain()
{
}
