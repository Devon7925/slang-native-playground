import playground;

[playground::BLACK(200, 200)]
RWTexture2D<float> tex_red;
[playground::BLACK(200, 200)]
RWTexture2D<float> tex_green;
[playground::BLACK(200, 200)]
RWTexture2D<float> tex_blue;

[playground::SLIDER(100.0, 0.0, 200.0)]
uniform float draw_x;
[playground::SLIDER(100.0, 0.0, 200.0)]
uniform float draw_y;
[playground::COLOR_PICK(1.0, 0.0, 1.0)]
uniform float3 color;

// Fills buffer with a sine wave
[shader("compute")]
[numthreads(64, 1, 1)]
void draw(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    tex_red[uint2(uint(draw_x), uint(draw_y))] = color.r;
    tex_green[uint2(uint(draw_x), uint(draw_y))] = color.g;
    tex_blue[uint2(uint(draw_x), uint(draw_y))] = color.b;
}

//! CALL(draw, 1)
float4 imageMain(uint2 dispatchThreadID, int2 screenSize)
{
    uint imageW;
    uint imageH;
    tex_red.GetDimensions(imageW, imageH);

    uint2 scaled = (uint2)floor(float2(dispatchThreadID.xy) / screenSize.y * float2(imageW, imageH));
    uint2 flipped = uint2(scaled.x, imageH - scaled.y);

    float4 imageColor = float4(tex_red[flipped], tex_green[flipped], tex_blue[flipped], 1.0);
    return imageColor;
}