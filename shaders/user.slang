import playground;

[playground::BLACK(200, 200)]
RWTexture2D<float> tex_red;
[playground::BLACK(200, 200)]
RWTexture2D<float> tex_green;
[playground::BLACK(200, 200)]
RWTexture2D<float> tex_blue;

[playground::ZEROS(10)]
RWStructuredBuffer<float> data;

[playground::SLIDER(10.0, 0.0, 20.0)]
uniform float brush_size;
[playground::COLOR_PICK(1.0, 0.0, 1.0)]
uniform float3 color;

[shader("compute")]
[numthreads(16, 16, 1)]
void draw(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    // if(getMousePosition().z > 0) {
    //     return;
    // }
    let offset = dispatchThreadId.xy - uint2(brush_size / 2);
    if (dispatchThreadId.x >= brush_size || dispatchThreadId.y >= brush_size)
        return;
    var mouse_pos = uint2(getMousePosition().xy * 200/data[1]) + offset;
    tex_red[mouse_pos] = color.r;
    tex_green[mouse_pos] = color.g;
    tex_blue[mouse_pos] = color.b; 
}

//! CALL(draw, 1)
float4 imageMain(uint2 dispatchThreadID, int2 screenSize)
{
    data[0] = float(screenSize.x);
    data[1] = float(screenSize.y);
    uint imageW;
    uint imageH;
    tex_red.GetDimensions(imageW, imageH);

    uint2 scaled = (uint2)floor(float2(dispatchThreadID.xy) / screenSize.y * float2(imageW, imageH));
    uint2 flipped = uint2(scaled.x, imageH - scaled.y);

    float4 imageColor = float4(tex_red[flipped], tex_green[flipped], tex_blue[flipped], 1.0);
    return imageColor;
}