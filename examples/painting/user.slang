import playground;

static const int MAX_BRUSH_SIZE = 64;
static const float FLOAT_MAX_BRUSH_SIZE = 64.0;

[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<float> tex_red;
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<float> tex_green;
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<float> tex_blue;

[playground::ZEROS(8)]
RWStructuredBuffer<uint> indirectBuffer;

static const uint thread_count = 8;

[playground::MOUSE_POSITION]
float4 mousePosition;
[playground::SLIDER(10.0, 4.0, 64.0)]
uniform float brush_size;
[playground::COLOR_PICK(1.0, 0.0, 1.0)]
uniform float3 color;

[shader("compute")]
[numthreads(thread_count, thread_count, 1)]
[playground::CALL_INDIRECT("indirectBuffer", 0)]
void draw(uint2 dispatchThreadId: SV_DispatchThreadID)
{
    if (mousePosition.z >= 0)
        return;

    let offset = float2(dispatchThreadId.xy) - brush_size;
    if (length(offset) > brush_size / 2)
        return;

    var mouse_pos = uint2(mousePosition.xy + offset);
    tex_red[mouse_pos] = color.r;
    tex_green[mouse_pos] = color.g;
    tex_blue[mouse_pos] = color.b;
}

[shader("compute")]
[numthreads(1, 1, 1)]
[playground::CALL(1, 1, 1)]
void update(uint2 dispatchThreadId: SV_DispatchThreadID)
{
    indirectBuffer[0] = uint(2.0 * brush_size + thread_count + 1.0) / thread_count;
    indirectBuffer[1] = uint(2.0 * brush_size + thread_count + 1.0) / thread_count;
    indirectBuffer[2] = 1;
}

float4 imageMain(uint2 dispatchThreadID, int2 screenSize)
{
    uint imageW;
    uint imageH;
    tex_red.GetDimensions(imageW, imageH);

    uint2 scaled = (uint2)floor(float2(dispatchThreadID.xy));
    uint2 flipped = uint2(scaled.x, imageH - scaled.y);

    float4 imageColor = float4(tex_red[flipped], tex_green[flipped], tex_blue[flipped], 1.0);
    return imageColor;
}
