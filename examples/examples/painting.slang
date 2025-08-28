import playground;
import rendering;

static const int MAX_BRUSH_SIZE = 64;
static const float FLOAT_MAX_BRUSH_SIZE = 64.0;

[playground::ZEROS(8)]
RWStructuredBuffer<uint> indirectBuffer;

static const uint thread_count = 8;

[playground::MOUSE_POSITION]
float4 mousePosition;
[playground::SLIDER(10.0, 4.0, 64.0)]
uniform float brush_size;
[playground::COLOR_PICK(1.0, 0.0, 1.0)]
uniform float3 color;
[playground::SCREEN_SIZE]
uniform int2 screenSize;

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

    var mouse_pos = int2(0, screenSize.y) + int2(1, -1) * int2(mousePosition.xy + offset);
    drawPixel(mouse_pos, () => float4(color, 1.0));
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