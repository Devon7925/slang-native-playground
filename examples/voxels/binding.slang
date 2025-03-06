import playground;
import common;

[playground::BLACK_3D(256, 128, 256)]
RWTexture3D<uint> voxels;

[playground::ZEROS(8)]
RWStructuredBuffer<float> playerData;

[playground::ZEROS(8)]
RWStructuredBuffer<uint> indirectBuffer;

float3 getCameraDirection(float2 imageSize)
{
    float2 cameraControl = -3.141592 / 2.0 * (2.0 * getMousePosition().xy / imageSize - 1.0);
    return float3(sin(cameraControl.x) * cos(cameraControl.y), sin(cameraControl.y), cos(cameraControl.x) * cos(cameraControl.y));
}

int3 get_voxel_update_offset()
{
    let idx = int(playerData[3]);
    return int3(idx & 1, (idx >> 1) & 1, (idx >> 2) & 1);
}

static const int3 start_pos = int3(0);
static const int3 render_size = int3(16, 8, 16);

int3 get_index(int3 global_pos)
{
    return global_pos;
}

uint get_data_unchecked(int3 global_pos)
{
    return voxels.Load(get_index(global_pos));
}

uint get_data(int3 global_pos)
{
    int3 start_offset = CHUNK_SIZE * start_pos;
    if (any(global_pos < start_offset))
        return MAT_OOB << 24;
    uint3 rel_pos = global_pos - start_offset;
    if (any(rel_pos >= CHUNK_SIZE * render_size))
        return MAT_OOB << 24;
    return get_data_unchecked(global_pos);
}
