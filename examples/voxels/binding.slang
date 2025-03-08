import playground;
import common;

static const int RENDER_SIZE_X = 16;
static const int RENDER_SIZE_Y = 8;
static const int RENDER_SIZE_Z = 16;
static const int MAX_CHUNKS = 2048;

static const int MAX_CHUNK_LOADS = 128;

static const int MAX_DIMENTION = 2048;
static const int MAX_DIMENSION_CHUNKS = MAX_DIMENTION / CHUNK_SIZE;
static const int MAX_CHUNKS_X = MAX_CHUNKS < MAX_DIMENSION_CHUNKS ? MAX_CHUNKS : MAX_DIMENSION_CHUNKS;
static const int X_CHUNK_LAYERS = (MAX_CHUNKS + MAX_CHUNKS_X - 1) / MAX_CHUNKS_X;
static const int MAX_CHUNKS_Y = X_CHUNK_LAYERS < MAX_DIMENSION_CHUNKS ? X_CHUNK_LAYERS : MAX_DIMENSION_CHUNKS;
static const int XY_CHUNK_LAYERS = (MAX_CHUNKS + MAX_CHUNKS_X * MAX_CHUNKS_Y - 1) / MAX_CHUNKS_X / MAX_CHUNKS_Y;
static const int MAX_CHUNKS_Z = XY_CHUNK_LAYERS < MAX_DIMENSION_CHUNKS ? XY_CHUNK_LAYERS : MAX_DIMENSION_CHUNKS;

static const int3 RENDER_SIZE = int3(RENDER_SIZE_X, RENDER_SIZE_Y, RENDER_SIZE_Z);

[playground::BLACK_3D(RENDER_SIZE_X, RENDER_SIZE_Y, RENDER_SIZE_Z)]
RWTexture3D<uint> chunks;

[playground::BLACK_3D(CHUNK_SIZE * MAX_CHUNKS_X, CHUNK_SIZE *MAX_CHUNKS_Y, CHUNK_SIZE *MAX_CHUNKS_Z)]
RWTexture3D<uint> voxels;

[playground::ZEROS(MAX_CHUNKS)]
RWStructuredBuffer<int> availableChunks;

[playground::ZEROS(MAX_CHUNK_LOADS)]
RWStructuredBuffer<int4> chunkLoads;

[playground::CALL(MAX_CHUNKS, 1, 1)]
[playground::CALL_ONCE]
[shader("compute")]
[numthreads(16, 1, 1)]
void init_available(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= MAX_CHUNKS)
        return;
    availableChunks[dispatchThreadId.x] = dispatchThreadId.x + 1;
}

int3 get_index(int3 global_pos)
{
    let chunk_pos = floorDiv(global_pos, CHUNK_SIZE);
    let chunk_idx = chunks.Load(floorMod(chunk_pos, RENDER_SIZE));
    let chunk_loc = int3(chunk_idx % MAX_CHUNKS_X, (chunk_idx / MAX_CHUNKS_X) % MAX_CHUNKS_Y, chunk_idx / MAX_CHUNKS_X / MAX_CHUNKS_Y);
    return CHUNK_SIZE * chunk_loc + floorMod(global_pos, CHUNK_SIZE);
}

void unload_chunk(int3 chunk_pos)
{
    let chunk_idx = chunks.Load(floorMod(chunk_pos, RENDER_SIZE));
    if (chunk_idx == 0)
        return;
    let chunk_loc = int3(chunk_idx % MAX_CHUNKS_X, (chunk_idx / MAX_CHUNKS_X) % MAX_CHUNKS_Y, chunk_idx / MAX_CHUNKS_X / MAX_CHUNKS_Y);
    chunks.Store(floorMod(chunk_pos, RENDER_SIZE), 0);
    availableChunks[gameData[0].chunksAvailable] = chunk_idx;
    gameData[0].chunksAvailable++;
}

struct GameData
{
    float3 playerPos;
    int voxelOffset;
    int3 startPos;
    int chunksAvailable;
    int currentChunkLoads;
}

[playground::ZEROS(1)]
RWStructuredBuffer<GameData> gameData;

[playground::ZEROS(8)]
RWStructuredBuffer<uint> indirectBuffer;

[playground::CALL_ONCE]
[playground::CALL(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void setup(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gameData[0].playerPos = float3(10.0, 100.0, 10.0);

    uint3 voxelSize;
    voxels.GetDimensions(voxelSize.x, voxelSize.y, voxelSize.z);
    indirectBuffer[0] = voxelSize.x / VOXEL_UPDATE_SIZE / 2;
    indirectBuffer[1] = voxelSize.y / VOXEL_UPDATE_SIZE / 2;
    indirectBuffer[2] = voxelSize.z / VOXEL_UPDATE_SIZE / 2;

    indirectBuffer[4] = 0;
    indirectBuffer[5] = CHUNK_SIZE / 4;
    indirectBuffer[6] = CHUNK_SIZE / 4;

    gameData[0].startPos = int3(0);
    gameData[0].chunksAvailable = MAX_CHUNKS;
    gameData[0].currentChunkLoads = 0;
}

[playground::CALL_ONCE]
[playground::CALL(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)]
[shader("compute")]
[numthreads(4, 4, 4)]
void setUnloadedChunk(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    voxels.Store(dispatchThreadId, MAT_OOB << 24);
}

float3 getCameraDirection(float2 imageSize)
{
    float2 cameraControl = -3.141592 / 2.0 * (2.0 * getMousePosition().xy / imageSize - 1.0);
    return float3(sin(cameraControl.x) * cos(cameraControl.y), sin(cameraControl.y), cos(cameraControl.x) * cos(cameraControl.y));
}

int3 get_voxel_update_offset()
{
    let idx = int(gameData[0].voxelOffset);
    return int3(idx & 1, (idx >> 1) & 1, (idx >> 2) & 1);
}

uint get_data_unchecked(int3 global_pos)
{
    return voxels.Load(get_index(global_pos));
}

uint get_data(int3 global_pos)
{
    int3 start_offset = CHUNK_SIZE * gameData[0].startPos;
    if (any(global_pos < start_offset))
        return MAT_OOB << 24;
    int3 rel_pos = global_pos - start_offset;
    if (any(rel_pos >= CHUNK_SIZE * RENDER_SIZE))
        return MAT_OOB << 24;
    return get_data_unchecked(global_pos);
}

void set_data(int3 global_pos, uint data)
{
    uint og_voxel_data = get_data(global_pos);
    if (og_voxel_data >> 24 == MAT_OOB)
        return;
    if (og_voxel_data >> 24 == MAT_AIR_OOB)
        return;
    voxels.Store(get_index(global_pos), data);
}
