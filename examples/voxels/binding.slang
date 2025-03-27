import playground;
import common;

[playground::TIME]
float time;
[playground::MOUSE_POSITION]
float4 mousePosition;

static const int RENDER_SIZE_X = 16;
static const int RENDER_SIZE_Y = 8;
static const int RENDER_SIZE_Z = 16;
static const int MAX_CHUNKS = 2048;

static const int MAX_CHUNK_LOADS = 128;
static const int MAX_CHUNK_UPDATES = 256;

static const int RESERVED_CHUNKS = 1;
static const int MAX_DIMENSION = 2048;
static const int MAX_DIMENSION_CHUNKS = MAX_DIMENSION / CHUNK_SIZE;
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

[playground::ZEROS(9 * RENDER_SIZE_X * RENDER_SIZE_Y * RENDER_SIZE_Z)]
RWStructuredBuffer<Atomic<uint>> updatedChunks;

[playground::ZEROS(8 * MAX_CHUNK_UPDATES)]
RWStructuredBuffer<int3> chunkUpdates;

struct GameData
{
    float3 playerPos;
    int voxelOffset;
    int3 startPos;
    Atomic<int> chunksAvailable;
    Atomic<int> currentChunkLoads;
    Atomic<int>[8] currentChunkUpdates;
}

[playground::ZEROS(1)]
RWStructuredBuffer<GameData> gameData;

[playground::ZEROS(12)]
RWStructuredBuffer<uint> indirectBuffer;

[playground::CALL(MAX_CHUNKS, 1, 1)]
[playground::CALL_ONCE]
[shader("compute")]
[numthreads(16, 1, 1)]
void init_available(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= MAX_CHUNKS - RESERVED_CHUNKS)
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
    let chunk_ptr = floorMod(chunk_pos, RENDER_SIZE);
    let chunk_idx = chunks.Load(chunk_ptr);
    if (chunk_idx == 0)
        return;
    chunks.Store(chunk_ptr, 0);
    let availableChunkIdx = gameData[0].chunksAvailable.increment();
    availableChunks[availableChunkIdx] = chunk_idx;

    for (var i = 0; i < 9; i++)
    {
        let chunk_idx = i * RENDER_SIZE_X * RENDER_SIZE_Y * RENDER_SIZE_Z + chunk_ptr.x + chunk_ptr.y * RENDER_SIZE_X + chunk_ptr.z * RENDER_SIZE_X * RENDER_SIZE_Y;
        updatedChunks[chunk_idx].exchange(0);
    }
}

bool chunk_in_bounds(int3 chunk_pos)
{
    if (any(chunk_pos < gameData[0].startPos))
        return false;
    if (any(chunk_pos >= gameData[0].startPos + RENDER_SIZE))
        return false;

    return true;
}

void loadChunk(int3 chunk_pos)
{
    if (!chunk_in_bounds(chunk_pos))
        return;
    let checkChunkIdx = floorMod(chunk_pos, RENDER_SIZE);
    let chunk_idx = 8 * RENDER_SIZE_X * RENDER_SIZE_Y * RENDER_SIZE_Z + checkChunkIdx.x + checkChunkIdx.y * RENDER_SIZE_X + checkChunkIdx.z * RENDER_SIZE_X * RENDER_SIZE_Y;
    let updateCheckIdx = gameData[0].currentChunkLoads.load();
    if (updateCheckIdx >= MAX_CHUNK_LOADS)
    {
        return;
    }
    let was_already_queued = updatedChunks[chunk_idx].exchange(1);
    if (was_already_queued == 1)
    {
        return;
    }
    let availableChunk = gameData[0].chunksAvailable.sub(1);
    let chunk = availableChunks[availableChunk - 1];
    let chunkLoadIdx = gameData[0].currentChunkLoads.increment();
    if (chunkLoadIdx >= MAX_CHUNK_LOADS)
    {
        updatedChunks[chunk_idx].store(0);
        let availableChunk = gameData[0].chunksAvailable.increment();
        availableChunks[availableChunk] = chunk;
        return;
    }
    chunkLoads[chunkLoadIdx] = int4(chunk_pos, chunk);
}

void queue_update_all(int3 chunk_pos)
{
    if (!chunk_in_bounds(chunk_pos))
        return;
    let checkChunkIdx = floorMod(chunk_pos, RENDER_SIZE);
    if (chunks.Load(checkChunkIdx) == 0)
    {
        loadChunk(chunk_pos);
        return;
    }
    let chunk_ptr = floorMod(chunk_pos, RENDER_SIZE);
    for (var i = 0; i < 8; i++)
    {
        let chunk_idx = i * RENDER_SIZE_X * RENDER_SIZE_Y * RENDER_SIZE_Z + chunk_ptr.x + chunk_ptr.y * RENDER_SIZE_X + chunk_ptr.z * RENDER_SIZE_X * RENDER_SIZE_Y;
        let updateCheckIdx = gameData[0].currentChunkUpdates[i].load();
        if (updateCheckIdx >= MAX_CHUNK_UPDATES)
        {
            continue;
        }
        let was_already_queued = updatedChunks[chunk_idx].exchange(1);
        if (was_already_queued == 1)
        {
            continue;
        }
        let updateIdx = gameData[0].currentChunkUpdates[i].increment();
        if (updateIdx >= MAX_CHUNK_UPDATES)
        {
            updatedChunks[chunk_idx].exchange(0);
            continue;
        }
        chunkUpdates[updateIdx + i * MAX_CHUNK_UPDATES] = chunk_pos;
    }
}

[playground::CALL_ONCE]
[playground::CALL(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void setup(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gameData[0].playerPos = float3(10.0, 100.0, 10.0);

    uint3 voxelSize;
    voxels.GetDimensions(voxelSize.x, voxelSize.y, voxelSize.z);
    indirectBuffer[0] = 0;
    indirectBuffer[1] = CHUNK_SIZE / VOXEL_UPDATE_SIZE / 2;
    indirectBuffer[2] = CHUNK_SIZE / VOXEL_UPDATE_SIZE / 2;

    indirectBuffer[4] = 0;
    indirectBuffer[5] = CHUNK_SIZE / 4;
    indirectBuffer[6] = CHUNK_SIZE / 4;

    indirectBuffer[8] = 0;
    indirectBuffer[9] = 1;
    indirectBuffer[10] = 1;

    gameData[0].startPos = int3(0);
    gameData[0].chunksAvailable.store(MAX_CHUNKS - RESERVED_CHUNKS);
    gameData[0].currentChunkLoads.store(0);
    for (var i = 0; i < 8; i++)
    {
        gameData[0].currentChunkUpdates[i].store(0);
    }
    gameData[0].voxelOffset = 0;
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
    float2 cameraControl = -3.141592 / 2.0 * (2.0 * mousePosition.xy / imageSize - 1.0);
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
    if (og_voxel_data != data)
    {
        let chunk_pos = floorDiv(global_pos, CHUNK_SIZE);
        let pos_in_chunk = floorMod(global_pos, CHUNK_SIZE);
        let min_pos = select(pos_in_chunk == 0, -1, 0);
        let max_pos = select(pos_in_chunk == CHUNK_SIZE - 1, 1, 0);
        for (var dx = min_pos.x; dx <= max_pos.x; dx++)
            for (var dy = min_pos.y; dy <= max_pos.y; dy++)
                for (var dz = min_pos.z; dz <= max_pos.z; dz++)
                    queue_update_all(chunk_pos + int3(dx, dy, dz));
    }
    voxels.Store(get_index(global_pos), data);
}
