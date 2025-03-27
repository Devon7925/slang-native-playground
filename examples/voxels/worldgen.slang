import playground;
import common;
import binding;

module worldgen;

static const float SEA_LEVEL = 16.0;
static const float3 MACRO_SCALE = float3(0.01, 0.00001, 0.01);
static const float3 DENSITY_SCALE = float3(0.04, 0.0, 0.04);
static const float3 TEMP_SCALE = float3(0.005, 0.0005, 0.005);
static const float3 CAVE_SCALE = float3(0.06, 0.035, 0.06);

float4 y_gradient(float y, float scale, float start_y, float end_y)
{
    return float4(0.0, scale * (step(start_y, scale * y) - step(end_y, scale * y)), 0.0, clamp(scale * y, start_y, end_y));
}

float4 clamp_gradient(float4 gradient, float min_val, float max_val)
{
    return float4(gradient.xyz * (step(min_val, gradient.w) - step(max_val, gradient.w)), clamp(gradient.w, min_val, max_val));
}

float4 abs_gradient(float4 gradient)
{
    return float4(gradient.xyz * sign(gradient.w), abs(gradient.w));
}

float4 max_gradient(float4 a, float4 b)
{
    if (a.w > b.w)
    {
        return a;
    }
    else
    {
        return b;
    }
}

float4 mul_gradient(float4 a, float4 b)
{
    return float4(a.xyz * b.w + a.w * b.xyz, a.w * b.w);
}

uint get_worldgen(int3 global_pos)
{
    int2 chunk_pos = floorDiv(global_pos.xz, 32);
    float3 tree_pos = 32.0 * float3(chunk_pos, 0.0) + 3.0 + 26.0 * float3(pcg3d(chunk_pos.xxy).xyz & 0xFF) / 256.0;
    tree_pos.z = tree_pos.z / 100.0 + 0.2;
    float3 random = float3(pcg3d(global_pos) & 0xFF) / 128.0;
    float3 true_pos = float3(global_pos);
    float4 macro_noise = float4(MACRO_SCALE, 1.0) * voronoise(MACRO_SCALE * true_pos, 1.0, 1.0);
    float4 density = float4(DENSITY_SCALE, 1.0) * voronoise(DENSITY_SCALE * true_pos, 1.0, 1.0);
    float4 temperature = float4(float3(0.0), 0.2) + float4(TEMP_SCALE, 1.0) * voronoise(TEMP_SCALE * true_pos, 1.0, 1.0) - y_gradient(true_pos.y - SEA_LEVEL, 1.0 / 80.0, -0.5, 2.0);
    float4 cave_density = mul_gradient(
        clamp_gradient(float4(CAVE_SCALE, 1.0) * voronoise(CAVE_SCALE * true_pos, 1.0, 1.0), -1.0, 0.5),
        -1.0 * y_gradient(true_pos.y - SEA_LEVEL, 1.0 / 80.0, -1.0, 0.3));

    float4 terrain_density =
        1.5 * cave_density + density + 0.7 * macro_noise + float4(float3(0.0), 0.1) - y_gradient(true_pos.y - SEA_LEVEL, 1.0 / 62.0, -0.6, 10.0) + mul_gradient(max_gradient(float4(float3(0.0), 1.0) - 20.0 * abs_gradient(temperature - float4(float3(0.0), 0.1)), float4(0.0)), -y_gradient(true_pos.y - SEA_LEVEL - 5.0, 0.3, -2.0, 0.0));
    if (terrain_density.w <= 0.0)
    {
        if (temperature.w + tree_pos.z > 0.2 && length(true_pos.xz - tree_pos.xy) < 0.7 + 0.2 / (temperature.w + tree_pos.z - 0.1) - 0.25 / (terrain_density.w - 0.2))
        {
            return MAT_WOOD << 24;
        }
        else if (length(float3(true_pos.xz - tree_pos.xy, 100.0 * (temperature.w + tree_pos.z - 0.2))) < 12.0 - 5.0 * random.x)
        {
            return MAT_LEAF << 24;
        }
        else if (true_pos[1] > SEA_LEVEL)
        {
            if (temperature.w - terrain_density.w < -0.9)
            {
                return MAT_ICE << 24;
            }
            return MAT_AIR << 24;
        }
        else if (temperature.w + 0.02 * random.x > 0.1)
        {
            return MAT_WATER << 24;
        }
        else if (terrain_density.w + clamp((true_pos.y - SEA_LEVEL) / 30.0, -2.0, 0.0) < -0.27 && temperature.w < 0.1)
        {
            return MAT_AIR << 24;
        }
        else
        {
            return MAT_ICE << 24;
        }
    }
    else if (terrain_density.w + clamp((SEA_LEVEL - true_pos.y) / 30.0, 0.0, 0.7) > 0.35)
    {
        return MAT_STONE << 24;
    }
    else if (terrain_density.w > 0.2)
    {
        return MAT_DIRT << 24;
    }
    else
    {
        if (temperature.w + 0.05 * random.x - 1.5 * terrain_density.y > terrain_density.w && true_pos[1] >= SEA_LEVEL && terrain_density.y < 0.01)
        {
            return MAT_GRASS << 24;
        }
        else
        {
            return MAT_DIRT << 24;
        }
    }
    return MAT_AIR << 24;
}

[numthreads(1, 1, 1)]
[playground::CALL_INDIRECT("indirectBuffer", 32)]
void worldgen_setup(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    let chunk_location = chunkLoads[dispatchThreadId.x].xyz;
    if (!chunk_in_bounds(chunk_location))
    {
        return;
    }
    let chunk_ptr = floorMod(chunk_location, RENDER_SIZE);
    chunks.Store(chunk_ptr, chunkLoads[dispatchThreadId.x].w);

    let chunk_idx = 8 * RENDER_SIZE_X * RENDER_SIZE_Y * RENDER_SIZE_Z + chunk_ptr.x + chunk_ptr.y * RENDER_SIZE_X + chunk_ptr.z * RENDER_SIZE_X * RENDER_SIZE_Y;
    updatedChunks[chunk_idx].store(0);
}

[numthreads(4, 4, 4)]
[playground::CALL_INDIRECT("indirectBuffer", 16)]
void worldgen(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    int3 pos = dispatchThreadId;
    pos.x = pos.x % CHUNK_SIZE;
    let idx = dispatchThreadId.x / CHUNK_SIZE;
    let chunk_location = chunkLoads[idx].xyz;
    if (!chunk_in_bounds(chunk_location))
    {
        return;
    }
    pos += CHUNK_SIZE * chunk_location;

    let chunk_idx = chunkLoads[idx].w;
    let chunk_loc = int3(chunk_idx % MAX_CHUNKS_X, (chunk_idx / MAX_CHUNKS_X) % MAX_CHUNKS_Y, chunk_idx / MAX_CHUNKS_X / MAX_CHUNKS_Y);
    uint data = get_worldgen(pos);
    voxels.Store(CHUNK_SIZE * chunk_loc + floorMod(pos, CHUNK_SIZE), data);

    if (all(floorMod(pos, CHUNK_SIZE) == 0))
    {
        queue_update_all(chunk_location);
    }
}
