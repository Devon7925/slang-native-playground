static const uint primary_ray_dist = 200;
static const uint shadow_ray_dist = 40;
static const uint reflection_ray_dist = 20;
static const uint ao_ray_dist = 15;

static const uint CHUNK_MAGNITUDE = 4;
static const uint CHUNK_SIZE = 1 << CHUNK_MAGNITUDE;
static const uint CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
static const uint POS_IN_CHUNK_MASK = CHUNK_SIZE - 1;

static const uint VOXEL_UPDATE_SIZE = 4;

static const float PI = 3.14159265359;

struct Projectile
{
    float4 pos;
    int4 chunk_update_pos;
    float4 dir;
    float4 size;
    float vel;
    float health;
    float lifetime;
    uint owner;
    float damage;
    uint proj_card_idx;
    uint wall_bounce;
    uint is_from_head;
    uint should_collide_with_terrain;
    uint _filler0;
    uint _filler1;
    uint _filler2;
};

struct Player
{
    float4 pos;
    float4 rot;
    float4 vel;
    float4 dir;
    float4 up;
    float4 right;
    int3 collision_vec;
    float size;
    float3 gravity;
    uint has_world_collision;
};

struct Collision
{
    uint id1;
    uint id2;
    uint properties;
    float when;
};

struct Hitsphere
{
    float3 offset;
    float radius;
    bool headshot;
};

static const Hitsphere HITSPHERES[] = {
    Hitsphere(float3(0.0, 0.0, 0.0), 0.6, true),
    Hitsphere(float3(0.0, -1.3, 0.0), 0.6, false),
    Hitsphere(float3(0.0, -1.9, 0.0), 0.9, false),
    Hitsphere(float3(0.0, -2.6, 0.0), 0.8, false),
    Hitsphere(float3(0.0, -3.3, 0.0), 0.6, false),
    Hitsphere(float3(0.0, -3.8, 0.0), 0.6, false),
};

static const float3 PLAYER_HITBOX_OFFSET = float3(0.0, -2.0, 0.0);
static const float3 PLAYER_HITBOX_SIZE = float3(1.8, 4.8, 1.8);
static const float PLAYER_DENSITY = 3.8;

static const uint MAT_AIR = 0;
static const uint MAT_STONE = 1;
static const uint MAT_OOB = 2;
static const uint MAT_DIRT = 3;
static const uint MAT_GRASS = 4;
static const uint MAT_PROJECTILE = 5;
static const uint MAT_ICE = 6;
static const uint MAT_WATER = 7;
static const uint MAT_PLAYER = 8;
static const uint MAT_AIR_OOB = 9;
static const uint MAT_WOOD = 10;
static const uint MAT_LEAF = 11;
static const uint MAT_UNBREAKABLE = 12;
static const uint[] material_damage_threshhold = { 0, 10, 0, 5, 5, 1, 5, 1, 0, 8, 5, 5 };

struct PhysicsProperties
{
    bool is_fluid;
    bool is_data_damage;
    bool is_data_standard_distance;
    float friction;
    float density;
};

static const PhysicsProperties physics_properties[] = {
    PhysicsProperties(true, false, true, 0.0, 1.0),   // MAT_AIR
    PhysicsProperties(false, true, false, 5.0, 1.0),  // MAT_STONE
    PhysicsProperties(false, false, false, 0.0, 1.0), // MAT_OOB
    PhysicsProperties(false, true, false, 5.0, 1.0),  // MAT_DIRT
    PhysicsProperties(false, true, false, 5.0, 1.0),  // MAT_GRASS
    PhysicsProperties(false, false, false, 0.0, 0.0), // MAT_PROJECTILE
    PhysicsProperties(false, true, false, 0.1, 1.0),  // MAT_ICE
    PhysicsProperties(true, false, true, 1.0, 4.4),   // MAT_WATER
    PhysicsProperties(false, false, false, 0.0, 0.0), // MAT_PLAYER
    PhysicsProperties(true, false, false, 0.0, 1.0),  // MAT_AIR_OOB
    PhysicsProperties(false, true, false, 5.0, 1.0),  // MAT_WOOD
    PhysicsProperties(false, false, true, 4.0, 1.0),  // MAT_LEAF
    PhysicsProperties(false, false, false, 5.0, 1.0), // MAT_UNBREAKABLE
};

static const bool is_transparent[] = {
    true,  // MAT_AIR
    false, // MAT_STONE
    false, // MAT_OOB
    false, // MAT_DIRT
    false, // MAT_GRASS
    false, // MAT_PROJECTILE
    true,  // MAT_ICE
    true,  // MAT_WATER
    false, // MAT_PLAYER
    true,  // MAT_AIR_OOB
    false, // MAT_WOOD
    true,  // MAT_LEAF
    false, // MAT_UNBREAKABLE
};

struct HeightData
{
    float offset;
    float3 scale;
    float impact;
    float3 movement;
};

static const HeightData height_data[] = {
    HeightData(0.0, float3(0.0), 0.0, float3(0.0)),               // MAT_AIR
    HeightData(0.2, float3(2.0), 0.75, float3(0.0)),              // MAT_STONE
    HeightData(0.0, float3(0.0), 0.0, float3(0.0)),               // MAT_OOB
    HeightData(0.4, float3(7.0), 0.3, float3(0.0)),               // MAT_DIRT
    HeightData(0.6, float3(35.0, 10.0, 35.0), 0.75, float3(0.0)), // MAT_GRASS
    HeightData(0.0, float3(0.0), 0.0, float3(0.0)),               // MAT_PROJECTILE
    HeightData(0.7, float3(1.7), 0.45, float3(0.0)),              // MAT_ICE
    HeightData(0.8, float3(1.0), 0.3, float3(0.25, -0.7, 0.2)),   // MAT_WATER
    HeightData(0.0, float3(0.0), 0.0, float3(0.0)),               // MAT_PLAYER
    HeightData(0.0, float3(0.0), 0.0, float3(0.0)),               // MAT_AIR_OOB
    HeightData(0.2, float3(10.0, 2.5, 10.0), 0.75, float3(0.0)),  // MAT_WOOD
    HeightData(0.7, float3(1.7), 0.45, float3(0.0)),              // MAT_LEAF
    HeightData(0.2, float3(2.0), 0.25, float3(0.0)),              // MAT_UNBREAKABLE
};

struct MaterialNoiseLayer
{
    float3 scale;
    float normal_impact;
    float roughness_impact;
    float transparency_impact;
    float3 movement;
    float3 layer_color;
};

struct MaterialRenderProps
{
    MaterialNoiseLayer layers[3];
    float ior;
    float roughness;
    float transparency;
    float depth_transparency;
    float3 color;
};

static const MaterialRenderProps material_render_props[] = {
    // AIR
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0))),
        0.0, 0.0, 1.0, 1.0, float3(1.0)),
    // STONE
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(2.0), 0.35, 0.1, 0.0, float3(0.0), float3(0.25)),
            MaterialNoiseLayer(float3(20.0), 0.2, 0.2, 0.0, float3(0.0), float3(0.05)),
            MaterialNoiseLayer(float3(0.5), 0.05, 0.0, 0.0, float3(0.0), float3(0.05))),
        0.04, 0.35, 0.0, 0.0, float3(0.55)),
    // OOB
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0))),
        0.0, 0.0, 1.0, 1.0, float3(1.0)),
    // DIRT
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(7.0), 0.2, -0.1, 0.0, float3(0.0), float3(0.15, 0.025, -0.1)),
            MaterialNoiseLayer(float3(20.0), 0.2, 0.2, 0.0, float3(0.0), float3(0.05)),
            MaterialNoiseLayer(float3(0.5), 0.05, 0.0, 0.0, float3(0.0), float3(0.05))),
        0.02, 0.75, 0.0, 0.0, float3(0.35, 0.225, 0.1)),
    // GRASS
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(7.0), 0.2, -0.1, 0.0, float3(0.0), float3(0.07, 0.1, 0.07)),
            MaterialNoiseLayer(float3(35.0, 10.0, 35.0), 0.6, -0.2, 0.0, float3(0.0), float3(0.15, 0.2, 0.15)),
            MaterialNoiseLayer(float3(0.07), 0.0, -0.2, 0.0, float3(0.0), float3(0.1, 0.2, 0.025))),
        0.02, 0.8, 0.0, 0.0, float3(0.17, 0.6, 0.2)),
    // PROJECTILE
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0))),
        0.0, 1.0, 0.5, 0.5, float3(1.0, 0.3, 0.3)),
    // ICE
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(1.7), 0.2, 0.1, 0.1, float3(0.0), float3(0.05, 0.05, 0.175)),
            MaterialNoiseLayer(float3(21.0), 0.1, 0.1, 0.05, float3(0.0), float3(0.05)),
            MaterialNoiseLayer(float3(0.5), 0.05, 0.0, 0.05, float3(0.0), float3(0.05))),
        0.05, 0.35, 0.3, 0.3, float3(0.7, 0.7, 0.925)),
    // WATER
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(1.0), 0.20, 0.0, 0.0, float3(0.25, -0.7, 0.2), float3(0.0)),
            MaterialNoiseLayer(float3(2.0), 0.10, 0.0, 0.0, float3(0.375, -0.5, 0.475), float3(0.0)),
            MaterialNoiseLayer(float3(4.0), 0.05, 0.0, 0.0, float3(0.5, -0.6, 0.5), float3(0.0))),
        0.05, 0.25, 0.8, 0.9, float3(0.25, 0.3, 0.6)),
    // PLAYER
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0))),
        0.0, 0.2, 0.0, 0.0, float3(0.8)),
    // AIR OOB
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0)),
            MaterialNoiseLayer(float3(0.0), 0.0, 0.0, 0.0, float3(0.0), float3(0.0))),
        0.0, 0.0, 1.0, 1.0, float3(1.0)),
    // WOOD
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(10.0, 2.5, 10.0), 0.17, 0.0, 0.0, float3(0.0), float3(0.25)),
            MaterialNoiseLayer(float3(20.0, 5.0, 20.0), 0.1, 0.0, 0.0, float3(0.0), float3(0.05)),
            MaterialNoiseLayer(float3(3.0), 0.12, 0.0, 0.0, float3(0.0), float3(0.05))),
        0.05, 0.8, 0.0, 0.0, float3(0.37, 0.225, 0.1)),
    // LEAF
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(1.7), 0.1, 0.0, 0.0, float3(0.0), float3(0.05, 0.175, 0.05)),
            MaterialNoiseLayer(float3(9.5), 0.3, 0.2, -0.5, float3(0.1, -0.2, 0.1), float3(0.05, 0.1, 0.05)),
            MaterialNoiseLayer(float3(3.5), 0.05, 0.0, 0.00, float3(0.0), float3(0.05))),
        0.05, 0.6, 0.4, 0.4, float3(0.1, 0.6, 0.1)),

    // UNBREAKABLE
    MaterialRenderProps(
        MaterialNoiseLayer[3](
            MaterialNoiseLayer(float3(2.0), 0.175, 0.1, 0.0, float3(0.0), float3(0.15)),
            MaterialNoiseLayer(float3(20.0), 0.1, 0.2, 0.0, float3(0.0), float3(0.05)),
            MaterialNoiseLayer(float3(0.5), 0.025, 0.0, 0.0, float3(0.0), float3(0.05))),
        0.04, 0.8, 0.0, 0.0, float3(0.15)),
};

uint get_dist(uint voxel_data, uint offset)
{
    return (voxel_data >> (offset * 3)) & 0x7;
}

vector<int, N> floorDiv<let N: int>(vector<int, N> a, vector<int, N> b) {
    return (a / b) - ((a % b) != 0 && (a < 0));
}

vector<int, N> floorMod<let N : int>(vector<int, N> a, vector<int, N> b) {
    vector<int, N> mod = a % b;
    return select(mod < 0, mod + abs(b), mod);
}

float3 quat_transform(float4 q, float3 v)
{
    return v + 2. * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

float4 quat_inverse(float4 q)
{
    return float4(-q.xyz, q.w) / dot(q, q);
}

float4 quaternion_from_arc(float3 src, float3 dst)
{
    float mag_avg = sqrt(dot(src, src) * dot(dst, dst));
    float dotprod = dot(src, dst);
    if (dotprod == mag_avg)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    else if (dotprod == -mag_avg)
    {
        float3 v = cross(float3(1.0, 0.0, 0.0), src);
        if (all(v == float3(0.0)))
        {
            v = v = cross(float3(0.0, 1.0, 0.0), src);
        }
        v = normalize(v);
        return float4(v, radians(180) / 2.0);
    }
    else
    {
        return normalize(float4(cross(src, dst), mag_avg + dotprod));
    }
}

int2 pcg2d(int2 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v >> 16u);

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v >> 16u);

    return v;
}

int3 pcg3d(int3 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> 16u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return v;
}

int4 pcg4d(int4 v)
{
    v = v * 1664525 + 1013904223;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v ^= v >> 16;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return v;
}

float4 voronoise(in float3 p, float u, float v)
{
    float k = 1.0 + 63.0 * pow(1.0 - v, 6.0);

    int4 i = int4(int3(floor(p)), 0);
    float3 f = fract(p);

    float2 a = float2(0.0, 0.0);
    float3 dir = float3(0.0);
    for (int z = -2; z <= 2; z++)
        for (int y = -2; y <= 2; y++)
            for (int x = -2; x <= 2; x++)
            {
                float3 g = float3(x, y, z);
                float4 hash = float4(pcg4d(i + int4(x, y, z, 0)) & 0xFF) / 128.0 - 1.0;
                float4 o = hash * float4(float3(u), 1.0);
                float3 d = g - f + o.xyz;
                float w = pow(1.0 - smoothstep(0.0, 1.414, length(d)), k);
                a += float2(o.w * w, w);
                dir += d * (2.0 * o.w - 1.0) * w;
            }

    return float4(dir, a.x) / a.y;
}

float4 grad_noise(in float3 x)
{
    // grid
    int3 p = int3(floor(x));
    float3 w = fract(x);

    // quintic interpolant
    float3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    float3 du = 30.0 * w * w * (w * (w - 2.0) + 1.0);

    // gradients
    float3 ga = float3(pcg3d(p + int3(0, 0, 0)) & 0xFF) / 128.0 - 1.0;
    float3 gb = float3(pcg3d(p + int3(1, 0, 0)) & 0xFF) / 128.0 - 1.0;
    float3 gc = float3(pcg3d(p + int3(0, 1, 0)) & 0xFF) / 128.0 - 1.0;
    float3 gd = float3(pcg3d(p + int3(1, 1, 0)) & 0xFF) / 128.0 - 1.0;
    float3 ge = float3(pcg3d(p + int3(0, 0, 1)) & 0xFF) / 128.0 - 1.0;
    float3 gf = float3(pcg3d(p + int3(1, 0, 1)) & 0xFF) / 128.0 - 1.0;
    float3 gg = float3(pcg3d(p + int3(0, 1, 1)) & 0xFF) / 128.0 - 1.0;
    float3 gh = float3(pcg3d(p + int3(1, 1, 1)) & 0xFF) / 128.0 - 1.0;

    // projections
    float va = dot(ga, w - float3(0.0, 0.0, 0.0));
    float vb = dot(gb, w - float3(1.0, 0.0, 0.0));
    float vc = dot(gc, w - float3(0.0, 1.0, 0.0));
    float vd = dot(gd, w - float3(1.0, 1.0, 0.0));
    float ve = dot(ge, w - float3(0.0, 0.0, 1.0));
    float vf = dot(gf, w - float3(1.0, 0.0, 1.0));
    float vg = dot(gg, w - float3(0.0, 1.0, 1.0));
    float vh = dot(gh, w - float3(1.0, 1.0, 1.0));

    // interpolation
    float v = va +
              u.x * (vb - va) +
              u.y * (vc - va) +
              u.z * (ve - va) +
              u.x * u.y * (va - vb - vc + vd) +
              u.y * u.z * (va - vc - ve + vg) +
              u.z * u.x * (va - vb - ve + vf) +
              u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);

    float3 d = ga +
               u.x * (gb - ga) +
               u.y * (gc - ga) +
               u.z * (ge - ga) +
               u.x * u.y * (ga - gb - gc + gd) +
               u.y * u.z * (ga - gc - ge + gg) +
               u.z * u.x * (ga - gb - ge + gf) +
               u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh) +
               du * (float3(vb - va, vc - va, ve - va) +
                     u.yzx * float3(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf) +
                     u.zxy * float3(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg) +
                     u.yzx * u.zxy * (-va + vb + vc - vd + ve - vf - vg + vh));

    return float4(d, v);
}
