import playground;
import common;
import binding;

static const float3 light_dir = normalize(float3(0.5, -1, 0.25));
struct SimpleRaycastResult
{
    float3 pos;
    uint voxel_data;
    float dist;
    float passthrough_light;
};

struct RaycastResultLayer
{
    float3 pos;
    float3 normal;
    uint voxel_data;
    float dist;
    bool is_leaving_medium;
};

struct RaycastResult
{
    RaycastResultLayer layers[5];
    uint layer_count;
    float dist;
};

float3 ray_box_dist(float3 pos, float3 ray, float3 vmin, float3 vmax)
{
    float3 normMinDiff = (vmin - pos) / ray;
    float3 normMaxDiff = (vmax - pos) / ray;
    return max(normMinDiff, normMaxDiff);
}

RaycastResultLayer single_raycast(float3 pos, float3 ray, uint max_iterations)
{
    uint offset = 0;
    if (ray.x < 0)
        offset += 1;
    if (ray.y < 0)
        offset += 2;
    if (ray.z < 0)
        offset += 4;

    float3 ray_pos = pos;
    float3 normal = float3(0);
    float depth = 0;
    uint voxel_data = MAT_OOB << 24;
    uint medium = get_data(int3(floor(ray_pos))) >> 24;
    bool did_hit = false;

    for (uint i = 0; i < max_iterations; i++)
    {
        float3 floor_pos = floor(ray_pos);
        voxel_data = get_data(int3(floor_pos));
        float3 v_min;
        float3 v_max;
        uint voxel_material = voxel_data >> 24;
        if (voxel_material == MAT_AIR_OOB)
        {
            v_min = floor(ray_pos / CHUNK_SIZE) * CHUNK_SIZE;
            v_max = v_min + float3(CHUNK_SIZE);
        }
        else if (voxel_material == medium)
        {
            uint dist = 0;
            if (physics_properties[voxel_material].is_data_standard_distance)
            {
                dist = get_dist(voxel_data, offset);
            }
            v_min = floor_pos - float3(dist);
            v_max = floor_pos + float3(dist + 1);
        }
        else
        {
            did_hit = true;
            break;
        }
        float3 delta = ray_box_dist(ray_pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        depth += dist_diff;
        ray_pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z)
        {
            normal = float3(-sign(ray.x), 0, 0);
            if (ray.x > 0 && ray_pos.x < v_max.x)
            {
                ray_pos.x = v_max.x;
            }
            else if (ray.x < 0 && ray_pos.x >= v_min.x)
            {
                ray_pos.x = v_min.x - 0.001;
            }
        }
        else if (delta.y < delta.z)
        {
            normal = float3(0, -sign(ray.y), 0);
            if (ray.y > 0 && ray_pos.y < v_max.y)
            {
                ray_pos.y = v_max.y;
            }
            else if (ray.y < 0 && ray_pos.y >= v_min.y)
            {
                ray_pos.y = v_min.y - 0.001;
            }
        }
        else
        {
            normal = float3(0, 0, -sign(ray.z));
            if (ray.z > 0 && ray_pos.z < v_max.z)
            {
                ray_pos.z = v_max.z;
            }
            else if (ray.z < 0 && ray_pos.z >= v_min.z)
            {
                ray_pos.z = v_min.z - 0.001;
            }
        }
    }

    if (!did_hit)
    {
        return RaycastResultLayer(pos, float3(0.0), MAT_OOB << 24, 0.0, false);
    }

    return RaycastResultLayer(ray_pos, normal, voxel_data, depth, false);
}

SimpleRaycastResult simple_raycast(float3 pos, float3 ray, uint max_iterations)
{
    uint offset = 0;
    if (ray.x < 0)
        offset += 1;
    if (ray.y < 0)
        offset += 2;
    if (ray.z < 0)
        offset += 4;

    float passthrough_light = 1.0;

    float3 ray_pos = pos;
    float depth = 0;
    uint voxel_data = MAT_OOB << 24;
    uint prev_voxel_material = MAT_OOB;
    bool did_hit = false;

    for (uint i = 0; i < max_iterations; i++)
    {
        float3 floor_pos = floor(ray_pos);
        voxel_data = get_data(int3(floor_pos));
        float3 v_min;
        float3 v_max;
        uint voxel_material = voxel_data >> 24;
        if (!is_transparent[voxel_material])
        {
            did_hit = true;
            break;
        }
        if (voxel_material == MAT_AIR_OOB)
        {
            v_min = floor(ray_pos / CHUNK_SIZE) * CHUNK_SIZE;
            v_max = v_min + float3(CHUNK_SIZE);
        }
        else
        {
            uint dist = 0;
            if (physics_properties[voxel_material].is_data_standard_distance)
            {
                dist = get_dist(voxel_data, offset);
            }
            v_min = floor_pos - float3(dist);
            v_max = floor_pos + float3(dist + 1);
        }
        float3 delta = ray_box_dist(ray_pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        depth += dist_diff;

        if (prev_voxel_material != voxel_material)
        {
            passthrough_light *= material_render_props[voxel_material].transparency;
        }
        prev_voxel_material = voxel_material;
        ray_pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z)
        {
            if (ray.x > 0 && ray_pos.x < v_max.x)
            {
                ray_pos.x = v_max.x;
            }
            else if (ray.x < 0 && ray_pos.x >= v_min.x)
            {
                ray_pos.x = v_min.x - 0.001;
            }
        }
        else if (delta.y < delta.z)
        {
            if (ray.y > 0 && ray_pos.y < v_max.y)
            {
                ray_pos.y = v_max.y;
            }
            else if (ray.y < 0 && ray_pos.y >= v_min.y)
            {
                ray_pos.y = v_min.y - 0.001;
            }
        }
        else
        {
            if (ray.z > 0 && ray_pos.z < v_max.z)
            {
                ray_pos.z = v_max.z;
            }
            else if (ray.z < 0 && ray_pos.z >= v_min.z)
            {
                ray_pos.z = v_min.z - 0.001;
            }
        }
    }

    if (!did_hit)
    {
        return SimpleRaycastResult(pos, MAT_OOB << 24, 0.0, passthrough_light);
    }

    return SimpleRaycastResult(ray_pos, voxel_data, depth, passthrough_light);
}

static const uint LAYER_COUNT = 5;
RaycastResult raycast(float3 pos, float3 ray, uint max_iterations)
{
    RaycastResultLayer[5] layers;
    uint offset = 0;
    if (ray.x < 0)
        offset += 1;
    if (ray.y < 0)
        offset += 2;
    if (ray.z < 0)
        offset += 4;

    float3 ray_pos = pos;
    float3 normal = -ray;
    float depth = 0;
    uint voxel_data = MAT_OOB << 24;
    uint medium = MAT_AIR;
    uint layer_idx = 0;
    for (uint i = 0; i < max_iterations; i++)
    {
        float3 floor_pos = floor(ray_pos);
        voxel_data = get_data(int3(floor_pos));
        uint voxel_material = voxel_data >> 24;
        float3 v_min;
        float3 v_max;
        if (voxel_material == MAT_AIR || voxel_material == MAT_AIR_OOB)
        {
            if (medium != MAT_AIR)
            {
                layers[layer_idx] = RaycastResultLayer(ray_pos, normal, medium << 24, depth, true);
                layer_idx++;
                if (layer_idx >= LAYER_COUNT)
                    break;
            }
        }
        if (voxel_material == MAT_AIR)
        {
            uint dist = get_dist(voxel_data, offset);
            v_min = floor_pos - float3(dist);
            v_max = floor_pos + float3(dist + 1);
            medium = MAT_AIR;
        }
        else if (voxel_material == MAT_AIR_OOB)
        {
            v_min = floor(ray_pos / CHUNK_SIZE) * CHUNK_SIZE;
            v_max = v_min + float3(CHUNK_SIZE);
            medium = MAT_AIR;
        }
        else if (is_transparent[voxel_material])
        {
            uint dist = 0;
            if (voxel_material == MAT_WATER)
            {
                dist = get_dist(voxel_data, offset);
            }
            v_min = floor_pos;
            v_max = floor_pos + float3(1);
            if (medium != voxel_material)
            {
                layers[layer_idx] = RaycastResultLayer(ray_pos, normal, voxel_data, depth, false);
                layer_idx++;
                if (layer_idx >= LAYER_COUNT)
                    break;
            }
            medium = voxel_material;
        }
        else
        {
            layers[layer_idx] = RaycastResultLayer(ray_pos, normal, voxel_data, depth, false);
            layer_idx++;
            break;
        }
        float3 delta = ray_box_dist(ray_pos, ray, v_min, v_max);
        float dist_diff = min(delta.x, min(delta.y, delta.z));
        depth += dist_diff;
        ray_pos += ray * dist_diff;
        if (delta.x < delta.y && delta.x < delta.z)
        {
            normal = float3(-sign(ray.x), 0, 0);
            if (ray.x < 0 && ray_pos.x >= v_min.x)
            {
                ray_pos.x = v_min.x - 0.001;
            }
        }
        else if (delta.y < delta.z)
        {
            normal = float3(0, -sign(ray.y), 0);
            if (ray.y < 0 && ray_pos.y >= v_min.y)
            {
                ray_pos.y = v_min.y - 0.001;
            }
        }
        else
        {
            normal = float3(0, 0, -sign(ray.z));
            if (ray.z < 0 && ray_pos.z >= v_min.z)
            {
                ray_pos.z = v_min.z - 0.001;
            }
        }
    }

    return RaycastResult(layers, layer_idx, depth);
}

struct MaterialProperties
{
    float3 albedo;
    float3 normal;
    float ior;
    float roughness;
    float metallic;
    float emmision;
    float transparency;
    float depth_transparency;
};

float max3(float3 v)
{
    return max(max(v.x, v.y), v.z);
}

MaterialProperties material_props(RaycastResultLayer resultLayer, float3 ray_dir, uint texture_layer_count, uint vertical_resolution)
{
    uint material = resultLayer.voxel_data >> 24;
    uint data = resultLayer.voxel_data & 0xFFFFFF;
    MaterialRenderProps mat_render_props = material_render_props[material];
    if (material == MAT_AIR || material == MAT_AIR_OOB)
    {
        // air: invalid state
        return MaterialProperties(float3(1.0, 0.0, 0.0), resultLayer.normal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    else if (material == MAT_OOB)
    {
        // out of bounds: invalid state
        return MaterialProperties(float3(0.0, 0.0, 1.0), resultLayer.normal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    else if (material == MAT_PROJECTILE || material == MAT_PLAYER)
    {
        return MaterialProperties(mat_render_props.color, resultLayer.normal, mat_render_props.ior, mat_render_props.roughness, 0.0, 0.1, mat_render_props.transparency, 0.0);
    }
    float3 normal = resultLayer.normal;
    float3 color = mat_render_props.color;
    float roughness = mat_render_props.roughness;
    float transparency = mat_render_props.transparency;
    for (int layer_idx = 0; layer_idx < texture_layer_count; layer_idx++)
    {
        MaterialNoiseLayer layer = mat_render_props.layers[layer_idx];
        float4 noise = grad_noise(layer.scale * resultLayer.pos + layer.movement * time);
        float distance_noise_factor = clamp(-0.1 * float(vertical_resolution) * dot(ray_dir, resultLayer.normal) / (max(resultLayer.dist, 0.1) * max3(layer.scale)), 0.0, 1.0);
        normal += distance_noise_factor * layer.normal_impact * noise.xyz * (float3(1) - abs(resultLayer.normal));
        color += layer.layer_color * lerp(0.0, noise.w, distance_noise_factor);
        roughness += distance_noise_factor * layer.roughness_impact * noise.w;
        transparency += distance_noise_factor * layer.transparency_impact * noise.w;
    }
    normal = normalize(normal);
    if (physics_properties[material].is_data_damage)
    {
        color *= (1.0 - float(data) / material_damage_threshhold[material]);
    }
    return MaterialProperties(
        color,
        normal,
        mat_render_props.ior,
        roughness,
        0.0,
        0.05,
        transparency,
        mat_render_props.depth_transparency);
}

MaterialProperties position_material(RaycastResultLayer resultLayer, float3 ray_dir, uint texture_layer_count, uint vertical_resolution)
{
    if (resultLayer.voxel_data >> 24 == MAT_PLAYER || resultLayer.voxel_data >> 24 == MAT_PROJECTILE)
    {
        return material_props(resultLayer, ray_dir, texture_layer_count, vertical_resolution);
    }
    float3 relative_pos = resultLayer.pos - floor(resultLayer.pos) - 0.5;
    float3 weights = abs(relative_pos);
    uint result_vox = resultLayer.voxel_data;
    HeightData voxel_height_data = height_data[result_vox >> 24];
    float result_height = (voxel_height_data.offset + voxel_height_data.impact * grad_noise(voxel_height_data.scale * resultLayer.pos + voxel_height_data.movement * time).w) * (1.0 - weights.x) * (1.0 - weights.y) * (1.0 - weights.z);
    uint face_voxel_material = get_data(int3(floor(resultLayer.pos + resultLayer.normal))) >> 24;
    for (int i = 1; i < 8; i++)
    {
        float3 voxel_direction = float3(float(i & 1), float((i & 2) >> 1), float((i & 4) >> 2)) * sign(relative_pos);
        if (dot(voxel_direction, resultLayer.normal) > 0.0)
            continue;
        uint voxel = get_data(int3(floor(resultLayer.pos) + voxel_direction));
        uint voxel_material = voxel >> 24;
        if (voxel_material == face_voxel_material)
            continue;
        if (
            voxel_material == MAT_OOB || voxel_material == MAT_AIR || voxel_material == MAT_AIR_OOB)
            continue;
        float weight = 1.0;
        if ((i & 1) == 0)
            weight *= 1.0 - weights.x;
        else
            weight *= weights.x;
        if ((i & 2) == 0)
            weight *= 1.0 - weights.y;
        else
            weight *= weights.y;
        if ((i & 4) == 0)
            weight *= 1.0 - weights.z;
        else
            weight *= weights.z;
        voxel_height_data = height_data[voxel_material];
        float distance_noise_factor = clamp(-0.1 * float(vertical_resolution) * dot(ray_dir, resultLayer.normal) / (max(resultLayer.dist, 0.1) * max3(voxel_height_data.scale)), 0.0, 1.0);
        float height = (voxel_height_data.offset - voxel_height_data.impact * distance_noise_factor * grad_noise(voxel_height_data.scale * resultLayer.pos).w) * weight;
        if (height > result_height)
        {
            result_height = height;
            result_vox = voxel;
        }
    }
    resultLayer.voxel_data = result_vox;
    return material_props(resultLayer, ray_dir, texture_layer_count, vertical_resolution);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float3 get_light(float3 L, float3 V, float3 lightColor, float attenuation, MaterialProperties mat_props)
{
    float3 F0 = float3(mat_props.ior);
    F0 = lerp(F0, mat_props.albedo, mat_props.metallic);
    // calculate per-light radiance
    float3 H = normalize(V + L);
    float3 radiance = lightColor * attenuation;

    // cook-torrance brdf
    float NDF = DistributionGGX(mat_props.normal, H, mat_props.roughness);
    float G = GeometrySmith(mat_props.normal, V, L, mat_props.roughness);
    float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float3 kS = F;
    float3 kD = float3(1.0) - kS;
    kD *= 1.0 - mat_props.metallic;

    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(mat_props.normal, V), 0.0) * max(dot(mat_props.normal, L), 0.0) + 0.0001;
    float3 specular = numerator / denominator;

    // add to outgoing radiance Lo
    float NdotL = max(dot(mat_props.normal, L), 0.0);
    return (kD * mat_props.albedo / PI + specular) * radiance * NdotL;
}
static const float epsilon = 0.001;
float3 get_color(float3 pos, float3 ray, RaycastResult primary_ray, uint vertical_resolution)
{
    float3 color = float3(0.0);
    float multiplier = 1.0;
    int i = 0;

    while (multiplier > 0.05 && i < primary_ray.layer_count)
    {
        if (primary_ray.layers[i].voxel_data >> 24 == MAT_OOB)
        {
            float sky_brightness = max(dot(ray, -light_dir), 0.0);
            sky_brightness += pow(sky_brightness, 20.0);
            color += multiplier * lerp(float3(0.35, 0.4, 0.8), float3(0.629, 0.908, 1.0), sky_brightness);
            break;
        }
        MaterialProperties mat_props = position_material(primary_ray.layers[i], ray, 3 - i / 2, vertical_resolution);
        color += (1.0 - mat_props.transparency) * multiplier * mat_props.albedo * mat_props.emmision;

        SimpleRaycastResult shade_check = simple_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, -light_dir, shadow_ray_dist);
        if (shade_check.voxel_data >> 24 == MAT_OOB)
        {
            color += multiplier * get_light(-light_dir, -ray, (1.0 - mat_props.transparency) * shade_check.passthrough_light * float3(1.0), 1.0, mat_props);
        }

        float3 reflection = reflect(ray, mat_props.normal);
        RaycastResultLayer reflection_check = single_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, reflection, reflection_ray_dist);
        MaterialRenderProps reflection_props = material_render_props[reflection_check.voxel_data >> 24];
        SimpleRaycastResult reflection_light_check = simple_raycast(reflection_check.pos - epsilon * reflection, -light_dir, ao_ray_dist);
        float3 light = float3(0);
        if (reflection_check.voxel_data >> 24 == MAT_OOB)
        {
            light += float3(pow(dot(reflection, -light_dir), 3.0));
        }
        else if (reflection_light_check.voxel_data >> 24 == MAT_OOB)
        {
            light += (1.0 - reflection_props.transparency) * reflection_light_check.passthrough_light * reflection_props.color * 0.15;
        }
        color += multiplier * get_light(reflection, -ray, (1 - mat_props.transparency) * light, 1.0, mat_props);

        float3 ao_dir = mat_props.normal;
        SimpleRaycastResult ao_check = simple_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, ao_dir, ao_ray_dist);
        if (ao_check.voxel_data >> 24 == MAT_OOB)
        {
            float light_power = pow(dot(ao_dir, -light_dir), 3.0);
            color += multiplier * get_light(ao_dir, -ray, (1 - mat_props.transparency) * ao_check.passthrough_light * float3(1.0), light_power, mat_props);
        }

        ao_dir = normalize(2.0 * mat_props.normal - light_dir);
        ao_check = simple_raycast(primary_ray.layers[i].pos + epsilon * primary_ray.layers[i].normal, ao_dir, ao_ray_dist);
        if (ao_check.voxel_data >> 24 == MAT_OOB)
        {
            float light_power = pow(dot(ao_dir, -light_dir), 3.0);
            color += multiplier * get_light(ao_dir, -ray, (1 - mat_props.transparency) * ao_check.passthrough_light * float3(1.0), light_power, mat_props);
        }

        if (mat_props.depth_transparency > 0.0 && !primary_ray.layers[i].is_leaving_medium)
        {
            float dist = 0;
            if (i + 1 < primary_ray.layer_count)
            {
                dist = primary_ray.layers[i + 1].dist - primary_ray.layers[i].dist;
            }
            else
            {
                dist = primary_ray.dist - primary_ray.layers[i].dist;
            }
            float depth_transparency = pow(mat_props.depth_transparency, dist);
            color += (1.0 - depth_transparency) * mat_props.transparency * multiplier * mat_props.albedo;
            multiplier *= depth_transparency;
        }

        multiplier *= mat_props.transparency;
        i++;
    }

    return color;
}
