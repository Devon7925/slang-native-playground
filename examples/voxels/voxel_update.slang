import playground;
import common;
import binding;

void set_data(int3 global_pos, uint data)
{
    uint og_voxel_data = get_data(global_pos);
    if (og_voxel_data >> 24 == MAT_OOB)
        return;
    if (og_voxel_data >> 24 == MAT_AIR_OOB)
        return;
    voxels.Store(get_index(global_pos), data);
}

[numthreads(VOXEL_UPDATE_SIZE, VOXEL_UPDATE_SIZE, VOXEL_UPDATE_SIZE)]
[playground::CALL_INDIRECT("indirectBuffer", 0)]
void voxelUpdate(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    int3 pos = 2 * dispatchThreadId + get_voxel_update_offset();
    uint2 pos_data[2][2][2];
    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < 2; j++)
        {
            for (uint k = 0; k < 2; k++)
            {
                uint raw_voxel = get_data(pos + int3(i, j, k));
                pos_data[i][j][k] = uint2(raw_voxel >> 24, raw_voxel & 0xFFFFFF);
            }
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < 2; j++)
        {
            for (uint k = 0; k < 2; k++)
            {
                if (physics_properties[pos_data[i][j][k].x].is_data_damage)
                {
                    if (pos_data[i][j][k].y >= material_damage_threshhold[pos_data[i][j][k].x])
                    {
                        pos_data[i][j][k] = uint2(MAT_AIR, 0);
                    }
                }
            }
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        for (uint k = 0; k < 2; k++)
        {
            uint bottom_voxel = pos_data[i][0][k].x;
            uint top_voxel = pos_data[i][1][k].x;
            if (bottom_voxel == MAT_AIR && top_voxel == MAT_WATER)
            {
                pos_data[i][0][k] = uint2(top_voxel, 0);
                pos_data[i][1][k] = uint2(bottom_voxel, 0);
            }
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        for (uint k = 0; k < 2; k++)
        {
            uint bottom_voxel = pos_data[1 - i][0][k].x;
            uint top_voxel = pos_data[i][1][k].x;
            if (bottom_voxel == MAT_AIR && top_voxel == MAT_WATER)
            {
                pos_data[1 - i][0][k] = uint2(top_voxel, 0);
                pos_data[i][1][k] = uint2(bottom_voxel, 0);
            }
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        for (uint k = 0; k < 2; k++)
        {
            uint bottom_voxel = pos_data[i][0][1 - k].x;
            uint top_voxel = pos_data[i][1][k].x;
            if (bottom_voxel == MAT_AIR && top_voxel == MAT_WATER)
            {
                pos_data[i][0][1 - k] = uint2(top_voxel, 0);
                pos_data[i][1][k] = uint2(bottom_voxel, 0);
            }
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        uint to_voxel = pos_data[pos.y % 2][1][i].x;
        uint from_voxel = pos_data[(1 + pos.y) % 2][1][i].x;
        if (from_voxel == MAT_WATER && to_voxel == MAT_AIR)
        {
            pos_data[pos.y % 2][1][i] = uint2(from_voxel, 0);
            pos_data[(1 + pos.y) % 2][1][i] = uint2(to_voxel, 0);
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        uint to_voxel = pos_data[i][1][pos.y % 2].x;
        uint from_voxel = pos_data[i][1][(1 + pos.y) % 2].x;
        if (from_voxel == MAT_WATER && to_voxel == MAT_AIR)
        {
            pos_data[i][1][pos.y % 2] = uint2(from_voxel, 0);
            pos_data[i][1][(1 + pos.y) % 2] = uint2(to_voxel, 0);
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < 2; j++)
        {
            for (uint k = 0; k < 2; k++)
            {
                if (pos_data[i][j][k].x == MAT_AIR)
                {
                    int offset = 3 * int(((k << 2) | (j << 1) | i));
                    int3 d = int3(1) - int3(i, j, k) * int3(2);
                    uint direction_dist = 7;
                    for (uint m = 1; m < 8; m++)
                    {
                        int3 dir = d * int3(m % 2, (m / 2) % 2, m / 4) + int3(i, j, k);
                        if (pos_data[dir.x][dir.y][dir.z].x == MAT_OOB)
                        {
                            direction_dist = min(direction_dist, bitfieldExtract(pos_data[i][j][k].y, offset, 3));
                            continue;
                        }
                        else if (pos_data[dir.x][dir.y][dir.z].x == MAT_AIR)
                        {
                            direction_dist = min(direction_dist, bitfieldExtract(pos_data[dir.x][dir.y][dir.z].y, offset, 3) + 1);
                            continue;
                        }
                        else if (pos_data[dir.x][dir.y][dir.z].x == MAT_AIR_OOB)
                        {
                            continue;
                        }
                        direction_dist = 0;
                        break;
                    }
                    pos_data[i][j][k].y = bitfieldInsert(pos_data[i][j][k].y, direction_dist, offset, 3);
                }
                else if (physics_properties[pos_data[i][j][k].x].is_data_standard_distance)
                {
                    int offset = 3 * int(((k << 2) | (j << 1) | i));
                    int3 d = int3(1) - int3(i, j, k) * int3(2);
                    uint direction_dist = 7;
                    for (uint m = 1; m < 8; m++)
                    {
                        int3 dir = d * int3(m % 2, (m / 2) % 2, m / 4) + int3(i, j, k);
                        if (pos_data[dir.x][dir.y][dir.z].x == MAT_OOB)
                        {
                            direction_dist = min(direction_dist, bitfieldExtract(pos_data[i][j][k].y, offset, 3));
                            continue;
                        }
                        else if (pos_data[dir.x][dir.y][dir.z].x == pos_data[i][j][k].x)
                        {
                            direction_dist = min(direction_dist, bitfieldExtract(pos_data[dir.x][dir.y][dir.z].y, offset, 3) + 1);
                            continue;
                        }
                        direction_dist = 0;
                        break;
                    }
                    pos_data[i][j][k].y = bitfieldInsert(pos_data[i][j][k].y, direction_dist, offset, 3);
                }
            }
        }
    }

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < 2; j++)
        {
            for (uint k = 0; k < 2; k++)
            {
                set_data(pos + int3(i, j, k), (pos_data[i][j][k].x << 24) | pos_data[i][j][k].y);
            }
        }
    }
}
