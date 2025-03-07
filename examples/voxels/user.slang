import playground;
import common;
import binding;
import worldgen;
import render;
import voxel_update;

[format("rgba8")]
[playground::BLACK_SCREEN(1.0, 1.0)]
WTexture2D<float4> screen;

[playground::KEY_INPUT("W")]
uniform float wPressed;
[playground::KEY_INPUT("A")]
uniform float aPressed;
[playground::KEY_INPUT("S")]
uniform float sPressed;
[playground::KEY_INPUT("D")]
uniform float dPressed;
[playground::KEY_INPUT("Space")]
uniform float spacePressed;
[playground::KEY_INPUT("Ctrl")]
uniform float ctrlPressed;
[playground::KEY_INPUT("Shift")]
uniform float shiftPressed;

[playground::SLIDER(5.0, 0.0, 20.0)]
uniform float speed;
[playground::DELTA_TIME]
uniform float deltaTime;

[playground::MOUSEPOSITION]
uniform float4 mouse;

[playground::CALL_ONCE]
[numthreads(1, 1, 1)]
[playground::CALL(1, 1, 1)]
void setup(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    playerData[0].playerPos = float3(10.0, 100.0, 10.0);

    uint3 voxelSize;
    voxels.GetDimensions(voxelSize.x, voxelSize.y, voxelSize.z);
    indirectBuffer[0] = voxelSize.x / VOXEL_UPDATE_SIZE / 2;
    indirectBuffer[1] = voxelSize.y / VOXEL_UPDATE_SIZE / 2;
    indirectBuffer[2] = voxelSize.z / VOXEL_UPDATE_SIZE / 2;
}

[shader("compute")]
[numthreads(1, 1, 1)]
[playground::CALL(1, 1, 1)]
void update(uint2 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 imageSize;
    screen.GetDimensions(imageSize.x, imageSize.y);

    float3 cameraPos = playerData[0].playerPos;

    float3 cameraDir = getCameraDirection(float2(imageSize));
    float3 up = float3(0.0, 1.0, 0.0);
    float3 left = normalize(cross(up, cameraDir));

    float3 inputVector = (1.0 + shiftPressed) * speed * deltaTime * float3(wPressed - sPressed, aPressed - dPressed, spacePressed - ctrlPressed);
    float3x3 inputTransform = float3x3(cameraDir, left, up);

    cameraPos += mul(inputVector, inputTransform);

    playerData[0].playerPos = cameraPos;

    playerData[0].voxelOffset = (playerData[0].voxelOffset + 1) % 8;
}

float4 imageMain(uint2 dispatchThreadID, int2 screenSize)
{
    let uv = (2.0 * (float2(dispatchThreadID) / float2(screenSize)) - 1.0) * float2(float(screenSize.x) / float(screenSize.y), 1.0);
    let cameraDir = getCameraDirection(screenSize);
    let up = float3(0.0, 1.0, 0.0);
    let cameraLeft = -normalize(cross(up, cameraDir));
    let cameraUp = cross(cameraLeft, cameraDir);

    let ray = normalize(cameraDir + uv.x * cameraLeft + uv.y * cameraUp);

    var pos = playerData[0].playerPos;

    RaycastResult primary_ray = raycast(pos, ray, primary_ray_dist);

    if (primary_ray.layer_count == 0)
    {
        return float4(1.0, 0.0, 0.0, 1.0);
    }

    return float4(get_color(pos, ray, primary_ray, screenSize.y), 1.0);
}
