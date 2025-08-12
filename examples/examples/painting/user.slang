import playground;
import utils;

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
