// A simple shader ported from https://www.shadertoy.com/view/XdlSDs.
// Author: dynamite
// License: CC BY-NC-SA 3.0 https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en

import playground;

[format("rgba8")]
[playground::BLACK_SCREEN(1.0, 1.0)]
WTexture2D<float4> outputTexture;

[format("rgba8")]
[playground::REBIND_FOR_DRAW("outputTexture")]
Texture2D<float4> drawTexture;

[playground::SAMPLER]
SamplerState sampler;

[shader("compute")]
[numthreads(16, 16, 1)]
[playground::CALL::SIZE_OF("outputTexture")]
void imagMain(uint2 dispatchThreadID: SV_DispatchThreadID)
{
    uint width = 0;
    uint height = 0;
    outputTexture.GetDimensions(width, height);

    float4 color = imagMain(dispatchThreadID.xy, int2(width, height));

    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
        return;

    outputTexture.Store(dispatchThreadID.xy, color);
}

struct Vertex
{
    float2 position;
    float2 uv;
}

struct VOut
{
    float4 position : SV_Position;
    float2 texcoord;
}

const static Vertex[] verticies = {
    Vertex(float2(-1.0, -1.0), float2(0.0, 0.0)),
    Vertex(float2(1.0, -1.0), float2(1.0, 0.0)),
    Vertex(float2(-1.0, 1.0), float2(0.0, 1.0)),
    Vertex(float2(-1.0, 1.0), float2(0.0, 1.0)),
    Vertex(float2(1.0, -1.0), float2(1.0, 0.0)),
    Vertex(float2(1.0, 1.0), float2(1.0, 1.0)),
};

[playground::DRAW(6, "fragmentMain")]
[shader("vertex")]
VOut vertexMain(uint vertexID: SV_VertexID)
{
    Vertex input = verticies[vertexID];
    VOut output;
    output.position = float4(input.position, 0.0, 1.0);
    output.texcoord = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VOut input) : SV_Target
{
    let color = drawTexture.Sample(sampler, input.texcoord);
    return color;
}

typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;

float4 imagMain(uint2 dispatchThreadID, int2 screenSize)
{
    vec2 p = (dispatchThreadID.xy * 2.0f - screenSize.xy) / (float)screenSize.y;
    float tau = float.getPi() * 2.0;
    float a = atan2(p.x,p.y);
    float r = length(p)*0.75;
    vec2 uv = vec2(a/tau,r);
	
    float time = getTime();

	//get the color
	float xCol = (uv.x - time/3) * 3.0;
	xCol = fmod(abs(xCol), 3.0f);
	vec3 horColour = vec3(0.25, 0.25, 0.25);
	
	if (xCol < 1.0)
    {
		horColour.r += 1.0 - xCol;
		horColour.g += xCol;
	}
	else if (xCol < 2.0)
    {
		xCol -= 1.0;
		horColour.g += 1.0 - xCol;
		horColour.b += xCol;
	}
	else
    {
		xCol -= 2.0;
		horColour.b += 1.0 - xCol;
		horColour.r += xCol;
	}

	// draw color beam
	uv = (2.0 * uv) - 1.0;
	float beamWidth = (0.7 + 
        0.5 * cos(uv.x * 10.0 * tau * 0.15 * clamp(floor(5.0 + 10.0 * cos(time)), 0.0, 10.0)))
        * abs(1.0 / (30.0 * uv.y));
	vec3 horBeam = vec3(beamWidth);
	return vec4(((horBeam) * horColour), 1.0);
}