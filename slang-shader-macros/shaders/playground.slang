[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_DELTA_TIMEAttribute
{
};

// Control a buffer with external data
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_EXTERNALAttribute
{
    int count;
};

// Initialize a \`float\` buffer with zeros of the provided size.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_ZEROSAttribute
{
    int count;
};

// Initialize a \`float\` texture with zeros of the provided size.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_BLACKAttribute
{
    int width;
    int height;
};

// Initialize a \`float\` texture with zeros of the provided size.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_BLACK_3DAttribute
{
    uint sizeX;
    uint sizeY;
    uint sizeZ;
};

// Initialize a \`float\` texture with zeros of a size proportional to the screen size.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_BLACK_SCREENAttribute
{
    float widthScale;
    float heightScale;
};

// Initialize a texture with image from URL.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_URLAttribute
{
    string url;
};

// Bind a sampler with reasonable defaults for texture sampling.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_SAMPLERAttribute
{
};

// Initialize a \`float\` buffer with uniform random floats between 0 and 1.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_RANDAttribute
{
    int count;
};

// Gives the current time in milliseconds.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_TIMEAttribute
{
};

// Gives the current shader playback frame index.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_FRAME_IDAttribute
{
};

// Gives mouse position info.
// \`xy\`: mouse position (in pixels) during last button down.
// \`abs(zw)\`: mouse position during last button click.
// \`sign(mouze.z)\`: button is down
// \`sign(mouze.w)\`: button is clicked
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_MOUSE_POSITIONAttribute
{
};

// Gives the size of the screen
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_SCREEN_SIZEAttribute
{
};

// Checks if a given key is down. `1.0` if key is down, `0.0` otherwise. Key name comes from either javascript `event.code` or `event.key`.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_KEYAttribute
{
    string keyName;
};

// Control a \`float\` uniform with a provided default, minimum, and maximum.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_SLIDERAttribute
{
    float default;
    float min;
    float max;
};

// Control a \`float3\` color uniform with a provided default color.
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_COLOR_PICKAttribute
{
    float defaultRed;
    float defaultGreen;
    float defaultBlue;
};

// Dispatch a compute pass with the given grid of threads.
// The number of work - groups will be determined by dividing by the number of threads per work - group and rounding up.
[__AttributeUsage(_AttributeTargets.Function)]
public struct playground_CALLAttribute
{
    int x;
    int y;
    int z;
};

// Dispatch a compute pass using the resource size to determine the work-group size.
[__AttributeUsage(_AttributeTargets.Function)]
public struct playground_CALL_SIZE_OFAttribute
{
    string resourceName;
};

// Only dispatch the compute pass once at the start of rendering.
[__AttributeUsage(_AttributeTargets.Function)]
public struct playground_CALL_ONCEAttribute
{
};

[__AttributeUsage(_AttributeTargets.Function)]
public struct playground_CALL_INDIRECTAttribute
{
    string indirectBuffer;
    uint offset;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_REBIND_FOR_DRAWAttribute
{
    string originalTexture;
};

[__AttributeUsage(_AttributeTargets.Function)]
public struct playground_DRAWAttribute
{
    uint vertexCount;
    string fragmentEntrypoint;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_MODELAttribute
{
    string path;
};

// Control a uniform with external data
[__AttributeUsage(_AttributeTargets.Var)]
public struct playground_EXTERNAL_UNIFORMAttribute
{};

// Control a uniform with external data
[__AttributeUsage(_AttributeTargets.Struct)]
public struct playground_ANNOTATIONAttribute
{
    string annotation;
};
