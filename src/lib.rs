mod compute_pipeline;
mod draw_pipeline;
#[cfg(not(target_arch = "wasm32"))]
mod egui_tools;

use draw_pipeline::DrawPipeline;
use rand::Rng;
use regex::Regex;
use slang_compiler_type_definitions::{
    CallCommand, CallCommandParameters, CompilationResult, DrawCommand, ResourceCommandData,
    UniformController, UniformSourceData,
};
use slang_shader_macros::compile_shader;
use wgpu::{BufferDescriptor, Extent3d, Features};

use std::{borrow::Cow, cell::RefCell, collections::HashMap, panic, rc::Rc, sync::Arc};

use compute_pipeline::ComputePipeline;
use std::collections::HashSet;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, SmolStr},
    window::{Window, WindowId},
};

#[cfg(debug_assertions)]
#[cfg(target_family = "wasm")]
extern crate console_error_panic_hook;

struct MouseState {
    last_mouse_clicked_pos: PhysicalPosition<f64>,
    last_mouse_down_pos: PhysicalPosition<f64>,
    current_mouse_pos: PhysicalPosition<f64>,
    mouse_clicked: bool,
    is_mouse_down: bool,
}

struct KeyboardState {
    pressed_keys: HashSet<Key>,
}

impl KeyboardState {
    fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
        }
    }

    fn key_pressed(&mut self, key: Key) {
        self.pressed_keys.insert(key);
    }

    fn key_released(&mut self, key: Key) {
        self.pressed_keys.remove(&key);
    }
}

struct State {
    window: Arc<Window>,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    compute_pipelines: HashMap<String, ComputePipeline>,
    draw_pipelines: Vec<DrawPipeline>,
    bindings: HashMap<String, wgpu::BindGroupLayoutEntry>,
    resource_commands: HashMap<String, ResourceCommandData>,
    call_commands: Vec<CallCommand>,
    draw_commands: Vec<DrawCommand>,
    uniform_components: Rc<RefCell<Vec<UniformController>>>,
    uniform_size: u64,
    hashed_strings: HashMap<u32, String>,
    allocated_resources: HashMap<String, GPUResource>,
    mouse_state: MouseState,
    keyboard_state: KeyboardState,
    print_receiver: Option<futures::channel::oneshot::Receiver<()>>,
    first_frame: bool,
    delta_time: f32,
    last_frame_time: web_time::Instant,
    launch_time: web_time::Instant,
}

#[derive(Debug, Clone)]
enum GPUResource {
    Texture(wgpu::Texture),
    Buffer(wgpu::Buffer),
    Sampler(wgpu::Sampler),
}

impl GPUResource {
    fn destroy(&mut self) {
        match self {
            GPUResource::Texture(texture) => texture.destroy(),
            GPUResource::Buffer(buffer) => buffer.destroy(),
            GPUResource::Sampler(_) => {}
        }
    }
}

fn safe_set<K: Into<String>>(map: &mut HashMap<String, GPUResource>, key: K, value: GPUResource) {
    let string_key = key.into();
    if let Some(current_entry) = map.get_mut(&string_key) {
        current_entry.destroy();
    }
    map.insert(string_key, value);
}

#[derive(PartialEq)]
enum ResourceMetadata {
    Indirect,
}

fn get_resource_metadata(
    _resource_commands: &HashMap<String, ResourceCommandData>,
    call_commands: &[CallCommand],
) -> HashMap<String, Vec<ResourceMetadata>> {
    let mut result: HashMap<String, Vec<ResourceMetadata>> = HashMap::new();

    for CallCommand {
        function: _,
        call_once: _,
        parameters,
    } in call_commands.iter()
    {
        if let CallCommandParameters::Indirect(buffer_name, _) = parameters {
            result
                .entry(buffer_name.clone())
                .or_default()
                .push(ResourceMetadata::Indirect);
        }
    }

    result
}

const PRINTF_BUFFER_ELEMENT_SIZE: usize = 12;
const PRINTF_BUFFER_SIZE: usize = PRINTF_BUFFER_ELEMENT_SIZE * 2048; // 12 bytes per printf struct
async fn process_resource_commands(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    resource_bindings: &HashMap<String, wgpu::BindGroupLayoutEntry>,
    resource_commands: &HashMap<String, ResourceCommandData>,
    resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
    random_pipeline: &mut ComputePipeline,
    uniform_size: u64,
) -> HashMap<String, GPUResource> {
    let mut allocated_resources: HashMap<String, GPUResource> = HashMap::new();
    let current_window_size = [300, 150]; //TODO

    safe_set(
        &mut allocated_resources,
        "uniformInput".to_string(),
        GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
        })),
    );

    let mut unprocessed_resource_commands = resource_commands.clone();
    while !unprocessed_resource_commands.is_empty() {
        let resource_commands: HashMap<String, ResourceCommandData> =
            std::mem::take(&mut unprocessed_resource_commands);

        for (resource_name, command_data) in resource_commands {
            match command_data {
                ResourceCommandData::Zeros {
                    count,
                    element_size,
                } => {
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource ${resource_name} is not defined in the bindings.");
                    };

                    if !matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
                        panic!("Resource ${resource_name} is an invalid type for ZEROS");
                    }

                    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

                    if resource_metadata
                        .get(&resource_name)
                        .unwrap_or(&vec![])
                        .contains(&ResourceMetadata::Indirect)
                    {
                        usage |= wgpu::BufferUsages::INDIRECT;
                    }

                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&resource_name),
                        mapped_at_creation: false,
                        size: (count * element_size) as u64,
                        usage,
                    });

                    // Initialize the buffer with zeros.
                    let zeros = vec![0u8; (count * element_size) as usize];
                    queue.write_buffer(&buffer, 0, &zeros);

                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        GPUResource::Buffer(buffer),
                    );
                }
                ResourceCommandData::Model { data } => {
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource ${resource_name} is not defined in the bindings.");
                    };

                    if !matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
                        panic!("Resource ${resource_name} is an invalid type for MODEL");
                    }

                    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&resource_name),
                        mapped_at_creation: false,
                        size: data.len() as u64,
                        usage,
                    });

                    // Initialize the buffer with zeros.
                    queue.write_buffer(&buffer, 0, &data);

                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        GPUResource::Buffer(buffer),
                    );
                }
                ResourceCommandData::Sampler => {
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource ${resource_name} is not defined in the bindings.");
                    };

                    if !matches!(binding_info.ty, wgpu::BindingType::Sampler { .. }) {
                        panic!("Resource ${resource_name} is an invalid type for Sampler");
                    }

                    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                        label: None,
                        ..Default::default()
                    });

                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        GPUResource::Sampler(sampler),
                    );
                }
                ResourceCommandData::Rand(count) => {
                    let element_size = 4; // RAND is only valid for floats
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource {} is not defined in the bindings.", resource_name);
                    };

                    if !matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
                        panic!("Resource ${resource_name} is an invalid type for RAND");
                    }

                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        mapped_at_creation: false,
                        size: (count * element_size) as u64,
                        usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST),
                    });

                    // Place a call to a shader that fills the buffer with random numbers.

                    // Dispatch a random number generation shader.
                    // Alloc resources for the shader.
                    let mut rand_float_resources: HashMap<String, GPUResource> = HashMap::new();

                    rand_float_resources
                        .insert("outputBuffer".to_string(), GPUResource::Buffer(buffer));

                    if !rand_float_resources.contains_key("uniformInput") {
                        rand_float_resources.insert(
                            "uniformInput".to_string(),
                            GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
                                label: None,
                                mapped_at_creation: false,
                                size: 16,
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            })),
                        );
                    }

                    // Set bindings on the pipeline.
                    random_pipeline.create_bind_group(&rand_float_resources);

                    let GPUResource::Buffer(seed_buffer) =
                        rand_float_resources.get("uniformInput").unwrap()
                    else {
                        panic!("Invalid state");
                    };
                    let mut rng = rand::rng();
                    let seed_value: &[f32] = &[rng.random::<f32>(), 0.0, 0.0, 0.0];
                    queue.write_buffer(seed_buffer, 0, bytemuck::cast_slice(seed_value));

                    // Encode commands to do the computation
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("compute builtin encoder"),
                        });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("compute builtin pass"),
                        timestamp_writes: None,
                    });

                    pass.set_bind_group(0, random_pipeline.bind_group.as_ref(), &[]);
                    pass.set_pipeline(random_pipeline.pipeline.as_ref().unwrap());

                    let size = [count, 1, 1];
                    let block_size = random_pipeline.thread_group_size.unwrap();
                    let work_group_size: Vec<u32> = size
                        .iter()
                        .zip(block_size.map(|s| s as u32))
                        .map(|(size, block_size)| size.div_ceil(block_size))
                        .collect();

                    pass.dispatch_workgroups(
                        work_group_size[0],
                        work_group_size[1],
                        work_group_size[2],
                    );
                    drop(pass);

                    // Finish encoding and submit the commands
                    let command_buffer = encoder.finish();
                    queue.submit([command_buffer]);

                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        rand_float_resources.remove("outputBuffer").unwrap(),
                    );
                }
                ResourceCommandData::Black {
                    width,
                    height,
                    format,
                } => {
                    let size = width * height;
                    let element_size = format.block_copy_size(None).unwrap();
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource {} is not defined in the bindings.", resource_name);
                    };

                    if !matches!(
                        binding_info.ty,
                        wgpu::BindingType::StorageTexture { .. }
                            | wgpu::BindingType::Texture { .. }
                    ) {
                        panic!("Resource {} is an invalid type for BLACK", resource_name);
                    }
                    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT;
                    if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
                        usage |= wgpu::TextureUsages::STORAGE_BINDING;
                    }
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        dimension: wgpu::TextureDimension::D2,
                        mip_level_count: 1,
                        sample_count: 1,
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        format,
                        usage,
                        view_formats: &[],
                    });

                    // Initialize the texture with zeros.
                    let zeros = vec![0; (size * element_size) as usize];
                    queue.write_texture(
                        texture.as_image_copy(),
                        &zeros,
                        wgpu::TexelCopyBufferLayout {
                            bytes_per_row: Some(width * element_size),
                            offset: 0,
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );

                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        GPUResource::Texture(texture),
                    );
                }
                ResourceCommandData::Black3D {
                    size_x,
                    size_y,
                    size_z,
                    format,
                } => {
                    let size = size_x * size_y * size_z;
                    let element_size = format.block_copy_size(None).unwrap();
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource {} is not defined in the bindings.", resource_name);
                    };

                    if !matches!(
                        binding_info.ty,
                        wgpu::BindingType::StorageTexture { .. }
                            | wgpu::BindingType::Texture { .. }
                    ) {
                        panic!("Resource {} is an invalid type for BLACK_3D", resource_name);
                    }
                    let mut usage =
                        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
                    if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
                        usage |= wgpu::TextureUsages::STORAGE_BINDING;
                    }
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        dimension: wgpu::TextureDimension::D3,
                        mip_level_count: 1,
                        sample_count: 1,
                        size: wgpu::Extent3d {
                            width: size_x,
                            height: size_y,
                            depth_or_array_layers: size_z,
                        },
                        format,
                        usage,
                        view_formats: &[],
                    });

                    // Initialize the texture with zeros.
                    let zeros = vec![0; (size * element_size) as usize];
                    queue.write_texture(
                        texture.as_image_copy(),
                        &zeros,
                        wgpu::TexelCopyBufferLayout {
                            bytes_per_row: Some(size_x * element_size),
                            offset: 0,
                            rows_per_image: Some(size_y),
                        },
                        wgpu::Extent3d {
                            width: size_x,
                            height: size_y,
                            depth_or_array_layers: size_z,
                        },
                    );

                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        GPUResource::Texture(texture),
                    );
                }
                ResourceCommandData::BlackScreen {
                    width_scale,
                    height_scale,
                    format,
                } => {
                    let width = (width_scale * current_window_size[0] as f32) as u32;
                    let height = (height_scale * current_window_size[1] as f32) as u32;
                    let size = width * height;
                    let element_size = format.block_copy_size(None).unwrap();
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource {} is not defined in the bindings.", resource_name);
                    };

                    if !matches!(
                        binding_info.ty,
                        wgpu::BindingType::StorageTexture { .. }
                            | wgpu::BindingType::Texture { .. }
                    ) {
                        panic!("Resource {} is an invalid type for BLACK", resource_name);
                    }
                    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::RENDER_ATTACHMENT;
                    if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
                        usage |= wgpu::TextureUsages::STORAGE_BINDING;
                    }
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&resource_name),
                        dimension: wgpu::TextureDimension::D2,
                        mip_level_count: 1,
                        sample_count: 1,
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        format,
                        usage,
                        view_formats: &[],
                    });

                    // Initialize the texture with zeros.
                    let zeros = vec![0; (size * element_size) as usize];
                    queue.write_texture(
                        texture.as_image_copy(),
                        &zeros,
                        wgpu::TexelCopyBufferLayout {
                            bytes_per_row: Some(width * element_size),
                            offset: 0,
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );
                    queue.submit(None);

                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        GPUResource::Texture(texture),
                    );
                }
                ResourceCommandData::Url {
                    data,
                    width,
                    height,
                    format,
                } => {
                    // Load image from URL and wait for it to be ready.
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource {} is not defined in the bindings.", resource_name);
                    };

                    let element_size = format.block_copy_size(None).unwrap();

                    if !matches!(binding_info.ty, wgpu::BindingType::Texture { .. }) {
                        panic!("Resource ${resource_name} is not a texture.");
                    }
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        dimension: wgpu::TextureDimension::D2,
                        mip_level_count: 1,
                        sample_count: 1,
                        view_formats: &[],
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    });
                    queue.write_texture(
                        texture.as_image_copy(),
                        &data,
                        wgpu::TexelCopyBufferLayout {
                            bytes_per_row: Some(width * element_size),
                            offset: 0,
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );
                    safe_set(
                        &mut allocated_resources,
                        resource_name,
                        GPUResource::Texture(texture),
                    );
                }
                ref command @ ResourceCommandData::RebindForDraw {
                    ref original_resource,
                } => {
                    let Some(binding_info) = resource_bindings.get(&resource_name) else {
                        panic!("Resource {} is not defined in the bindings.", resource_name);
                    };

                    if matches!(binding_info.ty, wgpu::BindingType::Texture { .. }) {
                        let Some(GPUResource::Texture(tex)) =
                            allocated_resources.get(original_resource)
                        else {
                            unprocessed_resource_commands
                                .insert(resource_name.to_string(), command.clone());
                            continue;
                        };

                        let texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            dimension: wgpu::TextureDimension::D2,
                            mip_level_count: 1,
                            sample_count: 1,
                            size: wgpu::Extent3d {
                                width: tex.width(),
                                height: tex.height(),
                                depth_or_array_layers: 1,
                            },
                            format: tex.format(),
                            usage: tex.usage(),
                            view_formats: &[],
                        });
                        safe_set(
                            &mut allocated_resources,
                            resource_name,
                            GPUResource::Texture(texture),
                        );
                    } else if matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
                        let Some(GPUResource::Buffer(buf)) =
                            allocated_resources.get(original_resource)
                        else {
                            unprocessed_resource_commands
                                .insert(resource_name.to_string(), command.clone());
                            continue;
                        };

                        let buffer = device.create_buffer(&BufferDescriptor {
                            label: None,
                            size: buf.size(),
                            usage: buf.usage(),
                            mapped_at_creation: false,
                        });
                        safe_set(
                            &mut allocated_resources,
                            resource_name,
                            GPUResource::Buffer(buffer),
                        );
                    } else {
                        panic!(
                            "Resource {} is an invalid type for REBIND_FOR_DRAW",
                            resource_name
                        )
                    }
                }
                ResourceCommandData::Slider {
                    default,
                    element_size,
                    offset,
                    ..
                } => {
                    let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                    else {
                        panic!("cannot get uniforms")
                    };
                    // Initialize the buffer with zeros.
                    let buffer_default = if element_size == 4 {
                        default.to_le_bytes()
                    } else {
                        panic!("Unsupported float size for slider")
                    };
                    queue.write_buffer(buffer, offset as u64, &buffer_default);
                }
                ResourceCommandData::ColorPick {
                    default,
                    element_size,
                    offset,
                } => {
                    let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                    else {
                        panic!("cannot get uniforms")
                    };
                    // Initialize the buffer with zeros.
                    let buffer_default = if element_size == 4 {
                        bytemuck::cast_slice(&default)
                    } else {
                        panic!("Unsupported float size for color pick")
                    };
                    queue.write_buffer(buffer, offset as u64, buffer_default);
                }
                ResourceCommandData::MousePosition { offset } => {
                    let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                    else {
                        panic!("cannot get uniforms")
                    };
                    // Initialize the buffer with zeros.
                    let buffer_default = [0u8; 16];
                    queue.write_buffer(buffer, offset as u64, &buffer_default);
                }
                ResourceCommandData::Time { offset } => {
                    let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                    else {
                        panic!("cannot get uniforms")
                    };
                    // Initialize the buffer with current time.
                    let time = 0.0f32;
                    let buffer_default = time.to_le_bytes();
                    queue.write_buffer(buffer, offset as u64, &buffer_default);
                }
                ResourceCommandData::DeltaTime { offset } => {
                    let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                    else {
                        panic!("cannot get uniforms")
                    };
                    // Initialize the buffer with zeros.
                    let buffer_default = 0f32.to_le_bytes();
                    queue.write_buffer(buffer, offset as u64, &buffer_default);
                }
                ResourceCommandData::KeyInput { offset, .. } => {
                    let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                    else {
                        panic!("cannot get uniforms")
                    };
                    // Initialize with key released state (0.0)
                    let value = 0.0f32;
                    let slice = [value];
                    let uniform_data = bytemuck::cast_slice(&slice);
                    queue.write_buffer(buffer, offset as u64, uniform_data);
                }
            }
        }
    }

    //
    // Some special-case allocations
    //
    safe_set(
        &mut allocated_resources,
        "g_printedBuffer".to_string(),
        GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size: PRINTF_BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC),
        })),
    );

    safe_set(
        &mut allocated_resources,
        "printfBufferRead".to_string(),
        GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size: PRINTF_BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
        })),
    );

    allocated_resources
}

enum FormatSpecifier {
    Text(String),
    Specifier {
        flags: String,
        width: Option<usize>,
        precision: Option<usize>,
        specifier_type: String,
    },
}

fn parse_printf_format(format_string: String) -> Vec<FormatSpecifier> {
    let mut format_specifiers: Vec<FormatSpecifier> = vec![];
    let regex = Regex::new(r"%([-+ #0]*)(\d+)?(\.\d+)?([diufFeEgGxXosc])").unwrap();
    let mut last_index = 0;

    let re_matches = regex.captures_iter(&format_string);
    for re_match in re_matches {
        let whole_match = re_match.get(0).unwrap();
        let literal_text = &format_string[last_index..whole_match.start()];

        // Add literal text before the match as a token, if any
        format_specifiers.push(FormatSpecifier::Text(literal_text.to_string()));

        let precision = re_match
            .get(3)
            .map(|precision| precision.as_str()[1..].parse::<usize>().unwrap()); // remove leading '.'

        let width_number = re_match
            .get(2)
            .map(|width| width.as_str().parse::<usize>().unwrap());

        // Add the format specifier as a token
        format_specifiers.push(FormatSpecifier::Specifier {
            flags: re_match
                .get(1)
                .map(|m| m.as_str())
                .unwrap_or("")
                .to_string(),
            width: width_number,
            precision,
            specifier_type: re_match.get(4).unwrap().as_str().to_string(),
        });

        last_index = whole_match.end();
    }

    // Add any remaining literal text after the last match
    if last_index < format_string.len() {
        format_specifiers.push(FormatSpecifier::Text(
            format_string[last_index..].to_string(),
        ));
    }

    format_specifiers
}

fn format_printf_string(parsed_tokens: &[FormatSpecifier], data: &[String]) -> String {
    let mut data_index = 0;

    parsed_tokens
        .iter()
        .map(|token| match &token {
            FormatSpecifier::Text(value) => value.clone(),
            FormatSpecifier::Specifier {
                flags,
                width,
                precision,
                specifier_type,
            } => {
                let value = data[data_index].clone();
                data_index += 1;
                format_specifier(value, flags, width, precision, specifier_type)
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

// Helper function to format each specifier
fn format_specifier(
    value: String,
    flags: &str,
    width: &Option<usize>,
    precision: &Option<usize>,
    specifier_type: &str,
) -> String {
    let mut formatted_value;
    let was_precision_specified = precision.is_some();
    let precision = precision.unwrap_or(6); //eww magic number
    match specifier_type {
        "d" | "i" => {
            // Integer (decimal)
            formatted_value = value.parse::<i32>().unwrap().to_string();
        }
        "u" => {
            // Unsigned integer
            formatted_value = value.parse::<u32>().unwrap().to_string();
        }
        "o" => {
            // Octal
            formatted_value = format!("{:o}", value.parse::<u32>().unwrap());
        }
        "x" => {
            // Hexadecimal (lowercase)
            formatted_value = format!("{:x}", value.parse::<u32>().unwrap());
        }
        "X" => {
            // Hexadecimal (uppercase)
            formatted_value = format!("{:X}", value.parse::<u32>().unwrap());
        }
        "f" | "F" => {
            // Floating-point
            formatted_value = format!("{:.1$}", value.parse::<f32>().unwrap(), precision);
        }
        "e" => {
            // Scientific notation (lowercase)
            formatted_value = format!("{:.1$e}", value.parse::<f32>().unwrap(), precision);
        }
        "E" => {
            // Scientific notation (uppercase)
            formatted_value = format!("{:.1$E}", value.parse::<f32>().unwrap(), precision);
        }
        "g" | "G" => {
            // Shortest representation of floating-point
            formatted_value = format!("{:.1$}", value.parse::<f32>().unwrap(), precision);
        }
        "c" => {
            // Character
            formatted_value = String::from(value.parse::<u8>().unwrap() as char);
        }
        "s" => {
            // String
            formatted_value = value.clone();
            if was_precision_specified {
                formatted_value = formatted_value[0..precision].to_string();
            }
        }
        "%" => {
            // Literal "%"
            return "%".to_string();
        }
        st => panic!("Unsupported specifier: {}", st),
    }

    // Handle width and flags (like zero-padding, space, left alignment, sign)
    if let Some(width) = width {
        let padding_char = if flags.contains('0') && !flags.contains('-') {
            '0'
        } else {
            ' '
        };
        let is_left_aligned = flags.contains('-');
        let needs_sign = flags.contains('+') && value.parse::<f32>().unwrap() >= 0.0;
        let needs_space =
            flags.contains(' ') && !needs_sign && value.parse::<f32>().unwrap() >= 0.0;

        if needs_sign {
            formatted_value = format!("+{formatted_value}");
        } else if needs_space {
            formatted_value = format!(" {formatted_value}");
        }

        if formatted_value.len() < *width {
            let padding = padding_char
                .to_string()
                .repeat(width - formatted_value.len());
            formatted_value = if is_left_aligned {
                format!("{formatted_value}{padding}")
            } else {
                format!("{padding}{formatted_value}")
            };
        }
    }

    formatted_value
}

fn parse_printf_buffer(
    hashed_strings: &HashMap<u32, String>,
    printf_value_resource: &wgpu::Buffer,
    buffer_element_size: usize,
) -> Vec<String> {
    // Read the printf buffer
    let mapped_range = printf_value_resource.slice(..).get_mapped_range();
    let printf_buffer_array: &[u32] = bytemuck::cast_slice(&mapped_range);

    let number_elements = printf_buffer_array.len() * 4 / buffer_element_size;

    // TODO: We currently doesn't support 64-bit data type (e.g. uint64_t, int64_t, double, etc.)
    // so 32-bit array should be able to contain everything we need.
    let mut data_array = vec![];
    let element_size_in_words = buffer_element_size / 4;
    let mut out_str_arry: Vec<String> = vec![];
    let mut format_string = "".to_string();
    for element_index in 0..number_elements {
        let offset = element_index * element_size_in_words;
        match printf_buffer_array[offset] {
            1 => {
                // format string
                format_string = hashed_strings
                    .get(&printf_buffer_array[offset + 1])
                    .unwrap()
                    .clone();
                // low field
            }
            2 => {
                // normal string
                data_array.push(
                    hashed_strings
                        .get(&printf_buffer_array[offset + 1])
                        .unwrap()
                        .clone(),
                ); // low field
            }
            3 => {
                // integer
                data_array.push(printf_buffer_array[offset + 1].to_string()); // low field
            }
            4 => {
                // float
                let float_data = f32::from_bits(printf_buffer_array[offset + 1]);
                data_array.push(float_data.to_string()); // low field
            }
            5 => {
                // TODO: We can't handle 64-bit data type yet.
                data_array.push(0.to_string()); // low field
            }
            0xFFFFFFFF => {
                let parsed_tokens = parse_printf_format(format_string);
                let output = format_printf_string(&parsed_tokens, &data_array);
                out_str_arry.push(output);
                format_string = "".to_string();
                data_array = vec![];
                if element_index < number_elements - 1 {
                    let next_offset = offset + element_size_in_words;
                    // advance to the next element to see if it's a format string, if it's not we just early return
                    // the results, otherwise just continue processing.
                    if printf_buffer_array[next_offset] != 1
                    // type field
                    {
                        return out_str_arry;
                    }
                }
            }
            _ => panic!("Invalid format type!"),
        }
    }

    if !format_string.is_empty() {
        // If we are here, it means that the printf buffer is used up, and we are in the middle of processing
        // one printf string, so we are still going to format it, even though there could be some data missing, which
        // will be shown as 'undef'.
        let parsed_tokens = parse_printf_format(format_string);
        let output = format_printf_string(&parsed_tokens, &data_array);
        out_str_arry.push(output);
        out_str_arry.push("Print buffer is out of boundary, some data is missing!!!".to_string());
    }

    out_str_arry
}

impl State {
    async fn new(window: Arc<Window>, compilation: CompilationResult) -> State {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    ..Default::default()
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let device = Arc::new(device);

        let size = window.inner_size();

        let surface = instance.create_surface(window.clone()).unwrap();
        let surface_format = wgpu::TextureFormat::Rgba8Unorm;

        let mut random_pipeline = ComputePipeline::new(device.clone());

        // Load randFloat shader code using the proc macro
        let compiled_result: CompilationResult =
            compile_shader!("rand_float.slang", ["src/shaders"]);

        let rand_code = compiled_result.out_code;
        let rand_group_size = compiled_result
            .entry_group_sizes
            .get("computeMain")
            .unwrap();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rand float"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&rand_code)),
        });

        random_pipeline.set_thread_group_size(*rand_group_size);
        random_pipeline.create_pipeline_layout(compiled_result.bindings);

        // Create the pipeline (without resource bindings for now)
        random_pipeline.create_pipeline(&module, None, None);

        let resource_metadata =
            get_resource_metadata(&compilation.resource_commands, &compilation.call_commands);

        let allocated_resources = process_resource_commands(
            &queue,
            &device,
            &compilation.bindings,
            &compilation.resource_commands,
            &resource_metadata,
            &mut random_pipeline,
            compilation.uniform_size,
        )
        .await;

        let mut compute_pipelines = HashMap::new();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&compilation.out_code)),
        });
        for (shader_name, thread_group_size) in compilation.entry_group_sizes {
            let mut pipeline = ComputePipeline::new(device.clone());

            // Filter out indirect buffer bindings if this pipeline uses them for dispatch
            let mut pipeline_bindings = compilation.bindings.clone();
            for call_command in &compilation.call_commands {
                if call_command.function == shader_name {
                    if let CallCommandParameters::Indirect(buffer_name, _) =
                        &call_command.parameters
                    {
                        pipeline_bindings.remove(buffer_name);
                    }
                }
            }

            pipeline.create_pipeline_layout(pipeline_bindings);
            pipeline.create_pipeline(&module, Some(&allocated_resources), Some(&shader_name));
            pipeline.set_thread_group_size(thread_group_size);
            compute_pipelines.insert(shader_name, pipeline);
        }

        let mut draw_pipelines = Vec::new();
        for draw_command in compilation.draw_commands.iter() {
            let mut pipeline = DrawPipeline::new(device.clone());
            pipeline.create_pipeline_layout(compilation.bindings.clone());
            pipeline.create_pipeline(
                &module,
                Some(&allocated_resources),
                &compilation.resource_commands,
                Some(&draw_command.vertex_entrypoint),
                Some(&draw_command.fragment_entrypoint),
            );
            draw_pipelines.push(pipeline);
        }

        let state = State {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            compute_pipelines,
            draw_pipelines,
            bindings: compilation.bindings,
            resource_commands: compilation.resource_commands,
            call_commands: compilation.call_commands,
            draw_commands: compilation.draw_commands,
            uniform_components: Rc::new(RefCell::new(compilation.uniform_controllers)),
            uniform_size: compilation.uniform_size,
            hashed_strings: compilation.hashed_strings,
            allocated_resources,
            mouse_state: MouseState {
                last_mouse_clicked_pos: PhysicalPosition::default(),
                last_mouse_down_pos: PhysicalPosition::default(),
                current_mouse_pos: PhysicalPosition::default(),
                mouse_clicked: false,
                is_mouse_down: false,
            },
            keyboard_state: KeyboardState::new(),
            print_receiver: None,
            first_frame: true,
            delta_time: 0.0,
            last_frame_time: web_time::Instant::now(),
            launch_time: web_time::Instant::now(),
        };

        // Configure surface for the first time
        state.configure_surface();

        state
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            // Request compatibility with the sRGB-format texture view we‘re going to create later.
            view_formats: vec![self.surface_format],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if cfg!(target_arch = "wasm32") {
                wgpu::PresentMode::Fifo
            } else {
                wgpu::PresentMode::Immediate
            },
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;

        // reconfigure the surface
        self.configure_surface();

        for (resource_name, command_data) in self.resource_commands.iter() {
            let ResourceCommandData::BlackScreen {
                format,
                width_scale,
                height_scale,
            } = command_data
            else {
                continue;
            };
            let width = (width_scale * new_size.width as f32) as u32;
            let height = (height_scale * new_size.height as f32) as u32;
            let size = width * height;
            let element_size = format.block_copy_size(None).unwrap();
            let Some(binding_info) = self.bindings.get(resource_name) else {
                panic!("Resource {} is not defined in the bindings.", resource_name);
            };

            if !matches!(
                binding_info.ty,
                wgpu::BindingType::StorageTexture { .. } | wgpu::BindingType::Texture { .. }
            ) {
                panic!("Resource {} is an invalid type for BLACK", resource_name);
            }
            let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT;
            if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
                usage |= wgpu::TextureUsages::STORAGE_BINDING;
            }
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(resource_name),
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                format: *format,
                usage,
                view_formats: &[],
            });

            // Initialize the texture with zeros.
            let zeros = vec![0; (size * element_size) as usize];
            self.queue.write_texture(
                texture.as_image_copy(),
                &zeros,
                wgpu::TexelCopyBufferLayout {
                    bytes_per_row: Some(width * element_size),
                    offset: 0,
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            let mut encoder = self.device.create_command_encoder(&Default::default());
            // copy old texture to new texture
            let Some(GPUResource::Texture(old_texture)) =
                self.allocated_resources.get(resource_name)
            else {
                panic!("Resource {} is not a Texture", resource_name);
            };
            encoder.copy_texture_to_texture(
                old_texture.as_image_copy(),
                texture.as_image_copy(),
                wgpu::Extent3d {
                    width: width.min(old_texture.width()),
                    height: height.min(old_texture.height()),
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(Some(encoder.finish()));

            safe_set(
                &mut self.allocated_resources,
                resource_name.to_string(),
                GPUResource::Texture(texture),
            );
        }
        for (_, compute_pipeline) in self.compute_pipelines.iter_mut() {
            compute_pipeline.create_bind_group(&self.allocated_resources);
        }
        for draw_pipeline in self.draw_pipelines.iter_mut() {
            draw_pipeline.create_bind_group(&self.allocated_resources, &self.resource_commands);
        }
    }

    fn render(&mut self) {
        if let Some(receiver) = self.print_receiver.as_mut() {
            let mut print_received = false;
            if let Ok(Some(_)) = receiver.try_recv() {
                let Some(GPUResource::Buffer(printf_buffer_read)) =
                    self.allocated_resources.get("printfBufferRead")
                else {
                    panic!("printfBufferRead is incorrect type or doesn't exist");
                };

                let format_print = parse_printf_buffer(
                    &self.hashed_strings,
                    printf_buffer_read,
                    PRINTF_BUFFER_ELEMENT_SIZE,
                );

                if !format_print.is_empty() {
                    let result = format!("Shader Output:\n{}\n", format_print.join(""));
                    #[cfg(not(target_arch = "wasm32"))]
                    print!("{}", result);
                    #[cfg(target_arch = "wasm32")]
                    web_sys::console::log_1(&result.into())
                }

                printf_buffer_read.unmap();

                print_received = true;
            }
            if print_received {
                self.print_receiver = None;
            } else {
                return;
            }
        }

        // Create texture view
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("failed to acquire next swapchain texture");
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                // Without add_srgb_suffix() the image we will be working with
                // might not be "gamma correct".
                format: Some(self.surface_format),
                ..Default::default()
            });

        let Some(GPUResource::Buffer(uniform_input)) = self.allocated_resources.get("uniformInput")
        else {
            panic!("uniformInput doesn't exist or is of incorrect type");
        };

        let mut buffer_data: Vec<u8> = vec![0; self.uniform_size as usize];

        // Calculate delta time
        let now = web_time::Instant::now();
        let frame_time = now - self.last_frame_time;
        self.delta_time = frame_time.as_secs_f32();
        self.last_frame_time = now;

        let uniform_source_data = UniformSourceData {
            launch_time: self.launch_time,
            delta_time: self.delta_time,
            last_mouse_down_pos: [self.mouse_state.last_mouse_down_pos.x as f32, self.mouse_state.last_mouse_down_pos.y as f32],
            last_mouse_clicked_pos: [self.mouse_state.last_mouse_clicked_pos.x as f32, self.mouse_state.last_mouse_clicked_pos.y as f32],
            mouse_down: self.mouse_state.is_mouse_down,
            mouse_clicked: self.mouse_state.mouse_clicked,
            pressed_keys: &self.keyboard_state.pressed_keys,
        };

        for UniformController {
            buffer_offset,
            controller,
            ..
        } in self.uniform_components.borrow().iter() {
            let slice = controller.get_data(&uniform_source_data);
            let uniform_data = bytemuck::cast_slice(&slice);
            buffer_data[*buffer_offset..(buffer_offset + uniform_data.len())].copy_from_slice(uniform_data);
        }
        
        self.queue.write_buffer(uniform_input, 0, &buffer_data);

        let mut encoder = self.device.create_command_encoder(&Default::default());

        for call_command in self.call_commands.iter() {
            if !self.first_frame && call_command.call_once {
                continue;
            }
            let pipeline = self
                .compute_pipelines
                .get(call_command.function.as_str())
                .unwrap();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute builtin pass"),
                timestamp_writes: None,
            });

            pass.set_bind_group(0, pipeline.bind_group.as_ref(), &[]);
            pass.set_pipeline(pipeline.pipeline.as_ref().unwrap());

            use slang_compiler_type_definitions::CallCommandParameters;

            match &call_command.parameters {
                CallCommandParameters::ResourceBased(resource_name, element_size) => {
                    let resource = self.allocated_resources.get(resource_name).unwrap();
                    let size = match resource {
                        GPUResource::Texture(texture) => [
                            texture.width(),
                            texture.height(),
                            texture.depth_or_array_layers(),
                        ],
                        GPUResource::Buffer(buffer) => {
                            [buffer.size() as u32 / element_size.unwrap_or(4), 1, 1]
                        }
                        GPUResource::Sampler(_) => panic!("Sampler doesn't have size"),
                    };
                    let block_size = pipeline.thread_group_size.unwrap();

                    let work_group_size: Vec<u32> = size
                        .iter()
                        .zip(block_size.map(|s| s as u32))
                        .map(|(size, block_size)| size.div_ceil(block_size))
                        .collect();

                    pass.dispatch_workgroups(
                        work_group_size[0],
                        work_group_size[1],
                        work_group_size[2],
                    );
                }
                CallCommandParameters::FixedSize(items) => {
                    if items.len() > 3 {
                        panic!("Too many parameters for call command");
                    }
                    let mut size = [1; 3];
                    for (i, n) in items.iter().enumerate() {
                        size[i] = *n;
                    }
                    let block_size = pipeline.thread_group_size.unwrap();

                    let work_group_size: Vec<u32> = size
                        .iter()
                        .zip(block_size.map(|s| s as u32))
                        .map(|(size, block_size)| size.div_ceil(block_size))
                        .collect();

                    pass.dispatch_workgroups(
                        work_group_size[0],
                        work_group_size[1],
                        work_group_size[2],
                    );
                }
                CallCommandParameters::Indirect(indirect_buffer, offset) => {
                    let Some(GPUResource::Buffer(resource)) =
                        self.allocated_resources.get(indirect_buffer)
                    else {
                        panic!("Could not get indirect buffer");
                    };
                    pass.dispatch_workgroups_indirect(resource, *offset as u64);
                }
            };
            drop(pass);
        }

        if self.compute_pipelines.contains_key("printMain") {
            let Some(GPUResource::Buffer(printf_buffer_read)) =
                self.allocated_resources.get("printfBufferRead")
            else {
                panic!("printfBufferRead is incorrect type or doesn't exist");
            };
            encoder.clear_buffer(printf_buffer_read, 0, None);
            let Some(GPUResource::Buffer(g_printed_buffer)) =
                self.allocated_resources.get("g_printedBuffer")
            else {
                panic!("g_printedBuffer is not a buffer");
            };
            encoder.copy_buffer_to_buffer(
                g_printed_buffer,
                0,
                printf_buffer_read,
                0,
                g_printed_buffer.size(),
            );
        }

        let mut pass = DrawPipeline::begin_render_pass(
            &self.device,
            Extent3d {
                width: surface_texture.texture.width(),
                height: surface_texture.texture.height(),
                depth_or_array_layers: 1,
            },
            &mut encoder,
            &texture_view,
        );
        for (draw_command, pipeline) in self
            .draw_commands
            .iter()
            .zip(self.draw_pipelines.iter_mut())
        {
            pass.set_bind_group(0, pipeline.bind_group.as_ref(), &[]);
            pass.set_pipeline(pipeline.pipeline.as_ref().unwrap());

            pass.draw(0..draw_command.vertex_count, 0..1);
        }
        drop(pass);

        // Submit the command in the queue to execute
        self.queue.submit([encoder.finish()]);

        if self.compute_pipelines.contains_key("printMain") {
            let Some(GPUResource::Buffer(printf_buffer_read)) =
                self.allocated_resources.get("printfBufferRead")
            else {
                panic!("printfBufferRead is incorrect type or doesn't exist");
            };
            let (sender, receiver) = futures::channel::oneshot::channel();
            self.print_receiver = Some(receiver);
            printf_buffer_read
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    r.unwrap();
                    sender.send(()).unwrap()
                });
            self.device.poll(wgpu::PollType::Wait).unwrap();
        }
        surface_texture.present();

        self.first_frame = false;
    }

    fn mousedown(&mut self) {
        self.mouse_state.last_mouse_clicked_pos = self.mouse_state.current_mouse_pos;
        self.mouse_state.mouse_clicked = true;
        self.mouse_state.is_mouse_down = true;
    }

    fn mousemove(&mut self, position: PhysicalPosition<f64>) {
        self.mouse_state.current_mouse_pos = position;
        if self.mouse_state.is_mouse_down {
            self.mouse_state.last_mouse_down_pos = position;
        }
    }

    fn mouseup(&mut self) {
        self.mouse_state.is_mouse_down = false;
    }

    fn key_pressed(&mut self, key: Key) {
        self.keyboard_state.key_pressed(key);
    }

    fn key_released(&mut self, key: Key) {
        self.keyboard_state.key_released(key);
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct DebugPanel {
    uniform_controllers: Rc<RefCell<Vec<UniformController>>>,
    last_frame_time: web_time::Instant,
    last_debug_frame_time: web_time::Instant,
    current_fps: f32,
    frame_time_samples: Vec<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct DebugAppState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub surface: wgpu::Surface<'static>,
    pub scale_factor: f32,
    pub egui_renderer: egui_tools::EguiRenderer,
    debug_panel: DebugPanel,
    window: Arc<Window>,
}

#[cfg(not(target_arch = "wasm32"))]
impl DebugAppState {
    async fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        window: Arc<Window>,
        width: u32,
        height: u32,
        #[cfg(not(target_arch = "wasm32"))] debug_panel: DebugPanel,
    ) -> Self {
        let power_pref = wgpu::PowerPreference::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let features = wgpu::Features::empty();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: Default::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let selected_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let swapchain_format = swapchain_capabilities
            .formats
            .iter()
            .find(|d| **d == selected_format)
            .expect("failed to select proper surface texture format!");

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: *swapchain_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo, // Use vsync for debug window
            desired_maximum_frame_latency: 0,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &surface_config);

        let egui_renderer =
            egui_tools::EguiRenderer::new(&device, surface_config.format, None, 1, &window);

        let scale_factor = 1.0;

        Self {
            device,
            queue,
            surface,
            surface_config,
            egui_renderer,
            scale_factor,
            #[cfg(not(target_arch = "wasm32"))]
            debug_panel,
            window,
        }
    }

    fn resize_surface(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn handle_input(&mut self, event: &WindowEvent) {
        self.egui_renderer.handle_input(&self.window, event);
    }
}

struct App {
    state: Option<State>,
    #[cfg(target_arch = "wasm32")]
    state_receiver: Option<futures::channel::oneshot::Receiver<State>>,
    #[cfg(not(target_arch = "wasm32"))]
    debug_app: Option<DebugAppState>,
    compilation: Option<CompilationResult>,
}
impl App {
    fn new(compilation: CompilationResult) -> Self {
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            state_receiver: None,
            #[cfg(not(target_arch = "wasm32"))]
            debug_app: None,
            compilation: Some(compilation),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn set_debug_window(&mut self, window: Window) {
        let window = Arc::new(window);
        let initial_width = 1360;
        let initial_height = 768;

        let _ =
            window.request_inner_size(winit::dpi::PhysicalSize::new(initial_width, initial_height));

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface!");

        #[cfg(not(target_arch = "wasm32"))]
        let debug_panel = DebugPanel {
            uniform_controllers: self.state.as_ref().unwrap().uniform_components.clone(),
            last_frame_time: web_time::Instant::now(),
            last_debug_frame_time: web_time::Instant::now(),
            current_fps: 0.0,
            frame_time_samples: Vec::new(),
        };

        let debug_state = DebugAppState::new(
            &instance,
            surface,
            window,
            initial_width,
            initial_width,
            #[cfg(not(target_arch = "wasm32"))]
            debug_panel,
        )
        .await;

        self.debug_app.get_or_insert(debug_state);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_resized(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.debug_app
                .as_mut()
                .unwrap()
                .resize_surface(width, height);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_redraw(&mut self) {
        let state = self.debug_app.as_mut().unwrap();

        // Attempt to handle minimizing window
        if let Some(min) = state.window.is_minimized() {
            if min {
                println!("Window is minimized");
                return;
            }
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [state.surface_config.width, state.surface_config.height],
            pixels_per_point: state.window.scale_factor() as f32 * state.scale_factor,
        };

        let surface_texture = state.surface.get_current_texture();

        match surface_texture {
            Err(wgpu::SurfaceError::Outdated) => {
                // Ignoring outdated to allow resizing and minimization
                println!("wgpu surface outdated");
                return;
            }
            Err(_) => {
                surface_texture.expect("Failed to acquire next swap chain texture");
                return;
            }
            Ok(_) => {}
        };

        let surface_texture = surface_texture.unwrap();

        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let window = state.window.as_ref();

        {
            state.egui_renderer.begin_frame(window);
            egui::CentralPanel::default().show(state.egui_renderer.context(), |ui| {
                ui.heading(format!("FPS: {:.1}", state.debug_panel.current_fps));
                ui.separator();
                ui.heading("Uniforms:");

                for UniformController {
                    name, controller, ..
                } in state
                    .debug_panel
                    .uniform_controllers
                    .borrow_mut()
                    .iter_mut()
                {
                    controller.render(name, ui);
                }
            });

            state.egui_renderer.end_frame_and_draw(
                &state.device,
                &state.queue,
                &mut encoder,
                window,
                &surface_view,
                screen_descriptor,
            );
        }

        state.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut builder = Window::default_attributes().with_title("Slang Native Playground");

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .expect("error window")
                .document()
                .expect("error document")
                .get_element_by_id("canvas")
                .expect("could not find id canvas")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("error HtmlCanvasElement");
            builder = builder.with_decorations(false).with_canvas(Some(canvas));
        }
        // Create window object
        let window = Arc::new(event_loop.create_window(builder).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            let state =
                pollster::block_on(State::new(window.clone(), self.compilation.take().unwrap()));
            self.state = Some(state);
        }

        #[cfg(target_arch = "wasm32")]
        {
            let (sender, receiver) = futures::channel::oneshot::channel();
            self.state_receiver = Some(receiver);
            let compilation = self.compilation.take().unwrap();
            wasm_bindgen_futures::spawn_local(async move {
                let mut state = State::new(window.clone(), compilation).await;
                state.resize(state.window.inner_size());
                if sender.send(state).is_err() {
                    panic!("Failed to create and send renderer!");
                }
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        if cfg!(debug_assertions) {
            let debug_window = event_loop
                .create_window(
                    Window::default_attributes().with_title("Slang Native Playground Debug"),
                )
                .unwrap();
            pollster::block_on(self.set_debug_window(debug_window));
            self.state.as_ref().unwrap().window.focus_window();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        if self.state.is_none() {
            let mut renderer_received = false;
            if let Some(receiver) = self.state_receiver.as_mut() {
                if let Ok(Some(state)) = receiver.try_recv() {
                    self.state = Some(state);
                    renderer_received = true;
                }
            }
            if renderer_received {
                self.state_receiver = None;
            } else {
                return;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self
            .debug_app
            .as_ref()
            .map(|w| w.window.id())
            .map(|w_id| w_id == _id)
            .unwrap_or(false)
        {
            self.debug_app.as_mut().unwrap().handle_input(&event);

            match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {}
                WindowEvent::Resized(new_size) => {
                    self.handle_resized(new_size.width, new_size.height);
                }
                _ => (),
            }
            return;
        }
        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            #[cfg(target_arch = "wasm32")]
            WindowEvent::RedrawRequested => {
                let state = self.state.as_mut().unwrap();
                state.render();
            }
            WindowEvent::Resized(size) => {
                state.resize(size);
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                state.mousemove(position);
            }
            WindowEvent::MouseInput {
                device_id: _,
                state: mouse_state,
                button,
            } => {
                if button == MouseButton::Left {
                    if mouse_state == ElementState::Pressed {
                        state.mousedown();
                    } else {
                        state.mouseup();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event,
                device_id: _,
                is_synthetic: _,
            } => {
                let keycode = remove_modifiers(event.logical_key);
                match event.state {
                    ElementState::Pressed => state.key_pressed(keycode),
                    ElementState::Released => state.key_released(keycode),
                }
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        if self.state.is_none() {
            let mut renderer_received = false;
            if let Some(receiver) = self.state_receiver.as_mut() {
                if let Ok(Some(state)) = receiver.try_recv() {
                    self.state = Some(state);
                    renderer_received = true;
                }
            }
            if renderer_received {
                self.state_receiver = None;
            } else {
                return;
            }
        }
        let state = self.state.as_mut().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        state.render();
        #[cfg(target_arch = "wasm32")]
        state.window.request_redraw();

        #[cfg(not(target_arch = "wasm32"))]
        // Only handle debug window if in debug mode
        if cfg!(debug_assertions) {
            let debug_state = self.debug_app.as_mut().unwrap();

            // Calculate time since last frame
            let now = web_time::Instant::now();
            let frame_time = now - debug_state.debug_panel.last_frame_time;

            debug_state
                .debug_panel
                .frame_time_samples
                .push(frame_time.as_secs_f32());

            // Keep a rolling average of the last 60 frames
            if debug_state.debug_panel.frame_time_samples.len() > 60 {
                debug_state.debug_panel.frame_time_samples.remove(0);
            }

            // Calculate average FPS
            debug_state.debug_panel.current_fps = debug_state.debug_panel.frame_time_samples.len()
                as f32
                / debug_state
                    .debug_panel
                    .frame_time_samples
                    .iter()
                    .sum::<f32>();

            let debug_frame_time = now - debug_state.debug_panel.last_debug_frame_time;

            // Target 60 FPS (16.67ms per frame)
            let target_frame_time: std::time::Duration =
                std::time::Duration::from_secs_f32(1.0 / 60.0);
            debug_state.debug_panel.last_frame_time = now;

            // Only redraw if enough time has passed since last frame
            if debug_frame_time >= target_frame_time {
                debug_state.debug_panel.last_debug_frame_time = now;

                self.handle_redraw();
            }
        }
    }
}

fn remove_modifiers(key: Key<SmolStr>) -> Key<SmolStr> {
    match key {
        Key::Character(c) => Key::Character(c.to_lowercase().into()),
        k => k,
    }
}

pub fn launch(compilation: CompilationResult) {
    #[cfg(debug_assertions)]
    #[cfg(target_family = "wasm")]
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    // wgpu uses `log` for all of our logging, so we initialize a logger with the `env_logger` crate.
    //
    // To change the log level, set the `RUST_LOG` environment variable. See the `env_logger`
    // documentation for more information.
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    // When the current loop iteration finishes, immediately begin a new
    // iteration regardless of whether or not new events are available to
    // process. Preferred for applications that want to render as fast as
    // possible, like games.
    event_loop.set_control_flow(ControlFlow::Poll);

    // When the current loop iteration finishes, suspend the thread until
    // another event arrives. Helps keeping CPU utilization low if nothing
    // is happening, which is preferred if the application might be idling in
    // the background.
    // event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(compilation);
    event_loop.run_app(&mut app).unwrap();
}
