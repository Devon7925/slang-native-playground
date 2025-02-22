mod compute_pipeline;
mod egui_tools;
mod graphics_pipeline;
mod slang_compiler;

use egui::Slider;
use egui_tools::EguiRenderer;
use egui_wgpu::ScreenDescriptor;
use rand::Rng;
use regex::Regex;
use slang_compiler::{
    CallCommand, CompilationResult, ResourceCommand, ResourceCommandData, UniformController,
    UniformControllerType,
};
use tokio::runtime;
use url::{ParseError, Url};
use wgpu::{Features, SurfaceError};

use std::{
    borrow::Cow,
    cell::RefCell,
    collections::HashMap,
    fs::read,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use compute_pipeline::ComputePipeline;
use graphics_pipeline::GraphicsPipeline;
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct MouseState {
    last_mouse_clicked_pos: PhysicalPosition<f64>,
    last_mouse_down_pos: PhysicalPosition<f64>,
    current_mouse_pos: PhysicalPosition<f64>,
    mouse_clicked: bool,
    is_mouse_down: bool,
}

struct State {
    window: Arc<Window>,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    pass_through_pipeline: GraphicsPipeline,
    compute_pipelines: HashMap<String, ComputePipeline>,
    bindings: HashMap<String, wgpu::BindGroupLayoutEntry>,
    resource_commands: Vec<ResourceCommand>,
    call_commands: Vec<CallCommand>,
    uniform_components: Arc<RefCell<Vec<UniformController>>>,
    hashed_strings: HashMap<u32, String>,
    allocated_resources: HashMap<String, GPUResource>,
    mouse_state: MouseState,
    first_frame: bool,
}

#[derive(Debug)]
enum GPUResource {
    Texture(wgpu::Texture),
    Buffer(wgpu::Buffer),
}

impl GPUResource {
    fn destroy(&mut self) {
        match self {
            GPUResource::Texture(texture) => texture.destroy(),
            GPUResource::Buffer(buffer) => buffer.destroy(),
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

const PRINTF_BUFFER_ELEMENT_SIZE: usize = 12;
const PRINTF_BUFFER_SIZE: usize = PRINTF_BUFFER_ELEMENT_SIZE * 2048; // 12 bytes per printf struct
async fn process_resource_commands(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    resource_bindings: &HashMap<String, wgpu::BindGroupLayoutEntry>,
    resource_commands: &Vec<ResourceCommand>,
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

    for ResourceCommand {
        resource_name,
        command_data,
    } in resource_commands
    {
        match command_data {
            ResourceCommandData::ZEROS {
                count,
                element_size,
            } => {
                let Some(binding_info) = resource_bindings.get(resource_name) else {
                    panic!("Resource ${resource_name} is not defined in the bindings.");
                };

                if !matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
                    panic!("Resource ${resource_name} is an invalid type for ZEROS");
                }

                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    mapped_at_creation: false,
                    size: (count * element_size) as u64,
                    usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST),
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
            ResourceCommandData::RAND(count) => {
                let element_size = 4; // RAND is only valid for floats
                let Some(binding_info) = resource_bindings.get(resource_name) else {
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
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("compute builtin encoder"),
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute builtin pass"),
                    timestamp_writes: None,
                });

                pass.set_bind_group(0, random_pipeline.bind_group.as_ref(), &[]);
                pass.set_pipeline(random_pipeline.pipeline.as_ref().unwrap());

                let size = [*count, 1, 1];
                let block_size = random_pipeline.thread_group_size.unwrap();
                let work_group_size: Vec<u32> = size
                    .iter()
                    .zip(block_size.map(|s| s as u32))
                    .map(|(size, block_size)| (size + block_size - 1) / block_size)
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
            ResourceCommandData::BLACK {
                width,
                height,
                format,
            } => {
                let size = width * height;
                let element_size = format.block_copy_size(None).unwrap();
                let Some(binding_info) = resource_bindings.get(resource_name) else {
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
                        width: *width,
                        height: *height,
                        depth_or_array_layers: 1,
                    },
                    format: *format,
                    usage: usage,
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
                        width: *width,
                        height: *height,
                        depth_or_array_layers: 1,
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
                let Some(binding_info) = resource_bindings.get(resource_name) else {
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
                    format: *format,
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
            ResourceCommandData::URL { url, format } => {
                // Load image from URL and wait for it to be ready.
                let Some(binding_info) = resource_bindings.get(resource_name) else {
                    panic!("Resource {} is not defined in the bindings.", resource_name);
                };

                let element_size = format.block_copy_size(None).unwrap();

                if !matches!(binding_info.ty, wgpu::BindingType::Texture { .. }) {
                    panic!("Resource ${resource_name} is not a texture.");
                }
                let parsed_url = Url::parse(url);
                let image_bytes = if let Err(ParseError::RelativeUrlWithoutBase) = parsed_url {
                    read(url).unwrap()
                } else {
                    reqwest::blocking::get(parsed_url.unwrap())
                        .unwrap()
                        .bytes()
                        .unwrap()
                        .to_vec()
                };
                let image = image::load_from_memory(&image_bytes).unwrap();
                let data = match format {
                    wgpu::TextureFormat::Rgba8Unorm => image.to_rgba8().to_vec(),
                    wgpu::TextureFormat::R8Unorm => image
                        .to_rgba8()
                        .to_vec()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i % 4) < 1)
                        .map(|(_, c)| c)
                        .cloned()
                        .collect(),
                    wgpu::TextureFormat::Rg8Unorm => image
                        .to_rgba8()
                        .to_vec()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i % 4) < 2)
                        .map(|(_, c)| c)
                        .cloned()
                        .collect(),
                    wgpu::TextureFormat::Rgba32Float => image
                        .to_rgba32f()
                        .to_vec()
                        .iter()
                        .flat_map(|c| c.to_le_bytes())
                        .collect(),
                    wgpu::TextureFormat::Rg32Float => image
                        .to_rgba32f()
                        .to_vec()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i % 4) < 2)
                        .map(|(_, c)| c)
                        .flat_map(|c| c.to_le_bytes())
                        .collect(),
                    f => panic!("URL unimplemented for image format {f:?}"),
                };
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    dimension: wgpu::TextureDimension::D2,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                    size: wgpu::Extent3d {
                        width: image.width(),
                        height: image.height(),
                        depth_or_array_layers: 1,
                    },
                    format: *format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                });
                queue.write_texture(
                    texture.as_image_copy(),
                    &data,
                    wgpu::TexelCopyBufferLayout {
                        bytes_per_row: Some(image.width() * element_size),
                        offset: 0,
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: image.width(),
                        height: image.height(),
                        depth_or_array_layers: 1,
                    },
                );
                safe_set(
                    &mut allocated_resources,
                    resource_name,
                    GPUResource::Texture(texture),
                );
            }
            ResourceCommandData::SLIDER {
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
                let buffer_default = if *element_size == 4 {
                    default.to_le_bytes()
                } else {
                    panic!("Unsupported float size for slider")
                };
                queue.write_buffer(buffer, *offset as u64, &buffer_default);
            }
            ResourceCommandData::COLORPICK {
                default,
                element_size,
                offset,
            } => {
                let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                else {
                    panic!("cannot get uniforms")
                };
                // Initialize the buffer with zeros.
                let buffer_default = if *element_size == 4 {
                    bytemuck::cast_slice(default)
                } else {
                    panic!("Unsupported float size for color pick")
                };
                queue.write_buffer(buffer, *offset as u64, &buffer_default);
            }
            ResourceCommandData::MOUSEPOSITION { offset } => {
                let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                else {
                    panic!("cannot get uniforms")
                };
                // Initialize the buffer with zeros.
                let buffer_default = [0u8; 8];
                queue.write_buffer(buffer, *offset as u64, &buffer_default);
            }
            ResourceCommandData::TIME { offset } => {
                let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput")
                else {
                    panic!("cannot get uniforms")
                };
                // Initialize the buffer with zeros.
                let time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f32();
                let buffer_default = time.to_le_bytes();
                queue.write_buffer(buffer, *offset as u64, &buffer_default);
            }
        }
    }

    //
    // Some special-case allocations
    //

    safe_set(
        &mut allocated_resources,
        "outputBuffer".to_string(),
        GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size: 2 * 2 * 4,
            usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC),
        })),
    );

    safe_set(
        &mut allocated_resources,
        "outputBufferRead".to_string(),
        GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size: 2 * 2 * 4,
            usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
        })),
    );

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

    return allocated_resources;
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

    return format_specifiers;
}

fn format_printf_string(parsed_tokens: &Vec<FormatSpecifier>, data: &Vec<String>) -> String {
    let mut data_index = 0;

    parsed_tokens
        .iter()
        .map(|token| match &token {
            &FormatSpecifier::Text(value) => value.clone(),
            &FormatSpecifier::Specifier {
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
    flags: &String,
    width: &Option<usize>,
    precision: &Option<usize>,
    specifier_type: &String,
) -> String {
    let mut formatted_value;
    let was_precision_specified = precision.is_some();
    let precision = precision.unwrap_or(6); //eww magic number
    match specifier_type.as_str() {
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
            formatted_value = format!("{:.1$}", value.parse::<f32>().unwrap(), precision as usize);
        }
        "e" => {
            // Scientific notation (lowercase)
            formatted_value = format!("{:.1$e}", value.parse::<f32>().unwrap(), precision as usize);
        }
        "E" => {
            // Scientific notation (uppercase)
            formatted_value = format!("{:.1$E}", value.parse::<f32>().unwrap(), precision as usize);
        }
        "g" | "G" => {
            // Shortest representation of floating-point
            formatted_value = format!("{:.1$}", value.parse::<f32>().unwrap(), precision as usize);
        }
        "c" => {
            // Character
            formatted_value = String::from(value.parse::<u8>().unwrap() as char);
        }
        "s" => {
            // String
            formatted_value = value.clone();
            if was_precision_specified {
                formatted_value = formatted_value[0..precision as usize].to_string();
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

    return formatted_value;
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
                    .get(&(printf_buffer_array[offset + 1] << 0))
                    .unwrap()
                    .clone();
                // low field
            }
            2 => {
                // normal string
                data_array.push(
                    hashed_strings
                        .get(&(printf_buffer_array[offset + 1] << 0))
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

    if format_string != "" {
        // If we are here, it means that the printf buffer is used up, and we are in the middle of processing
        // one printf string, so we are still going to format it, even though there could be some data missing, which
        // will be shown as 'undef'.
        let parsed_tokens = parse_printf_format(format_string);
        let output = format_printf_string(&parsed_tokens, &data_array);
        out_str_arry.push(output);
        out_str_arry.push("Print buffer is out of boundary, some data is missing!!!".to_string());
    }

    return out_str_arry;
}

impl State {
    async fn new(window: Arc<Window>) -> State {
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

        let compilation: CompilationResult =
            ron::from_str(include_str!("../compiled_shaders/compiled.ron")).unwrap();

        let surface = instance.create_surface(window.clone()).unwrap();
        let surface_format = wgpu::TextureFormat::Rgba8Unorm;

        let mut random_pipeline = ComputePipeline::new(device.clone());

        // Load randFloat shader code from the file.
        let compiled_result: CompilationResult =
            ron::from_str(include_str!("../compiled_shaders/rand_float_compiled.ron")).unwrap();

        let rand_code = compiled_result.out_code;
        let rand_group_size = compiled_result
            .entry_group_sizes
            .get("computeMain")
            .unwrap();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rand float"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&rand_code.as_str())),
        });

        random_pipeline.set_thread_group_size(*rand_group_size);
        random_pipeline.create_pipeline_layout(compiled_result.bindings);

        // Create the pipeline (without resource bindings for now)
        random_pipeline.create_pipeline(module, None, None);

        let allocated_resources = process_resource_commands(
            &queue,
            &device,
            &compilation.bindings,
            &compilation.resource_commands,
            &mut random_pipeline,
            compilation.uniform_size,
        )
        .await;

        let mut pass_through_pipeline = GraphicsPipeline::new(device.clone());
        let shader_module = device.create_shader_module(wgpu::include_wgsl!("passThrough.wgsl"));
        let Some(GPUResource::Texture(input_texture)) =
            allocated_resources.get(&"outputTexture".to_string())
        else {
            panic!("outputTexture is not a Texture");
        };
        pass_through_pipeline.create_pipeline(&shader_module, input_texture);
        pass_through_pipeline.create_bind_group();

        let mut compute_pipelines = HashMap::new();

        for (shader_name, thread_group_size) in compilation.entry_group_sizes {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader_name.as_str()),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&compilation.out_code.as_str())),
            });
            let mut pipeline = ComputePipeline::new(device.clone());
            pipeline.create_pipeline_layout(compilation.bindings.clone());
            pipeline.create_pipeline(module, Some(&allocated_resources), Some(&shader_name));
            pipeline.set_thread_group_size(thread_group_size);
            compute_pipelines.insert(shader_name, pipeline);
        }

        let state = State {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            pass_through_pipeline,
            compute_pipelines,
            bindings: compilation.bindings,
            resource_commands: compilation.resource_commands,
            call_commands: compilation.call_commands,
            uniform_components: Arc::new(RefCell::new(compilation.uniform_controllers)),
            hashed_strings: compilation.hashed_strings,
            allocated_resources,
            mouse_state: MouseState {
                last_mouse_clicked_pos: PhysicalPosition::default(),
                last_mouse_down_pos: PhysicalPosition::default(),
                current_mouse_pos: PhysicalPosition::default(),
                mouse_clicked: false,
                is_mouse_down: false,
            },
            first_frame: true,
        };

        // Configure surface for the first time
        state.configure_surface();

        state
    }

    fn get_window(&self) -> &Window {
        &self.window
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
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;

        // reconfigure the surface
        self.configure_surface();

        for resource_command in self.resource_commands.iter() {
            match resource_command {
                ResourceCommand {
                    resource_name,
                    command_data,
                } => match command_data {
                    ResourceCommandData::BlackScreen {
                        format,
                        width_scale,
                        height_scale,
                    } => {
                        let width = (width_scale * new_size.width as f32) as u32;
                        let height = (height_scale * new_size.height as f32) as u32;
                        let size = width * height;
                        let element_size = format.block_copy_size(None).unwrap();
                        let Some(binding_info) = self.bindings.get(resource_name) else {
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
                    _ => {}
                },
            }
        }
        for (_, compute_pipeline) in self.compute_pipelines.iter_mut() {
            compute_pipeline.create_bind_group(&self.allocated_resources);
        }

        let Some(GPUResource::Texture(in_tex)) = self.allocated_resources.get("outputTexture")
        else {
            panic!("outputTexture is not a Texture");
        };
        self.pass_through_pipeline.input_texture = Some(in_tex.clone());
        self.pass_through_pipeline.create_bind_group();
    }

    fn render(&mut self) {
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

        for UniformController {
            buffer_offset,
            controller,
            ..
        } in self.uniform_components.borrow_mut().iter()
        {
            match controller {
                UniformControllerType::SLIDER { value, .. } => {
                    let slice = [*value];
                    let uniform_data = bytemuck::cast_slice(&slice);
                    self.queue
                        .write_buffer(uniform_input, *buffer_offset as u64, uniform_data);
                }
                UniformControllerType::COLORPICK { value, .. } => {
                    let slice = [*value];
                    let uniform_data = bytemuck::cast_slice(&slice);
                    self.queue
                        .write_buffer(uniform_input, *buffer_offset as u64, uniform_data);
                }
                UniformControllerType::MOUSEPOSITION => {
                    let mut time_array = [0.0f32; 8];
                    time_array[0] = self.mouse_state.last_mouse_down_pos.x as f32;
                    time_array[1] = self.mouse_state.last_mouse_down_pos.y as f32;
                    time_array[2] = self.mouse_state.last_mouse_clicked_pos.x as f32;
                    time_array[3] = self.mouse_state.last_mouse_clicked_pos.y as f32;
                    if self.mouse_state.is_mouse_down {
                        time_array[2] = -time_array[2];
                    }
                    if self.mouse_state.mouse_clicked {
                        time_array[3] = -time_array[3];
                    }
                    let uniform_data = bytemuck::cast_slice(&time_array);
                    self.queue
                        .write_buffer(uniform_input, *buffer_offset as u64, uniform_data);
                }
                UniformControllerType::TIME => {
                    let start = SystemTime::now();
                    let since_the_epoch = start
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards");
                    let value = since_the_epoch.as_millis() as u16 as f32 * 0.001;
                    let slice = [value];
                    let uniform_data = bytemuck::cast_slice(&slice);
                    self.queue
                        .write_buffer(uniform_input, *buffer_offset as u64, uniform_data);
                }
            }
        }

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

            let size: [u32; 3] = match &call_command.parameters {
                slang_compiler::CallCommandParameters::ResourceBased(
                    resource_name,
                    element_size,
                ) => {
                    let resource = self.allocated_resources.get(resource_name).unwrap();
                    match resource {
                        GPUResource::Texture(texture) => [texture.width(), texture.height(), 1],
                        GPUResource::Buffer(buffer) => {
                            [buffer.size() as u32 / element_size.unwrap_or(4), 1, 1]
                        }
                    }
                }
                slang_compiler::CallCommandParameters::FixedSize(items) => {
                    if items.len() > 3 {
                        panic!("Too many parameters for call command");
                    }
                    let mut size = [1; 3];
                    for (i, n) in items.iter().enumerate() {
                        size[i] = *n;
                    }
                    size
                }
            };

            //TODO
            let block_size = pipeline.thread_group_size.unwrap();

            let work_group_size: Vec<u32> = size
                .iter()
                .zip(block_size.map(|s| s as u32))
                .map(|(size, block_size)| (size + block_size - 1) / block_size)
                .collect();

            pass.dispatch_workgroups(work_group_size[0], work_group_size[1], work_group_size[2]);
            drop(pass);
        }

        if let Some(print_compute_pipeline) = self.compute_pipelines.get("printMain") {
            let Some(&GPUResource::Buffer(ref printf_buffer_read)) =
                self.allocated_resources.get("printfBufferRead")
            else {
                panic!("outputBuffer is incorrect type or doesn't exist");
            };
            encoder.clear_buffer(&printf_buffer_read, 0, None);

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute builtin pass"),
                timestamp_writes: None,
            });

            pass.set_bind_group(0, print_compute_pipeline.bind_group.as_ref(), &[]);
            pass.set_pipeline(print_compute_pipeline.pipeline.as_ref().unwrap());

            let size = [1, 1, 1];
            let block_size = print_compute_pipeline.thread_group_size.unwrap();
            let work_group_size: Vec<u32> = size
                .iter()
                .zip(block_size.map(|s| s as u32))
                .map(|(size, block_size)| (size + block_size - 1) / block_size)
                .collect();

            pass.dispatch_workgroups(work_group_size[0], work_group_size[1], work_group_size[2]);
            drop(pass);

            let Some(&GPUResource::Buffer(ref output_buffer)) =
                self.allocated_resources.get("outputBuffer")
            else {
                panic!("outputBuffer is incorrect type or doesn't exist");
            };
            let Some(&GPUResource::Buffer(ref output_buffer_read)) =
                self.allocated_resources.get("outputBufferRead")
            else {
                panic!("outputBufferRead is incorrect type or doesn't exist");
            };
            let Some(&GPUResource::Buffer(ref g_printed_buffer)) =
                self.allocated_resources.get("g_printedBuffer")
            else {
                panic!("g_printedBuffer is not a buffer");
            };
            encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &output_buffer_read,
                0,
                output_buffer.size(),
            );
            encoder.copy_buffer_to_buffer(
                &g_printed_buffer,
                0,
                &printf_buffer_read,
                0,
                g_printed_buffer.size(),
            );
        }

        // Create the renderpass which will clear the screen.
        let mut renderpass = self
            .pass_through_pipeline
            .begin_render_pass(&mut encoder, &texture_view);

        renderpass.set_bind_group(0, self.pass_through_pipeline.bind_group.as_ref(), &[]);
        renderpass.set_pipeline(self.pass_through_pipeline.pipeline.as_ref().unwrap());

        // If you wanted to call any drawing commands, they would go here.
        renderpass.draw(0..6, 0..1);

        // End the renderpass.
        drop(renderpass);

        // Submit the command in the queue to execute
        self.queue.submit([encoder.finish()]);

        if self.compute_pipelines.contains_key("printMain") {
            let Some(&GPUResource::Buffer(ref printf_buffer_read)) =
                self.allocated_resources.get("printfBufferRead")
            else {
                panic!("outputBuffer is incorrect type or doesn't exist");
            };
            let (sender, receiver) = futures_channel::oneshot::channel();
            printf_buffer_read
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| {
                    let _ = sender.send(result);
                });
            self.device.poll(wgpu::PollType::Wait).unwrap();
            let rt = runtime::Builder::new_current_thread().build().unwrap();
            rt.spawn_blocking(|| async {
                receiver
                    .await
                    .expect("communication failed")
                    .expect("buffer reading failed")
            });

            let format_print = parse_printf_buffer(
                &self.hashed_strings,
                printf_buffer_read,
                PRINTF_BUFFER_ELEMENT_SIZE,
            );

            if format_print.len() != 0 {
                print!("Shader Output:\n{}\n", format_print.join(""));
            }

            printf_buffer_read.unmap();
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
}

struct DebugPanel {
    uniform_controllers: Arc<RefCell<Vec<UniformController>>>,
}

pub struct AppState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub surface: wgpu::Surface<'static>,
    pub scale_factor: f32,
    pub egui_renderer: EguiRenderer,
    debug_panel: DebugPanel,
}

impl AppState {
    async fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        window: &Window,
        width: u32,
        height: u32,
        debug_panel: DebugPanel,
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
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 0,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &surface_config);

        let egui_renderer = EguiRenderer::new(&device, surface_config.format, None, 1, window);

        let scale_factor = 1.0;

        Self {
            device,
            queue,
            surface,
            surface_config,
            egui_renderer,
            scale_factor,
            debug_panel,
        }
    }

    fn resize_surface(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }
}

#[derive(Default)]
struct App {
    instance: wgpu::Instance,
    state: Option<State>,
    debug_app: Option<AppState>,
    debug_window: Option<Arc<Window>>,
}
impl App {
    fn new() -> Self {
        let instance = egui_wgpu::wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        Self {
            instance: instance,
            state: None,
            debug_app: None,
            debug_window: None,
        }
    }

    async fn set_window(&mut self, window: Window) {
        let window = Arc::new(window);
        let initial_width = 1360;
        let initial_height = 768;

        let _ = window.request_inner_size(PhysicalSize::new(initial_width, initial_height));

        let surface = self
            .instance
            .create_surface(window.clone())
            .expect("Failed to create surface!");

        let debug_panel = DebugPanel {
            uniform_controllers: self.state.as_ref().unwrap().uniform_components.clone(),
        };

        let debug_state = AppState::new(
            &self.instance,
            surface,
            &window,
            initial_width,
            initial_width,
            debug_panel,
        )
        .await;

        self.debug_window.get_or_insert(window);
        self.debug_app.get_or_insert(debug_state);
    }

    fn handle_resized(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.debug_app
                .as_mut()
                .unwrap()
                .resize_surface(width, height);
        }
    }

    fn handle_redraw(&mut self) {
        // Attempt to handle minimizing window
        if let Some(window) = self.debug_window.as_ref() {
            if let Some(min) = window.is_minimized() {
                if min {
                    println!("Window is minimized");
                    return;
                }
            }
        }

        let state = self.debug_app.as_mut().unwrap();

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [state.surface_config.width, state.surface_config.height],
            pixels_per_point: self.debug_window.as_ref().unwrap().scale_factor() as f32
                * state.scale_factor,
        };

        let surface_texture = state.surface.get_current_texture();

        match surface_texture {
            Err(SurfaceError::Outdated) => {
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

        let window = self.debug_window.as_ref().unwrap();

        {
            state.egui_renderer.begin_frame(window);
            egui::CentralPanel::default().show(state.egui_renderer.context(), |ui| {
                ui.heading("Uniforms:");

                for UniformController {
                    name, controller, ..
                } in state
                    .debug_panel
                    .uniform_controllers
                    .borrow_mut()
                    .iter_mut()
                {
                    match controller {
                        UniformControllerType::SLIDER {
                            value, min, max, ..
                        } => {
                            ui.label(name.as_str());
                            ui.add(Slider::new(value, *min..=*max));
                        }
                        UniformControllerType::COLORPICK { value, .. } => {
                            ui.label(name.as_str());
                            ui.color_edit_button_rgb(value);
                        }
                        UniformControllerType::MOUSEPOSITION | UniformControllerType::TIME => {
                            // do nothing
                        }
                    }
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
        // Create window object
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Slang Native Playground"))
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone()));
        self.state = Some(state);

        if cfg!(debug_assertions) {
            let debug_window = event_loop
                .create_window(
                    Window::default_attributes().with_title("Slang Native Playground Debug"),
                )
                .unwrap();
            pollster::block_on(self.set_window(debug_window));
        }

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self
            .debug_window
            .as_ref()
            .map(|w| w.id() == id)
            .unwrap_or(false)
        {
            self.debug_app
                .as_mut()
                .unwrap()
                .egui_renderer
                .handle_input(self.debug_window.as_ref().unwrap(), &event);

            match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.handle_redraw();

                    self.debug_window.as_ref().unwrap().request_redraw();
                }
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
            WindowEvent::RedrawRequested => {
                state.render();
                // Emits a new redraw requested event.
                state.get_window().request_redraw();
            }
            WindowEvent::Resized(size) => {
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always folloed up by redraw request.
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
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let state = self.state.as_mut().unwrap();
        state.render();
        state.get_window().request_redraw();
    }
}

fn main() {
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

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
