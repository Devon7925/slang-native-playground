mod compute_pipeline;
mod draw_pipeline;

use draw_pipeline::DrawPipeline;
use regex::Regex;
use slang_playground_compiler::{
    CallCommand, CallCommandParameters, CompilationResult, DrawCommand, GPUResource, GraphicsAPI,
    ResourceCommandData, ResourceMetadata, UniformController, UniformSourceData, safe_set,
};
use wgpu::Extent3d;

use std::{
    borrow::Cow, cell::RefCell, collections::HashMap, panic, rc::Rc, sync::Arc, time::Duration,
};

use compute_pipeline::ComputePipeline;
use std::collections::HashSet;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton, WindowEvent},
};

struct MouseState {
    last_mouse_clicked_pos: PhysicalPosition<f64>,
    last_mouse_down_pos: PhysicalPosition<f64>,
    current_mouse_pos: PhysicalPosition<f64>,
    mouse_clicked: bool,
    is_mouse_down: bool,
}

struct KeyboardState {
    pressed_keys: HashSet<String>,
}

impl KeyboardState {
    fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
        }
    }

    fn key_pressed(&mut self, key: String) {
        self.pressed_keys.insert(key);
    }

    fn key_released(&mut self, key: String) {
        self.pressed_keys.remove(&key);
    }
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    compute_pipelines: HashMap<String, ComputePipeline>,
    draw_pipelines: Vec<DrawPipeline>,
    bindings: HashMap<String, wgpu::BindGroupLayoutEntry>,
    resource_commands: HashMap<String, Box<dyn ResourceCommandData>>,
    call_commands: Vec<CallCommand>,
    draw_commands: Vec<DrawCommand>,
    pub uniform_components: Rc<RefCell<Vec<UniformController>>>,
    uniform_size: u64,
    hashed_strings: HashMap<u32, String>,
    allocated_resources: HashMap<String, GPUResource>,
    mouse_state: MouseState,
    keyboard_state: KeyboardState,
    render_size: PhysicalSize<u32>,
    print_receiver: Option<futures::channel::oneshot::Receiver<()>>,
    first_frame: bool,
    delta_time: f32,
    last_frame_time: web_time::Instant,
    launch_time: web_time::Instant,
    frame_count: u64,
}

fn get_resource_metadata(
    _resource_commands: &HashMap<String, Box<dyn ResourceCommandData>>,
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
    resource_commands: &HashMap<String, Box<dyn ResourceCommandData>>,
    resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
    uniform_controllers: &Vec<UniformController>,
    uniform_size: u64,
    window_size: PhysicalSize<u32>,
) -> HashMap<String, GPUResource> {
    let mut allocated_resources: HashMap<String, GPUResource> = HashMap::new();

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

    let mut unprocessed_resource_commands = resource_commands
        .iter()
        .map(|(k, v)| (k.clone(), v))
        .collect::<HashMap<_, _>>();
    while !unprocessed_resource_commands.is_empty() {
        unprocessed_resource_commands.retain(|resource_name, command_data| {
            if let Ok(resource) = command_data.assign_resources(
                GraphicsAPI {
                    device,
                    queue,
                    allocated_resources: &mut allocated_resources,
                    resource_bindings,
                },
                resource_metadata,
                &resource_name,
                window_size,
            ) {
                safe_set(&mut allocated_resources, resource_name.clone(), resource);
                false
            } else {
                true
            }
        });
    }

    if !uniform_controllers.is_empty() {
        let keys = HashSet::new();
        let uniform_source_data = UniformSourceData::new(&keys);
        let Some(GPUResource::Buffer(buffer)) = allocated_resources.get("uniformInput") else {
            panic!("cannot get uniforms")
        };
        for UniformController {
            controller,
            buffer_offset,
            ..
        } in uniform_controllers
        {
            // Initialize the uniform data.
            let buffer_default = controller.get_data(&uniform_source_data);
            queue.write_buffer(buffer, *buffer_offset as u64, &buffer_default);
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
            formatted_value = format!("{:.1$}e", value.parse::<f32>().unwrap(), precision);
        }
        "E" => {
            // Scientific notation (uppercase)
            formatted_value = format!("{:.1$}E", value.parse::<f32>().unwrap(), precision);
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

impl Renderer {
    pub async fn new(
        compilation: CompilationResult,
        render_size: PhysicalSize<u32>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        let resource_metadata =
            get_resource_metadata(&compilation.resource_commands, &compilation.call_commands);

        let allocated_resources = process_resource_commands(
            &queue,
            &device,
            &compilation.bindings,
            &compilation.resource_commands,
            &resource_metadata,
            &compilation.uniform_controllers,
            compilation.uniform_size,
            render_size,
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

        let renderer = Self {
            device,
            queue: queue.clone(),
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
            render_size,
            print_receiver: None,
            first_frame: true,
            delta_time: 0.0,
            last_frame_time: web_time::Instant::now(),
            launch_time: web_time::Instant::now(),
            frame_count: 0,
        };

        renderer
    }

    /// Prepare uniforms and compute passes. Call at the start of a frame.
    pub fn begin_frame(&mut self) {
        let Some(GPUResource::Buffer(uniform_input)) = self.allocated_resources.get("uniformInput")
        else {
            panic!("uniformInput doesn't exist or is of incorrect type");
        };

        let mut buffer_data: Vec<u8> = vec![0; self.uniform_size as usize];

        // Calculate delta time
        let now = web_time::Instant::now();
        let frame_time: Duration = now - self.last_frame_time;
        self.delta_time = frame_time.as_secs_f32();
        self.last_frame_time = now;
        self.frame_count += 1;

        let uniform_source_data = UniformSourceData {
            launch_time: self.launch_time,
            delta_time: self.delta_time,
            last_mouse_down_pos: [
                self.mouse_state.last_mouse_down_pos.x as f32,
                self.mouse_state.last_mouse_down_pos.y as f32,
            ],
            last_mouse_clicked_pos: [
                self.mouse_state.last_mouse_clicked_pos.x as f32,
                self.mouse_state.last_mouse_clicked_pos.y as f32,
            ],
            mouse_down: self.mouse_state.is_mouse_down,
            mouse_clicked: self.mouse_state.mouse_clicked,
            pressed_keys: &self.keyboard_state.pressed_keys,
            frame_count: self.frame_count,
        };

        for UniformController {
            buffer_offset,
            controller,
            ..
        } in self.uniform_components.borrow().iter()
        {
            let slice = controller.get_data(&uniform_source_data);
            let uniform_data = bytemuck::cast_slice(&slice);
            buffer_data[*buffer_offset as usize..(*buffer_offset as usize + uniform_data.len())]
                .copy_from_slice(uniform_data);
        }

        self.queue.write_buffer(uniform_input, 0, &buffer_data);
    }

    /// Run compute passes. Accepts a mutable CommandEncoder.
    pub fn run_compute_passes(&mut self, encoder: &mut wgpu::CommandEncoder) {
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

            use slang_playground_compiler::CallCommandParameters;

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
        self.first_frame = false;
    }

    /// Run draw passes. Accepts a mutable CommandEncoder and a TextureView.
    pub fn run_draw_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
    ) {
        let mut pass = DrawPipeline::begin_render_pass(
            &self.device,
            Extent3d {
                width: self.render_size.width,
                height: self.render_size.height,
                depth_or_array_layers: 1,
            },
            encoder,
            texture_view,
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
    }

    /// Optionally handle print buffer output after frame submission
    pub fn handle_print_output(&mut self) {
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
                    &printf_buffer_read,
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
            }
        }
    }

    pub fn process_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                self.render_size = *size;
                // reconfigure the surface if provided
                for (resource_name, command_data) in self.resource_commands.iter() {
                    command_data.handle_resize(
                        GraphicsAPI {
                            queue: &self.queue,
                            device: &self.device,
                            resource_bindings: &self.bindings,
                            allocated_resources: &mut self.allocated_resources,
                        },
                        resource_name,
                        *size,
                    );
                }
                for (_, compute_pipeline) in self.compute_pipelines.iter_mut() {
                    compute_pipeline.create_bind_group(&self.allocated_resources);
                }
                for draw_pipeline in self.draw_pipelines.iter_mut() {
                    draw_pipeline
                        .create_bind_group(&self.allocated_resources, &self.resource_commands);
                }
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.mouse_state.current_mouse_pos = *position;
                if self.mouse_state.is_mouse_down {
                    self.mouse_state.last_mouse_down_pos = *position;
                }
            }
            WindowEvent::MouseInput {
                device_id: _,
                state: mouse_state,
                button,
            } => {
                if *button == MouseButton::Left {
                    if *mouse_state == ElementState::Pressed {
                        self.mouse_state.last_mouse_clicked_pos =
                            self.mouse_state.current_mouse_pos;
                        self.mouse_state.mouse_clicked = true;
                        self.mouse_state.is_mouse_down = true;
                    } else {
                        self.mouse_state.is_mouse_down = false;
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event,
                device_id: _,
                is_synthetic: _,
            } => {
                let event_code = match event.physical_key {
                    winit::keyboard::PhysicalKey::Code(code) => format!("{:?}", code),
                    winit::keyboard::PhysicalKey::Unidentified(code) => format!("{:?}", code),
                };
                let event_key = event.logical_key.to_text().unwrap_or("").to_string();

                match event.state {
                    ElementState::Pressed => {
                        self.keyboard_state.key_pressed(event_code);
                        self.keyboard_state.key_pressed(event_key);
                    }
                    ElementState::Released => {
                        self.keyboard_state.key_released(event_code);
                        self.keyboard_state.key_released(event_key);
                    }
                }
            }
            _ => (),
        }
    }
}
