mod graphics_pipeline;
mod compute_pipeline;
mod slang_compiler;

use slang_compiler::{CallCommand, ResourceCommand, ResourceCommandData, SlangCompiler};
use wgpu::TextureFormat;

use std::{borrow::Cow, collections::HashMap, sync::Arc, time::{SystemTime, UNIX_EPOCH}};

use compute_pipeline::ComputePipeline;
use graphics_pipeline::GraphicsPipeline;
use winit::{
    application::ApplicationHandler, dpi::PhysicalPosition, event::{ElementState, MouseButton, WindowEvent}, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::{Window, WindowId}
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
    call_commands: Vec<CallCommand>,
    allocated_resources: HashMap<String, GPUResource>,
    mouse_state: MouseState,
}

fn create_output_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    let texture_desc = wgpu::TextureDescriptor {
        label: Some("output storage texture"),
        size: wgpu::Extent3d { width: width, height: height, depth_or_array_layers: 1 },
        format: format,
        usage: wgpu::TextureUsages::COPY_SRC |
            wgpu::TextureUsages::STORAGE_BINDING |
            wgpu::TextureUsages::TEXTURE_BINDING,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        view_formats: &[],
    };

    device.create_texture(&texture_desc)
}

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


const PRINTF_BUFFER_ELEMENT_SIZE: u64 = 12;
const PRINTF_BUFFER_SIZE: u64 = PRINTF_BUFFER_ELEMENT_SIZE * 2048; // 12 bytes per printf struct
async fn process_resource_commands(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    resource_bindings: &HashMap<String, wgpu::BindGroupLayoutEntry>,
    resource_commands: Vec<ResourceCommand>,
) -> HashMap<String, GPUResource> {
    let mut allocated_resources: HashMap<String, GPUResource> = HashMap::new();

    for ResourceCommand { resource_name, command_data } in resource_commands {
        match command_data {
            ResourceCommandData::ZEROS{
                count,
                element_size,
            } => {
                let Some(binding_info) = resource_bindings.get(&resource_name) else {
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

                safe_set(&mut allocated_resources, resource_name, GPUResource::Buffer(buffer));
            }
            _ => {}
        }
    }
    //          else if (parsedCommand.type === "BLACK") {
    //         let size = parsedCommand.width * parsedCommand.height;
    //         let elementSize = 4; // Assuming 4 bytes per element (e.g., float) TODO: infer from type.
    //         let bindingInfo = resourceBindings.get(resourceName);
    //         if (!bindingInfo) {
    //             panic!("Resource ${resourceName} is not defined in the bindings.");
    //         }

    //         if (!bindingInfo.texture && !bindingInfo.storageTexture) {
    //             panic!("Resource ${resourceName} is an invalid type for BLACK");
    //         }
    //         try {
    //             let usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
    //             if (bindingInfo.storageTexture) {
    //                 usage |= GPUTextureUsage.STORAGE_BINDING;
    //             }
    //             let texture = device.createTexture({
    //                 size: [parsedCommand.width, parsedCommand.height],
    //                 format: bindingInfo.storageTexture ? 'r32float' : 'rgba8unorm',
    //                 usage: usage,
    //             });

    //             safeSet(allocatedResources, resourceName, texture);

    //             // Initialize the texture with zeros.
    //             let zeros = new Uint8Array(Array(size * elementSize).fill(0));
    //             device.queue.writeTexture({ texture }, zeros, { bytesPerRow: parsedCommand.width * elementSize }, { width: parsedCommand.width, height: parsedCommand.height });
    //         }
    //         catch (error) {
    //             panic!("Failed to create texture: ${error}");
    //         }
    //     } else if (parsedCommand.type === "URL") {
    //         // Load image from URL and wait for it to be ready.
    //         let bindingInfo = resourceBindings.get(resourceName);

    //         if (!bindingInfo) {
    //             panic!("Resource ${resourceName} is not defined in the bindings.");
    //         }

    //         if (!bindingInfo.texture) {
    //             panic!("Resource ${resourceName} is not a texture.");
    //         }

    //         let image = new Image();
    //         try {
    //             // TODO: Pop-up a warning if the image is not CORS-enabled.
    //             // TODO: Pop-up a warning for the user to confirm that its okay to load a cross-origin image (i.e. do you trust this code..)
    //             //
    //             image.crossOrigin = "anonymous";

    //             image.src = parsedCommand.url;
    //             await image.decode();
    //         }
    //         catch (error) {
    //             panic!("Failed to load & decode image from URL: ${parsedCommand.url}");
    //         }

    //         try {
    //             let imageBitmap = await createImageBitmap(image);
    //             let texture = device.createTexture({
    //                 size: [imageBitmap.width, imageBitmap.height],
    //                 format: 'rgba8unorm',
    //                 usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    //             });
    //             device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture: texture }, [imageBitmap.width, imageBitmap.height]);
    //             safeSet(allocatedResources, resourceName, texture);
    //         }
    //         catch (error) {
    //             panic!("Failed to create texture from image: ${error}");
    //         }
    //     } else if (parsedCommand.type === "RAND") {
    //         let elementSize = 4; // RAND is only valid for floats
    //         let bindingInfo = resourceBindings.get(resourceName);
    //         if (!bindingInfo) {
    //             panic!("Resource ${resourceName} is not defined in the bindings.");
    //         }

    //         if (!bindingInfo.buffer) {
    //             panic!("Resource ${resourceName} is not defined as a buffer.");
    //         }

    //         let buffer = device.createBuffer({
    //             size: parsedCommand.count * elementSize,
    //             usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    //         });

    //         safeSet(allocatedResources, resourceName, buffer);

    //         // Place a call to a shader that fills the buffer with random numbers.
    //         if (!randFloatPipeline) {
    //             let randomPipeline = new ComputePipeline(device);

    //             // Load randFloat shader code from the file.
    //             let randFloatShaderCode = await (await fetch('demos/rand_float.slang')).text();
    //             if (compiler == null) {
    //                 panic!("Compiler is not defined!");
    //             }
    //             let compiledResult = compiler.compile(randFloatShaderCode, "computeMain", "WGSL");
    //             if (!compiledResult) {
    //                 panic!("[Internal] Failed to compile randFloat shader");
    //             }

    //             let [code, layout, hashedStrings] = compiledResult;
    //             let module = device.createShaderModule({ code: code });

    //             randomPipeline.createPipelineLayout(layout);

    //             // Create the pipeline (without resource bindings for now)
    //             randomPipeline.createPipeline(module, null);

    //             randFloatPipeline = randomPipeline;
    //         }

    //         // Dispatch a random number generation shader.
    //         {
    //             let randomPipeline = randFloatPipeline;
  
    //             // Alloc resources for the shader.
    //             if (!randFloatResources)
    //                 randFloatResources = new Map();

    //             randFloatResources.set("outputBuffer", buffer);

    //             if (!randFloatResources.has("seed"))
    //                 randFloatResources.set("seed",
    //                     device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));

    //             let seedBuffer = randFloatResources.get("seed") as GPUBuffer;

    //             // Set bindings on the pipeline.
    //             randFloatPipeline.createBindGroup(randFloatResources);

    //             let seedValue = new Float32Array([Math.random(), 0, 0, 0]);
    //             device.queue.writeBuffer(seedBuffer, 0, seedValue);

    //             // Encode commands to do the computation
    //             let encoder = device.createCommandEncoder({ label: 'compute builtin encoder' });
    //             let pass = encoder.beginComputePass({ label: 'compute builtin pass' });

    //             pass.setBindGroup(0, randomPipeline.bindGroup || null);

    //             if (randomPipeline.pipeline == undefined) {
    //                 panic!("Random pipeline is undefined");
    //             }
    //             pass.setPipeline(randomPipeline.pipeline);

    //             let workGroupSizeX = Math.floor((parsedCommand.count + 63) / 64);
    //             pass.dispatchWorkgroups(workGroupSizeX, 1);
    //             pass.end();

    //             // Finish encoding and submit the commands
    //             let commandBuffer = encoder.finish();
    //             device.queue.submit([commandBuffer]);
    //             await device.queue.onSubmittedWorkDone();
    //         }
    //     } else {
    //         // exhaustiveness check
    //         let x: never = parsedCommand;
    //         panic!("Invalid resource command type");
    //     }
    // }

    //
    // Some special-case allocations
    //
    let current_window_size = [300, 150];//TODO

    safe_set(&mut allocated_resources, "outputTexture".to_string(), GPUResource::Texture(create_output_texture(&device, current_window_size[0], current_window_size[1], TextureFormat::Rgba8Unorm)));

    safe_set(&mut allocated_resources, "outputBuffer".to_string(), GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: 2 * 2 * 4,
        usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC),
    })));

    safe_set(&mut allocated_resources, "outputBufferRead".to_string(), GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: 2 * 2 * 4,
        usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
    })));

    safe_set(&mut allocated_resources, "g_printedBuffer".to_string(), GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: PRINTF_BUFFER_SIZE,
        usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC),
    })));

    safe_set(&mut allocated_resources, "printfBufferRead".to_string(), GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: PRINTF_BUFFER_SIZE,
        usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
    })));

    safe_set(&mut allocated_resources, "uniformInput".to_string(), GPUResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: 8*4,//TODO
        usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
    })));

    return allocated_resources;
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
                &wgpu::DeviceDescriptor::default(),
                None, // Trace path
            )
            .await
            .unwrap();

        let device = Arc::new(device);

        let size = window.inner_size();

        let compiler = SlangCompiler::new();
        let compilation = compiler.compile("imageMain".to_string());

        let surface = instance.create_surface(window.clone()).unwrap();   
        let surface_format = wgpu::TextureFormat::Rgba8Unorm;

        let allocated_resources = process_resource_commands(&queue, &device, &compilation.bindings, compilation.resource_commands).await;

        let mut pass_through_pipeline = GraphicsPipeline::new(device.clone());
        let shader_module = device.create_shader_module(wgpu::include_wgsl!("passThrough.wgsl"));
        let GPUResource::Texture(input_texture) = allocated_resources.get(&"outputTexture".to_string()).unwrap() else {
            panic!("outputTexture is not a Texture");
        };
        pass_through_pipeline.create_pipeline(&shader_module, input_texture);
        pass_through_pipeline.create_bind_group();

        let mut compute_pipelines = HashMap::new();

        for (shader_name, shader_code) in compilation.out_code {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader_name.as_str()),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_code.as_str())),
            });
            let mut pipeline = ComputePipeline::new(device.clone());
            pipeline.create_pipeline_layout(compilation.bindings.clone());
            pipeline.create_pipeline(module, &allocated_resources);
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
            call_commands: compilation.call_commands,
            allocated_resources,
            mouse_state: MouseState { last_mouse_clicked_pos: PhysicalPosition::default(), last_mouse_down_pos: PhysicalPosition::default(), current_mouse_pos: PhysicalPosition::default(), mouse_clicked: false, is_mouse_down: false }
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

        safe_set(&mut self.allocated_resources, "outputTexture", GPUResource::Texture(create_output_texture(&self.device, new_size.width, new_size.height, wgpu::TextureFormat::Rgba8Unorm)));
        for (_, compute_pipeline) in self.compute_pipelines.iter_mut() {
            compute_pipeline.create_bind_group(&self.allocated_resources);
        }
    
        let Some(GPUResource::Texture(in_tex)) = self.allocated_resources.get("outputTexture") else {
            panic!();
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

        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

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
        time_array[4] = since_the_epoch.as_millis() as u16 as f32 * 0.001;
        let Some(GPUResource::Buffer(uniform_input)) = self.allocated_resources.get("uniformInput") else {
            panic!("uniformInput doesn't exist or is of incorrect type");
        };
        self.queue.write_buffer(uniform_input, 0, bytemuck::cast_slice(&time_array));

        let mut encoder = self.device.create_command_encoder(&Default::default());

        for call_command in self.call_commands.iter() {
            let pipeline = self.compute_pipelines.get(call_command.function.as_str()).unwrap();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute builtin pass"),
                timestamp_writes: None,
            });

            pass.set_bind_group(0, pipeline.bind_group.as_ref(), &[]);
            pass.set_pipeline(pipeline.pipeline.as_ref().unwrap());

            let size: [u32; 3] = match &call_command.parameters {
                slang_compiler::CallCommandParameters::ResourceBased(resource_name, element_size) => {
                    let resource = self.allocated_resources.get(resource_name).unwrap();
                    match resource {
                        GPUResource::Texture(texture) => {
                            [texture.width(), texture.height(), 1]
                        },
                        GPUResource::Buffer(buffer) => {
                            [buffer.size() as u32 / element_size.unwrap_or(4), 1, 1]
                        },
                    }
                },
                slang_compiler::CallCommandParameters::FixedSize(items) => {
                    if items.len() > 3 {
                        panic!("Too many parameters for call command");
                    }
                    let mut size = [1; 3];
                    for (i, n) in items.iter().enumerate() {
                        size[i] = *n;
                    }
                    size
                },
            };

            //TODO
            let block_size_x = 64;
            let block_size_y = 1;
            let block_size_z = 1;
    

            let work_group_size_x = (size[0] + block_size_x - 1) / block_size_x;
            let work_group_size_y = (size[1] + block_size_y - 1) / block_size_y;
            let work_group_size_z = (size[2] + block_size_z - 1) / block_size_z;
            pass.dispatch_workgroups(work_group_size_x, work_group_size_y, work_group_size_z);
            drop(pass);
        }

        let main_compute_pipeline = self.compute_pipelines.get("imageMain").unwrap();

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute builtin pass"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, main_compute_pipeline.bind_group.as_ref(), &[]);
        pass.set_pipeline(main_compute_pipeline.pipeline.as_ref().unwrap());

        let work_group_size_x = (surface_texture.texture.width() + 15) / 16;
        let work_group_size_y = (surface_texture.texture.height() + 15) / 16;
        pass.dispatch_workgroups(work_group_size_x, work_group_size_y, 1);
        drop(pass);

        // Create the renderpass which will clear the screen.
        let mut renderpass = self.pass_through_pipeline.begin_render_pass(&mut encoder, &texture_view);

        renderpass.set_bind_group(0, self.pass_through_pipeline.bind_group.as_ref(), &[]);
        renderpass.set_pipeline(self.pass_through_pipeline.pipeline.as_ref().unwrap());

        // If you wanted to call any drawing commands, they would go here.
        renderpass.draw(0..6, 0..1);

        // End the renderpass.
        drop(renderpass);

        // Submit the command in the queue to execute
        self.queue.submit([encoder.finish()]);
        surface_texture.present();
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

#[derive(Default)]
struct App {
    state: Option<State>,
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

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
            WindowEvent::CursorMoved { device_id: _, position } => {
                state.mousemove(position);
            }
            WindowEvent::MouseInput { device_id: _, state: mouse_state, button } => {
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

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
