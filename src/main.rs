mod graphics_pipeline;
mod compute_pipeline;
mod slang_compiler;

use slang::Downcast;
use slang_compiler::SlangCompiler;
use wgpu::{BufferUsages, TextureFormat};

use std::{borrow::Cow, collections::HashMap, sync::Arc, time::{SystemTime, UNIX_EPOCH}};

use compute_pipeline::ComputePipeline;
use graphics_pipeline::GraphicsPipeline;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

fn compile_entrypoints() -> HashMap<String, String> {
    let global_session = slang::GlobalSession::new().unwrap();

    let search_path = std::ffi::CString::new("shaders").unwrap();

    // All compiler options are available through this builder.
    let session_options = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::High)
        .matrix_layout_row(true);

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Wgsl)
        .profile(global_session.find_profile("spirv_1_6"));

    let targets = [target_desc];
    let search_paths = [search_path.as_ptr()];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session.create_session(&session_desc).unwrap();
    let module = session.load_module("imageMain.slang").unwrap();

    let count = module.get_defined_entry_point_count();
    
    let mut result = HashMap::new();
    for i in 0..count {
        let entry_point = module.get_defined_entry_point(i).unwrap();

        let name = entry_point.get_function_reflection().name();

        let program = session
            .create_composite_component_type(&[
                module.downcast().clone(),
                entry_point.downcast().clone(),
            ])
            .unwrap();

        let linked_program = program.link().unwrap();

        let code = linked_program
            .entry_point_code(0, 0)
            .unwrap()
            .as_slice()
            .to_vec();
        //convert to string
        let code = String::from_utf8(code).unwrap();

        result.insert(name.to_string(), code);
    }
    result
}

struct State {
    window: Arc<Window>,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    pass_through_pipeline: GraphicsPipeline,
    main_compute_pipeline: ComputePipeline,
    allocated_resources: HashMap<String, GPUResource>
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
    pipeline: &ComputePipeline,
    // resource_bindings: Bindings,
    // resource_commands: ResourceCommand[]
) -> HashMap<String, GPUResource> {
    let mut allocated_resources: HashMap<String, GPUResource> = HashMap::new();

    // for (const { resourceName, parsedCommand } of resourceCommands) {
    //     if (parsedCommand.type === "ZEROS") {
    //         const elementSize = parsedCommand.elementSize;
    //         const bindingInfo = resourceBindings.get(resourceName);
    //         if (!bindingInfo) {
    //             throw new Error(`Resource ${resourceName} is not defined in the bindings.`);
    //         }

    //         if (!bindingInfo.buffer) {
    //             throw new Error(`Resource ${resourceName} is an invalid type for ZEROS`);
    //         }

    //         const buffer = pipeline.device.createBuffer({
    //             size: parsedCommand.count * elementSize,
    //             usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    //         });

    //         safeSet(allocatedResources, resourceName, buffer);

    //         // Initialize the buffer with zeros.
    //         let zeros: BufferSource = new Uint8Array(parsedCommand.count * elementSize);
    //         pipeline.device.queue.writeBuffer(buffer, 0, zeros);
    //     } else if (parsedCommand.type === "BLACK") {
    //         const size = parsedCommand.width * parsedCommand.height;
    //         const elementSize = 4; // Assuming 4 bytes per element (e.g., float) TODO: infer from type.
    //         const bindingInfo = resourceBindings.get(resourceName);
    //         if (!bindingInfo) {
    //             throw new Error(`Resource ${resourceName} is not defined in the bindings.`);
    //         }

    //         if (!bindingInfo.texture && !bindingInfo.storageTexture) {
    //             throw new Error(`Resource ${resourceName} is an invalid type for BLACK`);
    //         }
    //         try {
    //             let usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
    //             if (bindingInfo.storageTexture) {
    //                 usage |= GPUTextureUsage.STORAGE_BINDING;
    //             }
    //             const texture = pipeline.device.createTexture({
    //                 size: [parsedCommand.width, parsedCommand.height],
    //                 format: bindingInfo.storageTexture ? 'r32float' : 'rgba8unorm',
    //                 usage: usage,
    //             });

    //             safeSet(allocatedResources, resourceName, texture);

    //             // Initialize the texture with zeros.
    //             let zeros = new Uint8Array(Array(size * elementSize).fill(0));
    //             pipeline.device.queue.writeTexture({ texture }, zeros, { bytesPerRow: parsedCommand.width * elementSize }, { width: parsedCommand.width, height: parsedCommand.height });
    //         }
    //         catch (error) {
    //             throw new Error(`Failed to create texture: ${error}`);
    //         }
    //     } else if (parsedCommand.type === "URL") {
    //         // Load image from URL and wait for it to be ready.
    //         const bindingInfo = resourceBindings.get(resourceName);

    //         if (!bindingInfo) {
    //             throw new Error(`Resource ${resourceName} is not defined in the bindings.`);
    //         }

    //         if (!bindingInfo.texture) {
    //             throw new Error(`Resource ${resourceName} is not a texture.`);
    //         }

    //         const image = new Image();
    //         try {
    //             // TODO: Pop-up a warning if the image is not CORS-enabled.
    //             // TODO: Pop-up a warning for the user to confirm that its okay to load a cross-origin image (i.e. do you trust this code..)
    //             //
    //             image.crossOrigin = "anonymous";

    //             image.src = parsedCommand.url;
    //             await image.decode();
    //         }
    //         catch (error) {
    //             throw new Error(`Failed to load & decode image from URL: ${parsedCommand.url}`);
    //         }

    //         try {
    //             const imageBitmap = await createImageBitmap(image);
    //             const texture = pipeline.device.createTexture({
    //                 size: [imageBitmap.width, imageBitmap.height],
    //                 format: 'rgba8unorm',
    //                 usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    //             });
    //             pipeline.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture: texture }, [imageBitmap.width, imageBitmap.height]);
    //             safeSet(allocatedResources, resourceName, texture);
    //         }
    //         catch (error) {
    //             throw new Error(`Failed to create texture from image: ${error}`);
    //         }
    //     } else if (parsedCommand.type === "RAND") {
    //         const elementSize = 4; // RAND is only valid for floats
    //         const bindingInfo = resourceBindings.get(resourceName);
    //         if (!bindingInfo) {
    //             throw new Error(`Resource ${resourceName} is not defined in the bindings.`);
    //         }

    //         if (!bindingInfo.buffer) {
    //             throw new Error(`Resource ${resourceName} is not defined as a buffer.`);
    //         }

    //         const buffer = pipeline.device.createBuffer({
    //             size: parsedCommand.count * elementSize,
    //             usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    //         });

    //         safeSet(allocatedResources, resourceName, buffer);

    //         // Place a call to a shader that fills the buffer with random numbers.
    //         if (!randFloatPipeline) {
    //             const randomPipeline = new ComputePipeline(pipeline.device);

    //             // Load randFloat shader code from the file.
    //             const randFloatShaderCode = await (await fetch('demos/rand_float.slang')).text();
    //             if (compiler == null) {
    //                 throw new Error("Compiler is not defined!");
    //             }
    //             const compiledResult = compiler.compile(randFloatShaderCode, "computeMain", "WGSL");
    //             if (!compiledResult) {
    //                 throw new Error("[Internal] Failed to compile randFloat shader");
    //             }

    //             let [code, layout, hashedStrings] = compiledResult;
    //             const module = pipeline.device.createShaderModule({ code: code });

    //             randomPipeline.createPipelineLayout(layout);

    //             // Create the pipeline (without resource bindings for now)
    //             randomPipeline.createPipeline(module, null);

    //             randFloatPipeline = randomPipeline;
    //         }

    //         // Dispatch a random number generation shader.
    //         {
    //             const randomPipeline = randFloatPipeline;

    //             // Alloc resources for the shader.
    //             if (!randFloatResources)
    //                 randFloatResources = new Map();

    //             randFloatResources.set("outputBuffer", buffer);

    //             if (!randFloatResources.has("seed"))
    //                 randFloatResources.set("seed",
    //                     pipeline.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));

    //             const seedBuffer = randFloatResources.get("seed") as GPUBuffer;

    //             // Set bindings on the pipeline.
    //             randFloatPipeline.createBindGroup(randFloatResources);

    //             const seedValue = new Float32Array([Math.random(), 0, 0, 0]);
    //             pipeline.device.queue.writeBuffer(seedBuffer, 0, seedValue);

    //             // Encode commands to do the computation
    //             const encoder = pipeline.device.createCommandEncoder({ label: 'compute builtin encoder' });
    //             const pass = encoder.beginComputePass({ label: 'compute builtin pass' });

    //             pass.setBindGroup(0, randomPipeline.bindGroup || null);

    //             if (randomPipeline.pipeline == undefined) {
    //                 throw new Error("Random pipeline is undefined");
    //             }
    //             pass.setPipeline(randomPipeline.pipeline);

    //             const workGroupSizeX = Math.floor((parsedCommand.count + 63) / 64);
    //             pass.dispatchWorkgroups(workGroupSizeX, 1);
    //             pass.end();

    //             // Finish encoding and submit the commands
    //             const commandBuffer = encoder.finish();
    //             pipeline.device.queue.submit([commandBuffer]);
    //             await pipeline.device.queue.onSubmittedWorkDone();
    //         }
    //     } else {
    //         // exhaustiveness check
    //         let x: never = parsedCommand;
    //         throw new Error("Invalid resource command type");
    //     }
    // }

    //
    // Some special-case allocations
    //
let current_window_size = [300, 150];//TODO

    safe_set(&mut allocated_resources, "outputTexture".to_string(), GPUResource::Texture(create_output_texture(&pipeline.device, current_window_size[0], current_window_size[1], TextureFormat::Rgba8Unorm)));

    safe_set(&mut allocated_resources, "outputBuffer".to_string(), GPUResource::Buffer(pipeline.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: 2 * 2 * 4,
        usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC),
    })));

    safe_set(&mut allocated_resources, "outputBufferRead".to_string(), GPUResource::Buffer(pipeline.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: 2 * 2 * 4,
        usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
    })));

    safe_set(&mut allocated_resources, "g_printedBuffer".to_string(), GPUResource::Buffer(pipeline.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: PRINTF_BUFFER_SIZE,
        usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC),
    })));

    safe_set(&mut allocated_resources, "printfBufferRead".to_string(), GPUResource::Buffer(pipeline.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: PRINTF_BUFFER_SIZE,
        usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
    })));

    safe_set(&mut allocated_resources, "uniformInput".to_string(), GPUResource::Buffer(pipeline.device.create_buffer(&wgpu::BufferDescriptor {
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

        let compiled = compile_entrypoints();

        let size = window.inner_size();

        let compiler = SlangCompiler::new();
        let ret = compiler.compile("imageMain".to_string());

        let surface = instance.create_surface(window.clone()).unwrap();   
        let surface_format = wgpu::TextureFormat::Rgba8Unorm;

        let mut main_compute_pipeline = ComputePipeline::new(device.clone());
        let allocated_resources = process_resource_commands(&main_compute_pipeline).await;

        let mut pass_through_pipeline = GraphicsPipeline::new(device.clone());
        let shader_module = device.create_shader_module(wgpu::include_wgsl!("passThrough.wgsl"));
        let GPUResource::Texture(input_texture) = allocated_resources.get(&"outputTexture".to_string()).unwrap() else {
            panic!("outputTexture is not a Texture");
        };
        pass_through_pipeline.create_pipeline(&shader_module, input_texture);
        pass_through_pipeline.create_bind_group();

        let main_code = compiled.get("imageMain").unwrap().as_str();
        let compute_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("imageMain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(main_code)),
        });
        main_compute_pipeline.create_pipeline_layout(ret);
        main_compute_pipeline.create_pipeline(compute_shader_module, &allocated_resources);

        let state = State {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            pass_through_pipeline,
            main_compute_pipeline,
            allocated_resources,
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
        self.main_compute_pipeline.create_bind_group(&self.allocated_resources);
    
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

        let mut time_array = [0.0; 8];
        // time_array[0] = canvasCurrentMousePos.x;
        // time_array[1] = canvasCurrentMousePos.y;
        // time_array[2] = canvasLastMouseDownPos.x;
        // time_array[3] = canvasLastMouseDownPos.y;
        // if (canvasIsMouseDown)
        //     time_array[2] = -time_array[2];
        // if (canvasMouseClicked)
        //     time_array[3] = -time_array[3];
        time_array[4] = since_the_epoch.as_millis() as u16 as f32 * 0.001;
        let Some(GPUResource::Buffer(uniform_input)) = self.allocated_resources.get("uniformInput") else {
            panic!("uniformInput doesn't exist or is of incorrect type");
        };
        self.queue.write_buffer(uniform_input, 0, bytemuck::cast_slice(&time_array));

        let mut encoder = self.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute builtin pass"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, self.main_compute_pipeline.bind_group.as_ref(), &[]);
        pass.set_pipeline(self.main_compute_pipeline.pipeline.as_ref().unwrap());

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
