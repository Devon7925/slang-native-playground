#[cfg(not(target_arch = "wasm32"))]
mod egui_tools;

use slang_playground_compiler::{CompilationResult, UniformController};
use slang_renderer::Renderer;

use std::{cell::RefCell, rc::Rc, sync::Arc};

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[cfg(debug_assertions)]
#[cfg(target_family = "wasm")]
extern crate console_error_panic_hook;

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
                    ..Default::default()
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
    state: Option<Renderer>,
    #[cfg(target_arch = "wasm32")]
    state_receiver: Option<futures::channel::oneshot::Receiver<Renderer>>,
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
            use slang_renderer::Renderer;

            let state =
                pollster::block_on(Renderer::new(window.clone(), self.compilation.take().unwrap()));
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
        state.process_event(&event);
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
