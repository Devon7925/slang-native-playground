mod egui_tools;

use slang_playground_compiler::UniformController;

use std::{cell::RefCell, rc::Rc, sync::Arc};

use winit::{
    event::WindowEvent, event_loop::ActiveEventLoop, window::{Window, WindowId}
};

pub struct DebugPanel {
    uniform_controllers: Rc<RefCell<Vec<UniformController>>>,
    last_frame_time: web_time::Instant,
    last_debug_frame_time: web_time::Instant,
    current_fps: f32,
    frame_time_samples: Vec<f32>,
}

pub struct DebugAppState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface<'static>,
    scale_factor: f32,
    egui_renderer: egui_tools::EguiRenderer,
    debug_panel: DebugPanel,
    window: Arc<Window>,
}

impl DebugAppState {
    pub async fn new(
        event_loop: &ActiveEventLoop,
        (width, height): (u32, u32),
        uniform_controllers: Rc<RefCell<Vec<UniformController>>>,
    ) -> Self {
        let window = event_loop
            .create_window(
                Window::default_attributes().with_title("Slang Native Playground Debug"),
            )
            .unwrap();
        let window = Arc::new(window);

        window.request_inner_size(winit::dpi::PhysicalSize::new(width, height)).take();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface!");

        let debug_panel = DebugPanel {
            uniform_controllers,
            last_frame_time: web_time::Instant::now(),
            last_debug_frame_time: web_time::Instant::now(),
            current_fps: 0.0,
            frame_time_samples: Vec::new(),
        };

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
            debug_panel,
            window,
        }
    }

    pub fn resize_surface(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn handle_input(&mut self, event: &WindowEvent) {
        self.egui_renderer.handle_input(&self.window, event);
        
        match event {
            WindowEvent::RedrawRequested => {}
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    self
                        .resize_surface(new_size.width, new_size.height);
                }
            }
            _ => (),
        }
    }

    pub fn about_to_wait(&mut self) {
            // Calculate time since last frame
            let now = web_time::Instant::now();
            let frame_time = now - self.debug_panel.last_frame_time;

            self
                .debug_panel
                .frame_time_samples
                .push(frame_time.as_secs_f32());

            // Keep a rolling average of the last 60 frames
            if self.debug_panel.frame_time_samples.len() > 60 {
                self.debug_panel.frame_time_samples.remove(0);
            }

            // Calculate average FPS
            self.debug_panel.current_fps = self.debug_panel.frame_time_samples.len()
                as f32
                / self
                    .debug_panel
                    .frame_time_samples
                    .iter()
                    .sum::<f32>();

            let debug_frame_time = now - self.debug_panel.last_debug_frame_time;

            // Target 60 FPS (16.67ms per frame)
            let target_frame_time: std::time::Duration =
                std::time::Duration::from_secs_f32(1.0 / 60.0);
            self.debug_panel.last_frame_time = now;

            // Only redraw if enough time has passed since last frame
            if debug_frame_time >= target_frame_time {
                self.debug_panel.last_debug_frame_time = now;

                self.handle_redraw();
            }
    }

    fn handle_redraw(&mut self) {
        // Attempt to handle minimizing window
        if let Some(min) = self.window.is_minimized() {
            if min {
                println!("Window is minimized");
                return;
            }
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: self.window.scale_factor() as f32 * self.scale_factor,
        };

        let surface_texture = self.surface.get_current_texture();

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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let window = self.window.as_ref();

        {
            self.egui_renderer.begin_frame(window);
            egui::CentralPanel::default().show(self.egui_renderer.context(), |ui| {
                ui.heading(format!("FPS: {:.1}", self.debug_panel.current_fps));
                ui.separator();
                ui.heading("Uniforms:");

                for UniformController {
                    name, controller, ..
                } in self
                    .debug_panel
                    .uniform_controllers
                    .borrow_mut()
                    .iter_mut()
                {
                    controller.render(name, ui);
                }
            });

            self.egui_renderer.end_frame_and_draw(
                &self.device,
                &self.queue,
                &mut encoder,
                window,
                &surface_view,
                screen_descriptor,
            );
        }

        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }
    
    pub fn get_window_id(&self) -> WindowId {
        self.window.id()
    }
}
