use std::sync::Arc;

use wgpu::ColorWrites;

pub struct GraphicsPipeline {
    device: Arc<wgpu::Device>,
    pub pipeline: Option<wgpu::RenderPipeline>,
    sampler: Option<wgpu::Sampler>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    pub input_texture: Option<wgpu::Texture>,
    pub bind_group: Option<wgpu::BindGroup>,
}

impl GraphicsPipeline {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        GraphicsPipeline {
            device,
            pipeline: None,
            sampler: None,
            pipeline_layout: None,
            input_texture: None,
            bind_group: None,
        }
    }

    pub fn create_graphics_pipeline_layout(&mut self) {
        // Passthrough shader will need an input texture to be displayed on the screen
        let bind_group_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some("pass through pipeline bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        };

        let bind_group_layout = self
            .device
            .create_bind_group_layout(&bind_group_layout_descriptor);
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
                ..wgpu::PipelineLayoutDescriptor::default()
            });
        self.pipeline_layout = Some(layout);
    }

    pub fn create_pipeline(
        &mut self,
        shader_module: &wgpu::ShaderModule,
        input_texture: &wgpu::Texture,
    ) {
        self.create_graphics_pipeline_layout();

        if self.pipeline_layout == None {
            panic!("Pipeline layout not available")
        }

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pass through pipeline"),
                layout: self.pipeline_layout.as_ref(),
                vertex: wgpu::VertexState {
                    module: shader_module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader_module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::all(),
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        self.pipeline = Some(pipeline);

        self.sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
            ..Default::default()
        }));
        self.input_texture = Some(input_texture.clone());
        self.create_bind_group();
    }

    pub fn create_bind_group(&mut self) {
        if self.pipeline == None {
            panic!("Pipeline not created yet")
        }
        if self.sampler == None {
            panic!("Sampler not created yet")
        }
        if self.input_texture == None {
            panic!("Input texture not created yet")
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass through pipeline bind group"),
            layout: &self.pipeline.as_ref().unwrap().get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(self.sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .input_texture
                            .as_ref()
                            .unwrap()
                            .create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }

    pub fn begin_render_pass<'a>(&mut self, encoder: &'a mut wgpu::CommandEncoder, view: &wgpu::TextureView)-> wgpu::RenderPass<'a> {
        let attachment = wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.3,
                    g: 0.3,
                    b: 0.3,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        };
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("pass through renderPass"),
            color_attachments: &[Some(attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        encoder.begin_render_pass(&render_pass_descriptor)
    }
}
