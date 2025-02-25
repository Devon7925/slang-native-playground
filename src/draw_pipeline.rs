use std::{collections::HashMap, sync::Arc};

use wgpu::{ColorWrites, FragmentState, VertexState};

use crate::{slang_compiler::ResourceCommandData, GPUResource};

pub struct DrawPipeline {
    pub device: Arc<wgpu::Device>,

    pub pipeline: Option<wgpu::RenderPipeline>,
    pipeline_layout: Option<wgpu::PipelineLayout>,

    pub bind_group: Option<wgpu::BindGroup>,

    // resource name (string) -> binding descriptor
    resource_bindings: Option<HashMap<String, wgpu::BindGroupLayoutEntry>>,
}

impl DrawPipeline {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        DrawPipeline {
            device: device,
            pipeline: None,
            pipeline_layout: None,
            bind_group: None,
            resource_bindings: None,
        }
    }

    pub fn create_pipeline_layout(
        &mut self,
        resource_descriptors: HashMap<String, wgpu::BindGroupLayoutEntry>,
    ) {
        self.resource_bindings = Some(resource_descriptors);

        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = vec![];
        for (_, binding) in self.resource_bindings.as_ref().unwrap().iter() {
            entries.push(binding.clone());
        }
        let bind_group_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some("compute pipeline bind group layout"),
            entries: entries.as_slice(),
        };

        let bind_group_layout = self
            .device
            .create_bind_group_layout(&bind_group_layout_descriptor);
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
                ..Default::default()
            });

        self.pipeline_layout = Some(layout);
    }

    pub fn create_pipeline(
        &mut self,
        shader_module: &wgpu::ShaderModule,
        resources: Option<&HashMap<String, GPUResource>>,
        resource_commands: &HashMap<String, ResourceCommandData>,
        vertex_entry_point: Option<&str>,
        fragment_entry_point: Option<&str>,
    ) {
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("graphics pipeline"),
                layout: Some(self.pipeline_layout.as_ref().unwrap()),
                cache: None,

                vertex: VertexState {
                    module: shader_module,
                    entry_point: vertex_entry_point,
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: shader_module,
                    entry_point: fragment_entry_point,
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::all(),
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
            });

        self.pipeline = Some(pipeline);

        // If resources are provided, create the bind group right away
        if let Some(resources) = resources {
            self.create_bind_group(resources, resource_commands);
        }
    }

    pub fn create_bind_group(&mut self, allocated_resources: &HashMap<String, GPUResource>, resource_commands: &HashMap<String, ResourceCommandData>) {
        let mut entries: Vec<wgpu::BindGroupEntry> = vec![];
        let mut texture_views: HashMap<&String, wgpu::TextureView> = HashMap::new();
        let mut rebound_textures = HashMap::new();
        for (name, resource) in allocated_resources.iter() {
            match resource {
                GPUResource::Buffer(_) | GPUResource::Sampler(_) => {}
                GPUResource::Texture(texture) => {
                    if let Some(ResourceCommandData::RebindForDraw { original_texture }) = resource_commands.get(name) {
                        rebound_textures.insert(original_texture, name);
                    }
                    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    texture_views.insert(name, texture_view);
                }
            }
        }
        for (name, resource) in allocated_resources.iter() {
            let Some(bind_info) = self.resource_bindings.as_ref().unwrap().get(name) else {
                continue;
            };
            match resource {
                GPUResource::Buffer(buffer) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: bind_info.binding,
                        resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
                    });
                }
                GPUResource::Texture(_) => {
                    if let Some(ResourceCommandData::RebindForDraw { original_texture }) = resource_commands.get(name) {
                        entries.push(wgpu::BindGroupEntry {
                            binding: bind_info.binding,
                            resource: wgpu::BindingResource::TextureView(
                                texture_views.get(original_texture).unwrap(),
                            ),
                        });
                        continue;
                    }
                    if let Some(replacement) = rebound_textures.get(&name) {
                        entries.push(wgpu::BindGroupEntry {
                            binding: bind_info.binding,
                            resource: wgpu::BindingResource::TextureView(
                                texture_views.get(replacement).unwrap(),
                            ),
                        });
                        continue;
                    }
                        entries.push(wgpu::BindGroupEntry {
                            binding: bind_info.binding,
                            resource: wgpu::BindingResource::TextureView(
                                texture_views.get(name).unwrap(),
                            ),
                        });
                }
                GPUResource::Sampler(sampler) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: bind_info.binding,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                }
            }
        }

        // Check that all resources are bound
        if entries.len() != self.resource_bindings.as_ref().unwrap().len() {
            let mut missing_entries: Vec<String> = vec![];
            // print out the names of the resources that aren't bound
            for (name, resource) in self.resource_bindings.as_ref().unwrap().iter() {
                if entries
                    .iter()
                    .find(|entry| entry.binding == resource.binding)
                    .is_none()
                {
                    missing_entries.push(name.clone());
                }
            }

            panic!(
                "Cannot create bind-group. The following resources are not bound: {}",
                missing_entries.join(", ")
            );
        }

        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.as_ref().unwrap().get_bind_group_layout(0),
            entries: entries.as_slice(),
        }));
    }

    pub fn begin_render_pass<'a>(
        encoder: &'a mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        let attachment = wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
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
