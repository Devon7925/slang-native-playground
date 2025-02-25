use std::{collections::HashMap, sync::Arc};

use crate::GPUResource;

pub struct ComputePipeline {
    pub device: Arc<wgpu::Device>,

    pub pipeline: Option<wgpu::ComputePipeline>,
    pipeline_layout: Option<wgpu::PipelineLayout>,

    pub bind_group: Option<wgpu::BindGroup>,

    // // thread group size (array of 3 integers)
    pub thread_group_size: Option<[u64; 3]>,

    // resource name (string) -> binding descriptor
    resource_bindings: Option<HashMap<String, wgpu::BindGroupLayoutEntry>>,
}

impl ComputePipeline {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        ComputePipeline {
            device: device,
            pipeline: None,
            pipeline_layout: None,
            bind_group: None,
            thread_group_size: None,
            resource_bindings: None,
        }
    }

    pub fn set_thread_group_size(&mut self, size: [u64; 3]) {
        self.thread_group_size = Some(size);
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
        entry_point: Option<&str>,
    ) {
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute pipeline"),
                layout: Some(self.pipeline_layout.as_ref().unwrap()),
                module: shader_module,
                cache: None,
                entry_point,
                compilation_options: Default::default(),
            });

        self.pipeline = Some(pipeline);

        // If resources are provided, create the bind group right away
        if let Some(resources) = resources {
            self.create_bind_group(resources);
        }
    }

    pub fn create_bind_group(&mut self, allocated_resources: &HashMap<String, GPUResource>) {
        let mut entries: Vec<wgpu::BindGroupEntry> = vec![];
        let mut texture_views: Vec<wgpu::TextureView> = vec![];
        for (_, resource) in allocated_resources.iter() {
            match resource {
                GPUResource::Buffer(_) | GPUResource::Sampler(_) => {}
                GPUResource::Texture(texture) => {
                    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    texture_views.push(texture_view);
                }
            }
        }
        let mut tex_idx = 0;
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
                    entries.push(wgpu::BindGroupEntry {
                        binding: bind_info.binding,
                        resource: wgpu::BindingResource::TextureView(
                            texture_views.get(tex_idx).unwrap(),
                        ),
                    });
                    tex_idx += 1;
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
}
