mod resource_commands;
#[cfg(feature = "compilation")]
pub mod slang_compile;
mod uniform_controllers;

use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use wgpu::BindGroupLayoutEntry;
use winit::dpi::PhysicalSize;

#[cfg(feature = "compilation")]
pub use slang_reflector;
#[cfg(feature = "compilation")]
use slang_reflector::{BoundResource, UserAttributeParameter, VariableReflectionType};

#[derive(PartialEq)]
pub enum ResourceMetadata {
    Indirect,
}

#[derive(Debug, Clone)]
pub enum GPUResource {
    Texture(wgpu::Texture),
    Buffer(wgpu::Buffer),
    Sampler(wgpu::Sampler),
}

impl GPUResource {
    pub fn destroy(&mut self) {
        match self {
            GPUResource::Texture(texture) => texture.destroy(),
            GPUResource::Buffer(buffer) => buffer.destroy(),
            GPUResource::Sampler(_) => {}
        }
    }
}

pub struct GraphicsAPI<'a> {
    pub queue: &'a wgpu::Queue,
    pub device: &'a wgpu::Device,
    pub resource_bindings: &'a HashMap<String, wgpu::BindGroupLayoutEntry>,
    pub allocated_resources: &'a mut HashMap<String, GPUResource>,
}

#[typetag::serde(tag = "type")]
pub trait ResourceCommandData: Send + Sync + std::fmt::Debug + DynClone {
    fn is_available_in_compute(&self) -> bool {
        true
    }

    #[cfg(feature = "compilation")]
    fn generate_binding(&self) -> Option<VariableReflectionType> {
        None
    }

    fn get_rebind_original_resource(&self) -> Option<&String> {
        None
    }

    fn handle_resize(
        &self,
        _api: GraphicsAPI,
        _resource_name: &String,
        _new_size: PhysicalSize<u32>,
    ) {
    }

    fn handle_update(&self, _api: GraphicsAPI, _resource_name: &String, _data: &[u8]) {}

    #[cfg(feature = "compilation")]
    fn playground_name() -> String
    where
        Self: Sized;
    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData>
    where
        Self: Sized;

    #[cfg(feature = "compilation")]
    fn register(
        set: &mut HashMap<
            String,
            Box<
                dyn Fn(
                    &BoundResource,
                    &[UserAttributeParameter],
                    &str,
                ) -> Box<dyn ResourceCommandData>,
            >,
        >,
    ) where
        Self: Sized + 'static,
    {
        set.insert(Self::playground_name(), Box::new(Self::construct));
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()>;
}

dyn_clone::clone_trait_object!(ResourceCommandData);

pub struct UniformSourceData<'a> {
    pub launch_time: web_time::Instant,
    pub delta_time: f32,
    pub last_mouse_down_pos: [f32; 2],
    pub last_mouse_clicked_pos: [f32; 2],
    pub mouse_down: bool,
    pub mouse_clicked: bool,
    pub pressed_keys: &'a HashSet<String>,
    pub frame_count: u64,
    pub window_size: [u32; 2],
}

impl<'a> UniformSourceData<'a> {
    pub fn new(keys: &'a HashSet<String>) -> Self {
        Self {
            launch_time: web_time::Instant::now(),
            delta_time: 0.0,
            last_mouse_down_pos: [0.0, 0.0],
            last_mouse_clicked_pos: [0.0, 0.0],
            mouse_down: false,
            mouse_clicked: false,
            pressed_keys: keys,
            frame_count: 0,
            window_size: [0, 0],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UniformController {
    pub name: String,
    pub buffer_offset: usize,
    pub controller: Box<dyn UniformControllerType>,
}

#[typetag::serde(tag = "type")]
pub trait UniformControllerType: Send + Sync + std::fmt::Debug + DynClone {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8>;
    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, _name: &str, _ui: &mut egui::Ui) {}

    /// Allow external updates to a uniform controller from the host. Default no-op.
    fn handle_update(&mut self, _data: &[u8]) {}

    #[cfg(feature = "compilation")]
    fn generate_binding(&self) -> Option<VariableReflectionType> {
        None
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String
    where
        Self: Sized;

    #[cfg(feature = "compilation")]
    fn construct(
        uniform_type: &VariableReflectionType,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType>
    where
        Self: Sized;

    #[cfg(feature = "compilation")]
    fn register(
        set: &mut HashMap<
            String,
            Box<
                dyn Fn(
                    &VariableReflectionType,
                    &[UserAttributeParameter],
                    &str,
                ) -> Box<dyn UniformControllerType>,
            >,
        >,
    ) where
        Self: Sized + 'static,
    {
        set.insert(Self::playground_name(), Box::new(Self::construct));
    }
}

dyn_clone::clone_trait_object!(UniformControllerType);

#[derive(Clone, Debug)]
pub struct CompilationResult {
    pub out_code: String,
    pub entry_group_sizes: HashMap<String, [u64; 3]>,
    pub bindings: HashMap<String, BindGroupLayoutEntry>,
    pub resource_commands: HashMap<String, Box<dyn ResourceCommandData>>,
    pub call_commands: Vec<CallCommand>,
    pub draw_commands: Vec<DrawCommand>,
    pub hashed_strings: HashMap<u32, String>,
    pub uniform_size: u64,
    pub uniform_controllers: Vec<UniformController>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum CallCommandParameters {
    ResourceBased(String, Option<u32>),
    FixedSize(Vec<u32>),
    Indirect(String, u32),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CallCommand {
    pub function: String,
    pub call_once: bool,
    pub parameters: CallCommandParameters,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DrawCommand {
    pub vertex_count: u32,
    pub vertex_entrypoint: String,
    pub fragment_entrypoint: String,
}

pub fn safe_set<K: Into<String>>(
    map: &mut HashMap<String, GPUResource>,
    key: K,
    value: GPUResource,
) {
    let string_key = key.into();
    if let Some(current_entry) = map.get_mut(&string_key) {
        current_entry.destroy();
    }
    map.insert(string_key, value);
}
