mod controllers;
#[cfg(feature = "compilation")]
pub mod slang_compile;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use wgpu::BindGroupLayoutEntry;
use winit::keyboard::Key;

#[cfg(feature = "compilation")]
use slang_reflector::{
    UserAttributeParameter, VariableReflectionType,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ResourceCommandData {
    Zeros {
        count: u32,
        element_size: u32,
    },
    Rand(u32),
    Black {
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    },
    Black3D {
        size_x: u32,
        size_y: u32,
        size_z: u32,
        format: wgpu::TextureFormat,
    },
    BlackScreen {
        width_scale: f32,
        height_scale: f32,
        format: wgpu::TextureFormat,
    },
    Url {
        data: Vec<u8>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    },
    Model {
        data: Vec<u8>,
    },
    Sampler,
    RebindForDraw {
        original_resource: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResourceCommand {
    pub resource_name: String,
    pub command_data: ResourceCommandData,
}

pub struct UniformSourceData<'a> {
    pub launch_time: web_time::Instant,
    pub delta_time: f32,
    pub last_mouse_down_pos: [f32; 2],
    pub last_mouse_clicked_pos: [f32; 2],
    pub mouse_down: bool,
    pub mouse_clicked: bool,
    pub pressed_keys: &'a HashSet<Key>,
}

impl<'a> UniformSourceData<'a> {
    pub fn new(keys: &'a HashSet<Key>) -> Self {
        Self {
            launch_time: web_time::Instant::now(),
            delta_time: 0.0,
            last_mouse_down_pos: [0.0, 0.0],
            last_mouse_clicked_pos: [0.0, 0.0],
            mouse_down: false,
            mouse_clicked: false,
            pressed_keys: keys,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformController {
    pub name: String,
    pub buffer_offset: usize,
    pub controller: Box<dyn UniformControllerType>,
}

#[typetag::serde(tag = "type")]
pub trait UniformControllerType {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8>;
    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, _name: &str, _ui: &mut egui::Ui) {}

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
    ) 
    where
        Self: Sized + 'static {
            set.insert(Self::playground_name(), Box::new(Self::construct));
    }
}

pub struct CompilationResult {
    pub out_code: String,
    pub entry_group_sizes: HashMap<String, [u64; 3]>,
    pub bindings: HashMap<String, BindGroupLayoutEntry>,
    pub resource_commands: HashMap<String, ResourceCommandData>,
    pub call_commands: Vec<CallCommand>,
    pub draw_commands: Vec<DrawCommand>,
    pub hashed_strings: HashMap<u32, String>,
    pub uniform_size: u64,
    pub uniform_controllers: Vec<UniformController>,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum CallCommandParameters {
    ResourceBased(String, Option<u32>),
    FixedSize(Vec<u32>),
    Indirect(String, u32),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CallCommand {
    pub function: String,
    pub call_once: bool,
    pub parameters: CallCommandParameters,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DrawCommand {
    pub vertex_count: u32,
    pub vertex_entrypoint: String,
    pub fragment_entrypoint: String,
}
