use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use wgpu::BindGroupLayoutEntry;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ResourceCommandData {
    ZEROS {
        count: u32,
        element_size: u32,
    },
    RAND(u32),
    BLACK {
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    },
    BlackScreen {
        width_scale: f32,
        height_scale: f32,
        format: wgpu::TextureFormat,
    },
    URL {
        url: String,
        format: wgpu::TextureFormat,
    },
    Sampler,
    RebindForDraw {
        original_texture: String,
    },
    SLIDER {
        default: f32,
        min: f32,
        max: f32,
        element_size: usize,
        offset: usize,
    },
    COLORPICK {
        default: [f32; 3],
        element_size: usize,
        offset: usize,
    },
    MOUSEPOSITION {
        offset: usize,
    },
    TIME {
        offset: usize,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResourceCommand {
    pub resource_name: String,
    pub command_data: ResourceCommandData,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UniformController {
    pub name: String,
    pub buffer_offset: usize,
    pub controller: UniformControllerType,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum UniformControllerType {
    SLIDER {
        value: f32,
        min: f32,
        max: f32,
    },
    COLORPICK {
        value: [f32; 3],
    },
    MOUSEPOSITION,
    TIME,
}

#[derive(Debug, Deserialize, Serialize)]
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
