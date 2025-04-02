use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use wgpu::BindGroupLayoutEntry;
use winit::keyboard::Key;

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
}
#[derive(Deserialize, Serialize)]
pub struct UniformSlider {
    pub value: f32,
    pub min: f32,
    pub max: f32,
}

#[typetag::serde]
impl UniformControllerType for UniformSlider {
    fn get_data(&self, _uniform_source_data: &UniformSourceData) -> Vec<u8> {
        self.value.to_le_bytes().to_vec()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.label(name);
        ui.add(egui::Slider::new(&mut self.value, self.min..=self.max));
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformColorPick {
    pub value: [f32; 3],
}

#[typetag::serde]
impl UniformControllerType for UniformColorPick {
    fn get_data(&self, _uniform_source_data: &UniformSourceData) -> Vec<u8> {
        self.value.iter().map(|x| x.to_le_bytes()).flatten().collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.label(name);
        ui.color_edit_button_rgb(&mut self.value);
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformMousePosition;

#[typetag::serde]
impl UniformControllerType for UniformMousePosition {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        let mut data = vec![0.0f32; 4];
        data[0] = uniform_source_data.last_mouse_down_pos[0];
        data[1] = uniform_source_data.last_mouse_down_pos[1];
        data[2] = uniform_source_data.last_mouse_clicked_pos[0];
        data[3] = uniform_source_data.last_mouse_clicked_pos[1];
        if uniform_source_data.mouse_down {
            data[2] = -data[2];
        }
        if uniform_source_data.mouse_clicked {
            data[3] = -data[3];
        }
        data.iter().map(|x| x.to_le_bytes()).flatten().collect()
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformTime;

#[typetag::serde]
impl UniformControllerType for UniformTime {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        let value = web_time::Instant::now()
            .duration_since(uniform_source_data.launch_time)
            .as_secs_f32();
        value.to_le_bytes().to_vec()
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformDeltaTime;

#[typetag::serde]
impl UniformControllerType for UniformDeltaTime {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        uniform_source_data.delta_time.to_le_bytes().to_vec()
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformKeyInput {
    pub key: String,
}

#[typetag::serde]
impl UniformControllerType for UniformKeyInput {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        let keycode = match self.key.to_lowercase().as_str() {
            "enter" => Key::Named(winit::keyboard::NamedKey::Enter),
            "space" => Key::Named(winit::keyboard::NamedKey::Space),
            "shift" => Key::Named(winit::keyboard::NamedKey::Shift),
            "ctrl" => Key::Named(winit::keyboard::NamedKey::Control),
            "escape" => Key::Named(winit::keyboard::NamedKey::Escape),
            "backspace" => Key::Named(winit::keyboard::NamedKey::Backspace),
            "tab" => Key::Named(winit::keyboard::NamedKey::Tab),
            "arrowup" => Key::Named(winit::keyboard::NamedKey::ArrowUp),
            "arrowdown" => Key::Named(winit::keyboard::NamedKey::ArrowDown),
            "arrowleft" => Key::Named(winit::keyboard::NamedKey::ArrowLeft),
            "arrowright" => Key::Named(winit::keyboard::NamedKey::ArrowRight),
            k => Key::Character(k.into()),
        };
        let value = if uniform_source_data.pressed_keys.contains(&keycode) {
            1.0f32
        } else {
            0.0f32
        };
        value.to_le_bytes().to_vec()
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
