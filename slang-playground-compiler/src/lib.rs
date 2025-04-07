mod controllers;

use controllers::*;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ops::Deref,
    rc::Rc,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use url::Url;
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

    fn playground_name() -> String
    where
        Self: Sized;

    fn construct(
        uniform_type: &VariableReflectionType,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType>
    where
        Self: Sized;

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
use slang_reflector::{
    BoundParameter, BoundResource, EntrypointReflection, GlobalSession, ProgramLayoutReflector,
    ProgramReflection, ResourceAccess, ScalarType, TextureType, UserAttributeParameter,
    VariableReflection, VariableReflectionType,
};

#[derive(EnumIter, Debug, PartialEq, Clone)]
enum ShaderType {
    Image,
    Print,
}

impl ShaderType {
    fn get_entry_point_name(&self) -> &str {
        match self {
            ShaderType::Image => "imageMain",
            ShaderType::Print => "printMain",
        }
    }
}

pub struct SlangCompiler {
    global_slang_session: GlobalSession,
    uniform_controller_constructors: HashMap<
        String,
        Box<
            dyn Fn(
                &VariableReflectionType,
                &[UserAttributeParameter],
                &str,
            ) -> Box<dyn UniformControllerType>,
        >,
    >,
}

impl Default for SlangCompiler {
    fn default() -> Self {
        let mut default_controllers = HashMap::new();

        UniformSlider::register(&mut default_controllers);
        UniformColorPick::register(&mut default_controllers);
        UniformTime::register(&mut default_controllers);
        UniformMousePosition::register(&mut default_controllers);
        UniformDeltaTime::register(&mut default_controllers);
        UniformKeyInput::register(&mut default_controllers);

        Self::new(default_controllers)
    }
}

#[derive(Clone)]
struct CustomFileSystem {
    used_files: Rc<RefCell<HashSet<String>>>,
}

impl CustomFileSystem {
    fn new() -> Self {
        Self {
            used_files: Rc::new(RefCell::new(HashSet::new())),
        }
    }

    fn get_files(&self) -> Rc<RefCell<HashSet<String>>> {
        self.used_files.clone()
    }
}

impl slang_reflector::FileSystem for CustomFileSystem {
    fn load_file(&self, path: &str) -> slang_reflector::Result<slang_reflector::Blob> {
        let mut path = path.to_string();

        // Remove automatically added path prefix for github imports
        let re = &regex::Regex::new(r"^.*/github://").unwrap();
        path = re.replace_all(path.as_str(), "github://").to_string();

        if let Some(git_path) = path.strip_prefix("github://") {
            // first 2 parts of path are the user and repo
            // Use git api to get files ex. "https://api.github.com/repos/shader-slang/slang-playground/contents/example.slang"
            let parts: Vec<&str> = git_path.split('/').collect();
            if parts.len() < 3 {
                return Err(slang_reflector::Error::Blob(slang_reflector::Blob::from(
                    "Invalid github path",
                )));
            }
            let user = parts[0];
            let repo = parts[1];
            let file_path = parts[2..].join("/");
            let url = format!(
                "https://api.github.com/repos/{}/{}/contents/{}",
                user, repo, file_path
            );

            // Set the User-Agent header to avoid 403 Forbidden error
            let client = reqwest::blocking::Client::builder()
                .user_agent("slang-playground")
                .build()
                .map_err(|e| {
                    slang_reflector::Error::Blob(slang_reflector::Blob::from(e.to_string()))
                })?;

            let mut request = client.get(&url);

            // try to get token from GITHUB_TOKEN file in repo root if possible
            let token = std::fs::read_to_string("GITHUB_TOKEN");

            if let Ok(token) = token {
                request = request.header("Authorization", format!("token {}", token));
            };

            let response = request
                .header("Accept", "application/vnd.github.v3.raw")
                .send()
                .map_err(|e| {
                    slang_reflector::Error::Blob(slang_reflector::Blob::from(e.to_string()))
                })?;

            if !response.status().is_success() {
                if response.status() == 403 {
                    println!(
                        "cargo::warning=Loading file {} failed. Possibly rate limited.",
                        path.clone()
                    );
                }

                return Err(slang_reflector::Error::Blob(slang_reflector::Blob::from(
                    format!("Failed to get file from github: {}", response.status()),
                )));
            }

            let response = response.text().map_err(|e| {
                slang_reflector::Error::Blob(slang_reflector::Blob::from(e.to_string()))
            })?;
            self.used_files.borrow_mut().insert(path.clone());
            return Ok(slang_reflector::Blob::from(response.into_bytes()));
        } else {
            match std::fs::read(&path) {
                Ok(bytes) => {
                    self.used_files.borrow_mut().insert(path);
                    Ok(slang_reflector::Blob::from(bytes))
                }
                Err(e) => Err(slang_reflector::Error::Blob(slang_reflector::Blob::from(
                    format!("Failed to read file: {}", e),
                ))),
            }
        }
    }
}

impl SlangCompiler {
    pub fn new(
        uniform_controller_constructors: HashMap<
            String,
            Box<
                dyn Fn(
                    &VariableReflectionType,
                    &[UserAttributeParameter],
                    &str,
                ) -> Box<dyn UniformControllerType>,
            >,
        >,
    ) -> Self {
        let global_slang_session = slang_reflector::GlobalSession::new().unwrap();
        SlangCompiler {
            global_slang_session,
            uniform_controller_constructors,
        }
    }

    fn add_components(
        &self,
        slang_session: &slang_reflector::Session,
        used_files: impl IntoIterator<Item = String>,
        component_list: &mut Vec<slang_reflector::ComponentType>,
    ) {
        for imported_file in used_files {
            let module = slang_session
                .load_module(&imported_file)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to load module {}: {:?}",
                        imported_file,
                        e.to_string()
                    )
                });

            component_list.push(module.deref().clone());

            for entry_point in module.entry_points() {
                component_list.push(entry_point.deref().clone());
            }
        }

        let program = slang_session
            .create_composite_component_type(component_list)
            .unwrap();
        let linked_program = program.link().unwrap();
        let shader_reflection = linked_program.layout(0).unwrap();

        for st in ShaderType::iter().map(|st| st.get_entry_point_name().to_string()) {
            if shader_reflection
                .find_function_by_name(st.as_str())
                .is_some()
            {
                let module = slang_session
                    .load_module(format!("{}.slang", st).as_str())
                    .unwrap_or_else(|e| {
                        panic!("Failed to load module {}: {:?}", st, e.to_string())
                    });

                component_list.push(module.deref().clone());

                for entry_point in module.entry_points() {
                    component_list.push(entry_point.deref().clone());
                }
            }
        }
    }

    fn is_runnable_entry_point(entry_point_name: &String) -> bool {
        ShaderType::iter().any(|st| st.get_entry_point_name() == entry_point_name)
    }

    fn get_binding_descriptor(&self, parameter: &BoundResource) -> Option<wgpu::BindingType> {
        match parameter {
            BoundResource::Texture {
                tex_type,
                resource_access: ResourceAccess::None | ResourceAccess::Read,
                ..
            } => Some(wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: parameter_texture_view_dimension(tex_type),
            }),
            BoundResource::Texture {
                tex_type,
                resource_result,
                format,
                resource_access,
            } => Some(wgpu::BindingType::StorageTexture {
                access: match resource_access {
                    slang_reflector::ResourceAccess::Read => wgpu::StorageTextureAccess::ReadOnly,
                    slang_reflector::ResourceAccess::ReadWrite => {
                        wgpu::StorageTextureAccess::ReadWrite
                    }
                    slang_reflector::ResourceAccess::Write => wgpu::StorageTextureAccess::WriteOnly,
                    _ => panic!("Invalid resource access"),
                },
                format: get_wgpu_format_from_slang_format(format, resource_result),
                view_dimension: parameter_texture_view_dimension(tex_type),
            }),
            BoundResource::StructuredBuffer {
                resource_access: slang_reflector::ResourceAccess::Read,
                ..
            } => Some(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            BoundResource::StructuredBuffer {
                resource_access:
                    slang_reflector::ResourceAccess::Write | slang_reflector::ResourceAccess::ReadWrite,
                ..
            } => Some(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            BoundResource::Sampler => Some(wgpu::BindingType::Sampler(
                wgpu::SamplerBindingType::NonFiltering,
            )),
            a => {
                println!("cargo::warning=Could not generate binding for {:?}", a);
                None
            }
        }
    }

    fn get_resource_bindings(
        &self,
        reflection: &ProgramReflection,
        resource_commands: &HashMap<String, ResourceCommandData>,
    ) -> HashMap<String, wgpu::BindGroupLayoutEntry> {
        let mut resource_descriptors = HashMap::new();
        let mut uniform_input = false;
        for VariableReflection {
            name,
            reflection_type,
            ..
        } in reflection.variables.iter()
        {
            let BoundParameter::Resource {
                resource,
                binding_index,
            } = reflection_type
            else {
                uniform_input = true;
                continue;
            };

            let resource_info = self.get_binding_descriptor(resource);
            let mut visibility = wgpu::ShaderStages::NONE;
            if resource_commands
                .get(name)
                .map(is_available_in_compute)
                .unwrap_or(true)
            {
                visibility |= wgpu::ShaderStages::COMPUTE;
            }
            if is_available_in_graphics(resource) {
                visibility |= wgpu::ShaderStages::VERTEX_FRAGMENT;
            }
            let binding = wgpu::BindGroupLayoutEntry {
                ty: resource_info.unwrap(),
                binding: *binding_index,
                visibility,
                count: None,
            };

            resource_descriptors.insert(name.clone(), binding);
        }
        if uniform_input {
            resource_descriptors.insert(
                "uniformInput".to_string(),
                wgpu::BindGroupLayoutEntry {
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
                    count: None,
                },
            );
        }

        resource_descriptors
    }

    fn resource_commands_from_attributes(
        &self,
        reflection: &Vec<VariableReflection>,
    ) -> HashMap<String, ResourceCommandData> {
        let mut commands: HashMap<String, ResourceCommandData> = HashMap::new();

        for VariableReflection {
            reflection_type,
            name,
            user_attributes,
        } in reflection
        {
            let BoundParameter::Resource { resource, .. } = reflection_type else {
                continue;
            };
            for attribute in user_attributes {
                let Some(playground_attribute_name) = attribute.name.strip_prefix("playground_")
                else {
                    continue;
                };
                let command = if playground_attribute_name == "ZEROS" {
                    let BoundResource::StructuredBuffer {
                        resource_result: element_type,
                        ..
                    } = resource
                    else {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, it only supports buffers"
                        )
                    };
                    let [UserAttributeParameter::Int(count)] = attribute.parameters[..] else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };
                    assert!(
                        count >= 0,
                        "{playground_attribute_name} count for {name} cannot have negative size",
                    );
                    Some(ResourceCommandData::Zeros {
                        count: count as u32,
                        element_size: element_type.get_size(),
                    })
                } else if playground_attribute_name == "SAMPLER" {
                    if !matches!(resource, BoundResource::Sampler,) {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, it only supports samplers",
                        )
                    }
                    Some(ResourceCommandData::Sampler)
                } else if playground_attribute_name == "RAND" {
                    assert!(
                        matches!(resource, BoundResource::StructuredBuffer{ resource_result: element, ..} if matches!(element, VariableReflectionType::Scalar(ScalarType::Float32))),
                        "{playground_attribute_name} attribute cannot be applied to {name}, it only supports float buffers"
                    );
                    let [UserAttributeParameter::Int(count)] = attribute.parameters[..] else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };
                    assert!(
                        count >= 0,
                        "{playground_attribute_name} count for {name} cannot have negative size",
                    );
                    Some(ResourceCommandData::Rand(count as u32))
                } else if playground_attribute_name == "BLACK" {
                    let BoundResource::Texture {
                        tex_type: TextureType::Dim2,
                        resource_result: resource_type,
                        format,
                        ..
                    } = resource
                    else {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, it only supports 2D textures",
                        )
                    };

                    let [
                        UserAttributeParameter::Int(width),
                        UserAttributeParameter::Int(height),
                    ] = attribute.parameters[..]
                    else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };

                    assert!(
                        width >= 0,
                        "{playground_attribute_name} width for {name} cannot have negative size",
                    );
                    assert!(
                        height >= 0,
                        "{playground_attribute_name} height for {name} cannot have negative size",
                    );

                    Some(ResourceCommandData::Black {
                        width: width as u32,
                        height: height as u32,
                        format: get_wgpu_format_from_slang_format(format, resource_type),
                    })
                } else if playground_attribute_name == "BLACK_3D" {
                    let BoundResource::Texture {
                        tex_type: TextureType::Dim3,
                        resource_result: resource_type,
                        format,
                        ..
                    } = resource
                    else {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, it only supports 3D textures",
                        )
                    };

                    let [
                        UserAttributeParameter::Int(size_x),
                        UserAttributeParameter::Int(size_y),
                        UserAttributeParameter::Int(size_z),
                    ] = attribute.parameters[..]
                    else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };

                    macro_rules! check_positive {
                        ($id:ident) => {
                            if $id < 0 {
                                panic!(
                                    "{playground_attribute_name} {} for {name} cannot have negative size",
                                    stringify!($id),
                                )
                            }
                        };
                    }

                    check_positive!(size_x);
                    check_positive!(size_y);
                    check_positive!(size_z);

                    Some(ResourceCommandData::Black3D {
                        size_x: size_x as u32,
                        size_y: size_y as u32,
                        size_z: size_z as u32,
                        format: get_wgpu_format_from_slang_format(format, resource_type),
                    })
                } else if playground_attribute_name == "BLACK_SCREEN" {
                    let BoundResource::Texture {
                        tex_type: TextureType::Dim2,
                        resource_result: resource_type,
                        format,
                        ..
                    } = resource
                    else {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, it only supports 2D textures",
                        )
                    };

                    let [
                        UserAttributeParameter::Float(width_scale),
                        UserAttributeParameter::Float(height_scale),
                    ] = attribute.parameters[..]
                    else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };

                    assert!(
                        width_scale >= 0.0,
                        "{playground_attribute_name} width for {name} cannot have negative size",
                    );
                    assert!(
                        height_scale >= 0.0,
                        "{playground_attribute_name} height for {name} cannot have negative size",
                    );

                    Some(ResourceCommandData::BlackScreen {
                        width_scale,
                        height_scale,
                        format: get_wgpu_format_from_slang_format(format, resource_type),
                    })
                } else if playground_attribute_name == "URL" {
                    let BoundResource::Texture {
                        tex_type: TextureType::Dim2,
                        resource_result: resource_type,
                        format,
                        ..
                    } = resource
                    else {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, it only supports 2D textures",
                        )
                    };

                    let format = get_wgpu_format_from_slang_format(format, resource_type);

                    let [UserAttributeParameter::String(url)] = &attribute.parameters[..] else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };

                    let parsed_url = Url::parse(&url);
                    let image_bytes =
                        if let Err(url::ParseError::RelativeUrlWithoutBase) = parsed_url {
                            std::fs::read(url).unwrap()
                        } else {
                            reqwest::blocking::get(parsed_url.unwrap())
                                .unwrap()
                                .bytes()
                                .unwrap()
                                .to_vec()
                        };
                    let image = image::load_from_memory(&image_bytes).unwrap();
                    let data = match format {
                        wgpu::TextureFormat::Rgba8Unorm => image.to_rgba8().to_vec(),
                        wgpu::TextureFormat::R8Unorm => image
                            .to_rgba8()
                            .to_vec()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (*i % 4) < 1)
                            .map(|(_, c)| c)
                            .cloned()
                            .collect(),
                        wgpu::TextureFormat::Rg8Unorm => image
                            .to_rgba8()
                            .to_vec()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (*i % 4) < 2)
                            .map(|(_, c)| c)
                            .cloned()
                            .collect(),
                        wgpu::TextureFormat::Rgba32Float => image
                            .to_rgba32f()
                            .to_vec()
                            .iter()
                            .flat_map(|c| c.to_le_bytes())
                            .collect(),
                        wgpu::TextureFormat::Rg32Float => image
                            .to_rgba32f()
                            .to_vec()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (*i % 4) < 2)
                            .map(|(_, c)| c)
                            .flat_map(|c| c.to_le_bytes())
                            .collect(),
                        f => panic!("URL unimplemented for image format {f:?}"),
                    };

                    Some(ResourceCommandData::Url {
                        data,
                        width: image.width(),
                        height: image.height(),
                        format,
                    })
                } else if playground_attribute_name == "MODEL" {
                    let BoundResource::StructuredBuffer {
                        resource_result: element_type,
                        ..
                    } = resource
                    else {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, it only supports buffers",
                        )
                    };
                    let VariableReflectionType::Struct(fields) = element_type else {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {name}, inner type must be struct",
                        )
                    };
                    let mut field_types = vec![];
                    for (field_name, field) in fields {
                        match field_name.as_str() {
                            "position" => {
                                if !matches!(
                                    field,
                                    VariableReflectionType::Vector(ScalarType::Float32, 3)
                                ) {
                                    panic!(
                                        "Unsupported type for {field_name} field of MODEL struct for {name}"
                                    )
                                }
                                field_types.push(ModelField::Position)
                            }
                            "normal" => {
                                if !matches!(
                                    field,
                                    VariableReflectionType::Vector(ScalarType::Float32, 3)
                                ) {
                                    panic!(
                                        "Unsupported type for {field_name} field of MODEL struct for {name}"
                                    )
                                }
                                field_types.push(ModelField::Normal)
                            }
                            "uv" => {
                                if !matches!(
                                    field,
                                    VariableReflectionType::Vector(ScalarType::Float32, 3)
                                ) {
                                    panic!(
                                        "Unsupported type for {field_name} field of MODEL struct for {name}"
                                    )
                                }
                                field_types.push(ModelField::TexCoords)
                            }
                            field_name => panic!(
                                "{field_name} is not a valid field for MODEL attribute on {name}, valid fields are: position, normal, uv",
                            ),
                        }
                    }

                    let [UserAttributeParameter::String(path)] = &attribute.parameters[..] else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };

                    // load obj file from path
                    let (models, _) = tobj::load_obj(
                        &path,
                        &tobj::LoadOptions {
                            triangulate: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let mut data: Vec<u8> = Vec::new();
                    for model in models {
                        let mesh = &model.mesh;
                        println!(
                            "cargo::warning=Loading mesh with {} verticies",
                            mesh.indices.len()
                        );
                        for i in 0..mesh.indices.len() {
                            for field_type in field_types.iter() {
                                match field_type {
                                    ModelField::Position => {
                                        data.extend_from_slice(
                                            &mesh.positions[3 * mesh.indices[i] as usize]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(
                                            &mesh.positions[3 * mesh.indices[i] as usize + 1]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(
                                            &mesh.positions[3 * mesh.indices[i] as usize + 2]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(&1.0f32.to_le_bytes());
                                    }
                                    ModelField::Normal => {
                                        data.extend_from_slice(
                                            &mesh.normals[3 * mesh.normal_indices[i] as usize]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(
                                            &mesh.normals[3 * mesh.normal_indices[i] as usize + 1]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(
                                            &mesh.normals[3 * mesh.normal_indices[i] as usize + 2]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(&1.0f32.to_le_bytes());
                                    }
                                    ModelField::TexCoords => {
                                        data.extend_from_slice(
                                            &mesh.texcoords[2 * mesh.texcoord_indices[i] as usize]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(
                                            &mesh.texcoords
                                                [2 * mesh.texcoord_indices[i] as usize + 1]
                                                .to_le_bytes(),
                                        );
                                        data.extend_from_slice(&1.0f32.to_le_bytes());
                                        data.extend_from_slice(&1.0f32.to_le_bytes());
                                    }
                                }
                            }
                        }
                    }

                    Some(ResourceCommandData::Model { data })
                } else if playground_attribute_name == "REBIND_FOR_DRAW" {
                    assert!(
                        matches!(
                            resource,
                            BoundResource::Texture {
                                tex_type: TextureType::Dim2,
                                ..
                            } | BoundResource::StructuredBuffer { .. }
                        ),
                        "{playground_attribute_name} attribute cannot be applied to {name}, it only supports 2D textures and structured buffers",
                    );

                    let [UserAttributeParameter::String(original_resource)] =
                        &attribute.parameters[..]
                    else {
                        panic!(
                            "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                        )
                    };

                    Some(ResourceCommandData::RebindForDraw {
                        original_resource: original_resource.clone(),
                    })
                } else {
                    None
                };

                if let Some(command) = command {
                    commands.insert(name.clone(), command);
                }
            }
        }

        commands
    }
    fn uniform_controllers_from_attributes(
        &self,
        reflection: &Vec<VariableReflection>,
    ) -> Vec<UniformController> {
        let mut commands: Vec<UniformController> = Vec::new();

        for VariableReflection {
            reflection_type,
            name,
            user_attributes,
        } in reflection
        {
            let BoundParameter::Uniform {
                uniform_offset,
                resource_result,
            } = reflection_type
            else {
                continue;
            };
            for attribute in user_attributes {
                let Some(playground_attribute_name) = attribute.name.strip_prefix("playground_")
                else {
                    continue;
                };
                
                if let Some(constructor) = self.uniform_controller_constructors.get(playground_attribute_name) {
                    let controller = constructor(resource_result, &attribute.parameters, name);
                    commands.push(UniformController {
                        name: name.to_string(),
                        buffer_offset: *uniform_offset,
                        controller,
                    });
                }
            }
        }

        commands
    }

    pub fn compile(&self, search_paths: Vec<&str>, entry_module_name: &str) -> CompilationResult {
        let search_paths = search_paths
            .into_iter()
            .map(std::ffi::CString::new)
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        let search_paths = search_paths.iter().map(|p| p.as_ptr()).collect::<Vec<_>>();

        let session_options = slang_reflector::CompilerOptions::default()
            .optimization(slang_reflector::OptimizationLevel::High)
            .matrix_layout_row(true);

        let target_desc = slang_reflector::TargetDesc::default()
            .format(slang_reflector::CompileTarget::Wgsl)
            .profile(self.global_slang_session.find_profile("spirv_1_6"));

        let custom_file_system = CustomFileSystem::new();

        let targets = [target_desc];

        let session_desc = slang_reflector::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_paths)
            .options(&session_options)
            .file_system(custom_file_system.clone());

        let Ok(slang_session) = self.global_slang_session.create_session(&session_desc) else {
            panic!("Failed to create slang session");
        };

        let mut components: Vec<slang_reflector::ComponentType> = vec![];

        let user_module = slang_session
            .load_module(entry_module_name)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to load module {}: {:?}",
                    entry_module_name,
                    e.to_string()
                )
            });

        let count = user_module.entry_point_count();
        for i in 0..count {
            let name = user_module
                .entry_point_by_index(i)
                .unwrap()
                .function_reflection()
                .name()
                .to_string();
            if SlangCompiler::is_runnable_entry_point(&name) {
                panic!("User defined playground entrypoint {}", name)
            }
        }

        let used_files = custom_file_system.get_files().borrow().clone();
        self.add_components(&slang_session, used_files, &mut components);

        let program = slang_session
            .create_composite_component_type(components.as_slice())
            .unwrap();
        let linked_program = program.link().unwrap();

        let shader_reflection = linked_program.layout(0).unwrap();
        let out_code = linked_program
            .target_code(0)
            .unwrap_or_else(|err| panic!("Failed to compile shader: {:?}", err.to_string()))
            .as_slice()
            .to_vec();
        let out_code = String::from_utf8(out_code).unwrap();

        let mut entry_group_sizes = HashMap::new();
        for entry in shader_reflection.entry_points() {
            let group_size = entry.compute_thread_group_size();
            if entry.stage() == slang_reflector::Stage::Compute {
                entry_group_sizes.insert(entry.name().to_string(), group_size);
            }
        }

        let global_reflection = shader_reflection.reflect();

        let resource_commands =
            self.resource_commands_from_attributes(&global_reflection.variables);
        let uniform_controllers =
            self.uniform_controllers_from_attributes(&global_reflection.variables);
        let call_commands = parse_call_commands(&global_reflection);
        let draw_commands = parse_draw_commands(&global_reflection);

        let bindings = self.get_resource_bindings(&global_reflection, &resource_commands);

        CompilationResult {
            out_code,
            entry_group_sizes,
            bindings,
            uniform_controllers,
            resource_commands,
            call_commands,
            draw_commands,
            uniform_size: get_uniform_size(&global_reflection),
            hashed_strings: global_reflection.hashed_strings,
        }
    }
}

fn parameter_texture_view_dimension(tex_type: &TextureType) -> wgpu::TextureViewDimension {
    match tex_type {
        TextureType::Dim1 => wgpu::TextureViewDimension::D1,
        TextureType::Dim2 => wgpu::TextureViewDimension::D2,
        TextureType::Dim3 => wgpu::TextureViewDimension::D3,
        TextureType::Cube => wgpu::TextureViewDimension::Cube,
    }
}

#[derive(Debug, Clone, Copy)]
enum ModelField {
    Position,
    Normal,
    TexCoords,
}

fn is_available_in_compute(resource_command: &ResourceCommandData) -> bool {
    !matches!(resource_command, ResourceCommandData::RebindForDraw { .. })
}

fn is_available_in_graphics(parameter: &BoundResource) -> bool {
    match parameter {
        BoundResource::Texture {
            resource_access: ResourceAccess::Read,
            ..
        } => true,
        BoundResource::StructuredBuffer {
            resource_access: ResourceAccess::Read,
            ..
        } => true,
        BoundResource::Sampler => true,
        _ => false,
    }
}

fn round_up_to_nearest(size: u64, arg: u64) -> u64 {
    size.div_ceil(arg) * arg
}

fn get_wgpu_format_from_slang_format(
    format: &slang_reflector::ImageFormat,
    resource_type: &VariableReflectionType,
) -> wgpu::TextureFormat {
    use slang_reflector::ImageFormat;
    use wgpu::TextureFormat;
    match format {
        ImageFormat::SLANGIMAGEFORMATUnknown => match resource_type {
            VariableReflectionType::Vector(ScalarType::Float32, 2) => TextureFormat::Rg32Float,
            VariableReflectionType::Vector(ScalarType::Float32, 3 | 4) => {
                TextureFormat::Rgba32Float
            }
            VariableReflectionType::Scalar(ScalarType::Int32) => TextureFormat::R32Sint,
            VariableReflectionType::Scalar(ScalarType::Uint32) => TextureFormat::R32Uint,
            VariableReflectionType::Scalar(ScalarType::Float32) => TextureFormat::R32Float,
            _ => panic!("Could not infer image format from resource type {resource_type:?}"),
        },
        ImageFormat::SLANGIMAGEFORMATR8Snorm => TextureFormat::R8Snorm,
        ImageFormat::SLANGIMAGEFORMATR8 => TextureFormat::R8Unorm,
        ImageFormat::SLANGIMAGEFORMATR8ui => TextureFormat::R8Uint,
        ImageFormat::SLANGIMAGEFORMATR8i => TextureFormat::R8Sint,
        ImageFormat::SLANGIMAGEFORMATR16ui => TextureFormat::R16Uint,
        ImageFormat::SLANGIMAGEFORMATR16i => TextureFormat::R16Sint,
        ImageFormat::SLANGIMAGEFORMATR16 => TextureFormat::R16Unorm,
        ImageFormat::SLANGIMAGEFORMATRg8 => TextureFormat::Rg8Unorm,
        ImageFormat::SLANGIMAGEFORMATRg8Snorm => TextureFormat::Rg8Snorm,
        ImageFormat::SLANGIMAGEFORMATRg8ui => TextureFormat::Rg8Uint,
        ImageFormat::SLANGIMAGEFORMATRg8i => TextureFormat::Rg8Sint,
        ImageFormat::SLANGIMAGEFORMATR32ui => TextureFormat::R32Uint,
        ImageFormat::SLANGIMAGEFORMATR32i => TextureFormat::R32Sint,
        ImageFormat::SLANGIMAGEFORMATR32f => TextureFormat::R32Float,
        ImageFormat::SLANGIMAGEFORMATRg16ui => TextureFormat::Rg16Uint,
        ImageFormat::SLANGIMAGEFORMATRg16i => TextureFormat::Rg16Sint,
        ImageFormat::SLANGIMAGEFORMATRg16 => TextureFormat::Rg16Unorm,
        ImageFormat::SLANGIMAGEFORMATRgba8Snorm => TextureFormat::Rgba8Snorm,
        ImageFormat::SLANGIMAGEFORMATRgba8ui => TextureFormat::Rgba8Uint,
        ImageFormat::SLANGIMAGEFORMATRgba8i => TextureFormat::Rgba8Sint,
        ImageFormat::SLANGIMAGEFORMATRgba8 => TextureFormat::Rgba8Unorm,
        ImageFormat::SLANGIMAGEFORMATRg32ui => TextureFormat::Rg32Uint,
        ImageFormat::SLANGIMAGEFORMATRg32i => TextureFormat::Rg32Sint,
        ImageFormat::SLANGIMAGEFORMATRg32f => TextureFormat::Rg32Float,
        ImageFormat::SLANGIMAGEFORMATRgba16ui => TextureFormat::Rgba16Uint,
        ImageFormat::SLANGIMAGEFORMATRgba16i => TextureFormat::Rgba16Sint,
        ImageFormat::SLANGIMAGEFORMATRgba16 => TextureFormat::Rgba16Unorm,
        ImageFormat::SLANGIMAGEFORMATRgba32ui => TextureFormat::Rgba32Uint,
        ImageFormat::SLANGIMAGEFORMATRgba32i => TextureFormat::Rgba32Sint,
        ImageFormat::SLANGIMAGEFORMATRgba32f => TextureFormat::Rgba32Float,
        f => panic!("Unsupported image format {f:?}"),
    }
}

fn parse_call_commands(reflection: &ProgramReflection) -> Vec<CallCommand> {
    let mut call_commands: Vec<CallCommand> = vec![];
    for EntrypointReflection {
        name,
        user_attributes,
    } in &reflection.entry_points
    {
        let mut call_command = None;
        let mut call_once = false;
        for attribute in user_attributes {
            let Some(playground_attribute_name) = attribute.name.strip_prefix("playground_") else {
                continue;
            };
            if playground_attribute_name == "CALL_SIZE_OF" {
                if call_command.is_some() {
                    panic!("Cannot have multiple CALL attributes for the same function");
                }

                let [UserAttributeParameter::String(resource_name)] = &attribute.parameters[..]
                else {
                    panic!(
                        "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                    )
                };

                let resource_reflection = reflection
                    .variables
                    .iter()
                    .find(|param| &param.name == resource_name)
                    .unwrap();

                let mut element_size: Option<u32> = None;
                if let VariableReflection {
                    reflection_type:
                        BoundParameter::Resource {
                            resource:
                                BoundResource::StructuredBuffer {
                                    resource_result, ..
                                },
                            ..
                        },
                    ..
                } = resource_reflection
                {
                    element_size = Some(resource_result.get_size());
                }

                call_command = Some(CallCommand {
                    function: name.clone(),
                    call_once: false,
                    parameters: CallCommandParameters::ResourceBased(
                        resource_name.to_string(),
                        element_size,
                    ),
                });
            } else if playground_attribute_name == "CALL" {
                if call_command.is_some() {
                    panic!("Cannot have multiple CALL attributes for the same function");
                }

                let [
                    UserAttributeParameter::Int(size_x),
                    UserAttributeParameter::Int(size_y),
                    UserAttributeParameter::Int(size_z),
                ] = attribute.parameters[..]
                else {
                    panic!(
                        "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                    )
                };

                call_command = Some(CallCommand {
                    function: name.clone(),
                    call_once: false,
                    parameters: CallCommandParameters::FixedSize(vec![
                        size_x as u32,
                        size_y as u32,
                        size_z as u32,
                    ]),
                });
            } else if playground_attribute_name == "CALL_INDIRECT" {
                if call_command.is_some() {
                    panic!("Cannot have multiple CALL attributes for the same function");
                }

                let [
                    UserAttributeParameter::String(resource_name),
                    UserAttributeParameter::Int(offset),
                ] = &attribute.parameters[..]
                else {
                    panic!(
                        "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                    )
                };

                let resource_reflection = reflection
                    .variables
                    .iter()
                    .find(|param| &param.name == resource_name)
                    .unwrap();

                assert!(
                    matches!(
                        resource_reflection,
                        VariableReflection {
                            reflection_type: BoundParameter::Resource {
                                resource: BoundResource::StructuredBuffer { .. },
                                ..
                            },
                            ..
                        }
                    ),
                    "Invalid type for CALL_INDIRECT buffer"
                );

                call_command = Some(CallCommand {
                    function: name.clone(),
                    call_once: false,
                    parameters: CallCommandParameters::Indirect(
                        resource_name.clone(),
                        *offset as u32,
                    ),
                });
            } else if playground_attribute_name == "CALL_ONCE" {
                if call_once {
                    panic!("Cannot have multiple CALL_ONCE attributes for the same function");
                }
                call_once = true;
            }
        }

        if let Some(mut call_command) = call_command {
            call_command.call_once = call_once;
            call_commands.push(call_command);
        }
    }

    call_commands
}

fn parse_draw_commands(reflection: &ProgramReflection) -> Vec<DrawCommand> {
    let mut draw_commands: Vec<DrawCommand> = vec![];
    for EntrypointReflection {
        name,
        user_attributes,
    } in &reflection.entry_points
    {
        for attribute in user_attributes {
            let Some(playground_attribute_name) = attribute.name.strip_prefix("playground_") else {
                continue;
            };
            if playground_attribute_name == "DRAW" {
                let [
                    UserAttributeParameter::Int(vertex_count),
                    UserAttributeParameter::String(fragment_entrypoint),
                ] = &attribute.parameters[..]
                else {
                    panic!(
                        "Invalid attribute parameter type for {playground_attribute_name} attribute on {name}"
                    )
                };

                draw_commands.push(DrawCommand {
                    vertex_count: *vertex_count as u32,
                    vertex_entrypoint: name.clone(),
                    fragment_entrypoint: fragment_entrypoint.to_string(),
                });
            }
        }
    }

    draw_commands
}

fn get_uniform_size(shader_reflection: &ProgramReflection) -> u64 {
    let mut size = 0;

    for VariableReflection {
        reflection_type, ..
    } in &shader_reflection.variables
    {
        let BoundParameter::Uniform {
            uniform_offset,
            resource_result,
        } = reflection_type
        else {
            continue;
        };
        size = size.max(*uniform_offset as u64 + resource_result.get_size() as u64)
    }

    round_up_to_nearest(size, 16)
}
