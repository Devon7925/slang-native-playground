use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ops::Deref,
    rc::Rc,
};

use slang::{
    GlobalSession, ParameterCategory, ProgramLayout, ResourceAccess, ResourceShape, ScalarType,
    Stage, TypeKind,
    reflection::{TypeLayout, VariableLayout},
};
use slang_compiler_type_definitions::{
    CallCommand, CallCommandParameters, CompilationResult, DrawCommand, ResourceCommandData,
    UniformColorPick, UniformController, UniformDeltaTime, UniformKeyInput,
    UniformMousePosition, UniformSlider, UniformTime,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use url::Url;

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
}

impl Default for SlangCompiler {
    fn default() -> Self {
        Self::new()
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

impl slang::FileSystem for CustomFileSystem {
    fn load_file(&self, path: &str) -> slang::Result<slang::Blob> {
        let mut path = path.to_string();

        // Remove automatically added path prefix for github imports
        let re = &regex::Regex::new(r"^.*/github://").unwrap();
        path = re.replace_all(path.as_str(), "github://").to_string();

        if let Some(git_path) = path.strip_prefix("github://") {
            // first 2 parts of path are the user and repo
            // Use git api to get files ex. "https://api.github.com/repos/shader-slang/slang-playground/contents/example.slang"
            let parts: Vec<&str> = git_path.split('/').collect();
            if parts.len() < 3 {
                return Err(slang::Error::Blob(slang::Blob::from("Invalid github path")));
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
                .map_err(|e| slang::Error::Blob(slang::Blob::from(e.to_string())))?;

            let mut request = client.get(&url);

            // try to get token from GITHUB_TOKEN file in repo root if possible
            let token = std::fs::read_to_string("GITHUB_TOKEN");

            if let Ok(token) = token {
                request = request.header("Authorization", format!("token {}", token));
            };

            let response = request
                .header("Accept", "application/vnd.github.v3.raw")
                .send()
                .map_err(|e| slang::Error::Blob(slang::Blob::from(e.to_string())))?;

            if !response.status().is_success() {
                if response.status() == 403 {
                    println!(
                        "cargo::warning=Loading file {} failed. Possibly rate limited.",
                        path.clone()
                    );
                }

                return Err(slang::Error::Blob(slang::Blob::from(format!(
                    "Failed to get file from github: {}",
                    response.status()
                ))));
            }

            let response = response
                .text()
                .map_err(|e| slang::Error::Blob(slang::Blob::from(e.to_string())))?;
            self.used_files.borrow_mut().insert(path.clone());
            return Ok(slang::Blob::from(response.into_bytes()));
        } else {
            match std::fs::read(&path) {
                Ok(bytes) => {
                    self.used_files.borrow_mut().insert(path);
                    Ok(slang::Blob::from(bytes))
                }
                Err(e) => Err(slang::Error::Blob(slang::Blob::from(format!(
                    "Failed to read file: {}",
                    e
                )))),
            }
        }
    }
}

impl SlangCompiler {
    pub fn new() -> Self {
        let global_slang_session = slang::GlobalSession::new().unwrap();
        SlangCompiler {
            global_slang_session,
        }
    }

    fn add_components(
        &self,
        slang_session: &slang::Session,
        used_files: impl IntoIterator<Item = String>,
        component_list: &mut Vec<slang::ComponentType>,
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
                        panic!(
                            "Failed to load module {}: {:?}",
                            st,
                            e.to_string()
                        )
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

    fn get_binding_descriptor(
        &self,
        index: u32,
        reflection: &TypeLayout,
        program_reflection: &TypeLayout,
        parameter: &slang::reflection::VariableLayout,
    ) -> Option<wgpu::BindingType> {
        let binding_type = program_reflection
            .descriptor_set_descriptor_range_type(0, parameter.binding_index() as i64);

        match binding_type {
            slang::BindingType::Texture => Some(wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: parameter_texture_view_dimension(parameter),
            }),
            slang::BindingType::MutableTeture => {
                let format = reflection.binding_range_image_format(index as i64);
                Some(wgpu::BindingType::StorageTexture {
                    access: match parameter.ty().unwrap().resource_access() {
                        slang::ResourceAccess::Read => wgpu::StorageTextureAccess::ReadOnly,
                        slang::ResourceAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
                        slang::ResourceAccess::Write => wgpu::StorageTextureAccess::WriteOnly,
                        _ => panic!("Invalid resource access"),
                    },
                    format: get_wgpu_format_from_slang_format(
                        format,
                        parameter.ty().unwrap().resource_result_type(),
                    ),
                    view_dimension: parameter_texture_view_dimension(parameter),
                })
            }
            slang::BindingType::ConstantBuffer => Some(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            slang::BindingType::MutableTypedBuffer | slang::BindingType::MutableRawBuffer => {
                Some(wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                })
            }
            slang::BindingType::RawBuffer => Some(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            slang::BindingType::Sampler => Some(wgpu::BindingType::Sampler(
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
        reflection: &TypeLayout,
        top_level_reflection: &TypeLayout,
        resource_commands: &HashMap<String, ResourceCommandData>,
    ) -> HashMap<String, wgpu::BindGroupLayoutEntry> {
        if matches!(reflection.kind(), TypeKind::ConstantBuffer) {
            return self.get_resource_bindings(
                reflection.element_type_layout(),
                top_level_reflection,
                resource_commands,
            );
        }

        let mut resource_descriptors = HashMap::new();
        let mut uniform_input = false;
        for (parameter_idx, parameter) in reflection.fields().enumerate() {
            let name = parameter.variable().unwrap().name().to_string();
            if parameter.category() == ParameterCategory::Uniform {
                uniform_input = true;
                continue;
            }

            let offset = reflection.field_binding_range_offset(parameter_idx as i64);

            let resource_info = self.get_binding_descriptor(
                offset as u32,
                &reflection,
                &top_level_reflection,
                parameter,
            );
            let mut visibility = wgpu::ShaderStages::NONE;
            if resource_commands
                .get(&name)
                .map(is_available_in_compute)
                .unwrap_or(true)
            {
                visibility |= wgpu::ShaderStages::COMPUTE;
            }
            if is_available_in_graphics(parameter) {
                visibility |= wgpu::ShaderStages::VERTEX_FRAGMENT;
            }
            let binding = wgpu::BindGroupLayoutEntry {
                ty: resource_info.unwrap(),
                binding: parameter.binding_index(),
                visibility,
                count: None,
            };

            resource_descriptors.insert(name, binding);
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
        reflection: &TypeLayout,
        top_level_reflection: &TypeLayout,
    ) -> HashMap<String, ResourceCommandData> {
        if matches!(reflection.kind(), TypeKind::ConstantBuffer) {
            return self.resource_commands_from_attributes(
                reflection.element_type_layout(),
                top_level_reflection,
            );
        }

        let mut commands: HashMap<String, ResourceCommandData> = HashMap::new();

        for (parameter_idx, parameter) in reflection.fields().enumerate() {
            let offset = reflection.field_binding_range_offset(parameter_idx as i64);
            for attribute in parameter.variable().unwrap().user_attributes() {
                let Some(playground_attribute_name) = attribute.name().strip_prefix("playground_")
                else {
                    continue;
                };
                let command = if playground_attribute_name == "ZEROS" {
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || parameter.ty().unwrap().resource_shape()
                            != ResourceShape::SlangStructuredBuffer
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports buffers",
                            parameter.variable().unwrap().name()
                        )
                    }
                    let count = attribute.argument_value_int(0).unwrap();
                    if count < 0 {
                        panic!(
                            "{playground_attribute_name} count for {} cannot have negative size",
                            parameter.variable().unwrap().name()
                        )
                    }
                    Some(ResourceCommandData::Zeros {
                        count: count as u32,
                        element_size: get_layout_size(
                            parameter.type_layout().element_type_layout(),
                        ),
                    })
                } else if playground_attribute_name == "SAMPLER" {
                    if parameter.ty().unwrap().kind() != TypeKind::SamplerState {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports samplers",
                            parameter.variable().unwrap().name()
                        )
                    }
                    Some(ResourceCommandData::Sampler)
                } else if playground_attribute_name == "RAND" {
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || parameter.ty().unwrap().resource_shape()
                            != ResourceShape::SlangStructuredBuffer
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports buffers",
                            parameter.variable().unwrap().name()
                        )
                    }
                    if parameter.ty().unwrap().resource_result_type().kind() != TypeKind::Scalar
                        || parameter.ty().unwrap().resource_result_type().scalar_type()
                            != ScalarType::Float32
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float buffers",
                            parameter.variable().unwrap().name()
                        )
                    }
                    let count = attribute.argument_value_int(0).unwrap();
                    if count < 0 {
                        panic!(
                            "{playground_attribute_name} count for {} cannot have negative size",
                            parameter.variable().unwrap().name()
                        )
                    }
                    Some(ResourceCommandData::Rand(count as u32))
                } else if playground_attribute_name == "BLACK" {
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || parameter.ty().unwrap().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.variable().unwrap().name()
                        )
                    }

                    let width = attribute.argument_value_int(0).unwrap();
                    let height = attribute.argument_value_int(1).unwrap();
                    if width < 0 {
                        panic!(
                            "{playground_attribute_name} width for {} cannot have negative size",
                            parameter.variable().unwrap().name()
                        )
                    }
                    if height < 0 {
                        panic!(
                            "{playground_attribute_name} height for {} cannot have negative size",
                            parameter.variable().unwrap().name()
                        )
                    }

                    let format = reflection.binding_range_image_format(offset);

                    Some(ResourceCommandData::Black {
                        width: width as u32,
                        height: height as u32,
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().unwrap().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "BLACK_3D" {
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || parameter.ty().unwrap().resource_shape() != ResourceShape::SlangTexture3d
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports 3D textures",
                            parameter.variable().unwrap().name()
                        )
                    }

                    let size_x = attribute.argument_value_int(0).unwrap();
                    let size_y = attribute.argument_value_int(1).unwrap();
                    let size_z = attribute.argument_value_int(2).unwrap();
                    macro_rules! check_positive {
                        ($id:ident) => {
                            if $id < 0 {
                                panic!(
                                    "{playground_attribute_name} {} for {} cannot have negative size",
                                    stringify!($id),
                                    parameter.variable().unwrap().name()
                                )
                            }
                        };
                    }

                    check_positive!(size_x);
                    check_positive!(size_y);
                    check_positive!(size_z);

                    let format = reflection.binding_range_image_format(offset);

                    Some(ResourceCommandData::Black3D {
                        size_x: size_x as u32,
                        size_y: size_y as u32,
                        size_z: size_z as u32,
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().unwrap().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "BLACK_SCREEN" {
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || parameter.ty().unwrap().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.variable().unwrap().name()
                        )
                    }

                    let width_scale = attribute.argument_value_float(0).unwrap();
                    let height_scale = attribute.argument_value_float(1).unwrap();
                    if width_scale < 0.0 {
                        panic!(
                            "{playground_attribute_name} width for {} cannot have negative size",
                            parameter.variable().unwrap().name()
                        )
                    }
                    if height_scale < 0.0 {
                        panic!(
                            "{playground_attribute_name} height for {} cannot have negative size",
                            parameter.variable().unwrap().name()
                        )
                    }

                    let format = reflection.binding_range_image_format(offset);

                    Some(ResourceCommandData::BlackScreen {
                        width_scale,
                        height_scale,
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().unwrap().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "URL" {
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || parameter.ty().unwrap().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "URL attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.variable().unwrap().name()
                        )
                    }

                    let format = reflection.binding_range_image_format(offset);

                    let format = get_wgpu_format_from_slang_format(
                        format,
                        parameter.ty().unwrap().resource_result_type(),
                    );

                    let url = attribute
                        .argument_value_string(0)
                        .unwrap()
                        .trim_matches('"')
                        .to_string();

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
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || parameter.ty().unwrap().resource_shape()
                            != ResourceShape::SlangStructuredBuffer
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports buffers",
                            parameter.variable().unwrap().name()
                        )
                    }
                    if parameter.ty().unwrap().element_type().kind() != TypeKind::Struct {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, inner type must be struct",
                            parameter.variable().unwrap().name()
                        )
                    }
                    let mut field_types = vec![];
                    for field in parameter.ty().unwrap().element_type().fields() {
                        match field.name() {
                            "position" => {
                                if field.ty().kind() != TypeKind::Vector
                                    || field.ty().element_count() != 3
                                    || field.ty().element_type().kind() != TypeKind::Scalar
                                    || field.ty().element_type().scalar_type()
                                        != ScalarType::Float32
                                {
                                    panic!(
                                        "Unsupported type for position field of MODEL struct for {}",
                                        parameter.variable().unwrap().name()
                                    )
                                }
                                field_types.push(ModelField::Position)
                            }
                            "normal" => {
                                if field.ty().kind() != TypeKind::Vector
                                    || field.ty().element_count() != 3
                                    || field.ty().element_type().kind() != TypeKind::Scalar
                                    || field.ty().element_type().scalar_type()
                                        != ScalarType::Float32
                                {
                                    panic!(
                                        "Unsupported type for normal field of MODEL struct for {}",
                                        parameter.variable().unwrap().name()
                                    )
                                }
                                field_types.push(ModelField::Normal)
                            }
                            "uv" => {
                                if field.ty().kind() != TypeKind::Vector
                                    || field.ty().element_count() != 3
                                    || field.ty().element_type().kind() != TypeKind::Scalar
                                    || field.ty().element_type().scalar_type()
                                        != ScalarType::Float32
                                {
                                    panic!(
                                        "Unsupported type for normal field of MODEL struct for {}",
                                        parameter.variable().unwrap().name()
                                    )
                                }
                                field_types.push(ModelField::TexCoords)
                            }
                            field_name => panic!(
                                "{field_name} is not a valid field for MODEL attribute on {}, valid fields are: position, normal, uv",
                                parameter.variable().unwrap().name()
                            ),
                        }
                    }

                    let path = attribute
                        .argument_value_string(0)
                        .unwrap()
                        .trim_matches('"')
                        .to_string();

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
                    if parameter.ty().unwrap().kind() != TypeKind::Resource
                        || !(parameter.ty().unwrap().resource_shape()
                            == ResourceShape::SlangTexture2d
                            || parameter.ty().unwrap().resource_shape()
                                == ResourceShape::SlangStructuredBuffer)
                    {
                        panic!(
                            "REBIND_FOR_DRAW attribute cannot be applied to {}, it only supports 2D textures and structured buffers",
                            parameter.variable().unwrap().name()
                        )
                    }

                    Some(ResourceCommandData::RebindForDraw {
                        original_resource: attribute
                            .argument_value_string(0)
                            .unwrap()
                            .trim_matches('"')
                            .to_string(),
                    })
                } else if playground_attribute_name == "SLIDER" {
                    if parameter.ty().unwrap().kind() != TypeKind::Scalar
                        || parameter.ty().unwrap().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().unwrap().name()
                        )
                    }

                    Some(ResourceCommandData::Slider {
                        default: attribute.argument_value_float(0).unwrap(),
                        min: attribute.argument_value_float(1).unwrap(),
                        max: attribute.argument_value_float(2).unwrap(),
                        element_size: parameter.type_layout().size(ParameterCategory::Uniform),
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "COLOR_PICK" {
                    if parameter.ty().unwrap().kind() != TypeKind::Vector
                        || parameter.ty().unwrap().element_count() <= 2
                        || parameter.ty().unwrap().element_type().kind() != TypeKind::Scalar
                        || parameter.ty().unwrap().element_type().scalar_type()
                            != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float vectors",
                            parameter.variable().unwrap().name()
                        )
                    }

                    Some(ResourceCommandData::ColorPick {
                        default: [
                            attribute.argument_value_float(0).unwrap(),
                            attribute.argument_value_float(1).unwrap(),
                            attribute.argument_value_float(2).unwrap(),
                        ],
                        element_size: parameter.type_layout().size(ParameterCategory::Uniform)
                            / parameter.ty().unwrap().element_count(),
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "MOUSE_POSITION" {
                    if parameter.ty().unwrap().kind() != TypeKind::Vector
                        || parameter.ty().unwrap().element_count() <= 3
                        || parameter.ty().unwrap().element_type().kind() != TypeKind::Scalar
                        || parameter.ty().unwrap().element_type().scalar_type()
                            != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float vectors",
                            parameter.variable().unwrap().name()
                        )
                    }

                    Some(ResourceCommandData::MousePosition {
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "KEY_INPUT" {
                    if parameter.ty().unwrap().kind() != TypeKind::Scalar
                        || parameter.ty().unwrap().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().unwrap().name()
                        )
                    }

                    Some(ResourceCommandData::KeyInput {
                        key: attribute
                            .argument_value_string(0)
                            .unwrap()
                            .trim_matches('"')
                            .to_string(),
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "TIME" {
                    if parameter.ty().unwrap().kind() != TypeKind::Scalar
                        || parameter.ty().unwrap().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().unwrap().name()
                        )
                    }

                    Some(ResourceCommandData::Time {
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "DELTA_TIME" {
                    if parameter.ty().unwrap().kind() != TypeKind::Scalar
                        || parameter.ty().unwrap().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().unwrap().name()
                        )
                    }

                    Some(ResourceCommandData::DeltaTime {
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else {
                    None
                };

                if let Some(command) = command {
                    commands.insert(parameter.variable().unwrap().name().to_string(), command);
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

        let session_options = slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::High)
            .matrix_layout_row(true);

        let target_desc = slang::TargetDesc::default()
            .format(slang::CompileTarget::Wgsl)
            .profile(self.global_slang_session.find_profile("spirv_1_6"));

        let custom_file_system = CustomFileSystem::new();

        let targets = [target_desc];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_paths)
            .options(&session_options)
            .file_system(custom_file_system.clone());

        let Ok(slang_session) = self.global_slang_session.create_session(&session_desc) else {
            panic!("Failed to create slang session");
        };

        let mut components: Vec<slang::ComponentType> = vec![];

        let user_module = match slang_session.load_module(entry_module_name) {
            Ok(module) => module,
            Err(e) => {
                panic!(
                    "Failed to load module {}: {:?}",
                    entry_module_name,
                    e.to_string()
                );
            }
        };

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
        let hashed_strings = load_strings(&shader_reflection);
        let out_code = linked_program
            .target_code(0)
            .unwrap_or_else(|err| panic!("Failed to compile shader: {:?}", err.to_string()))
            .as_slice()
            .to_vec();
        let out_code = String::from_utf8(out_code).unwrap();

        let mut entry_group_sizes = HashMap::new();
        for entry in shader_reflection.entry_points() {
            let group_size = entry.compute_thread_group_size();
            if entry.stage() == Stage::Compute {
                entry_group_sizes.insert(entry.name().to_string(), group_size);
            }
        }

        let global_layout = shader_reflection.global_params_type_layout();

        let resource_commands =
            self.resource_commands_from_attributes(&global_layout, &global_layout);
        let call_commands = parse_call_commands(&shader_reflection);
        let draw_commands = parse_draw_commands(&shader_reflection);

        let bindings =
            self.get_resource_bindings(&global_layout, &global_layout, &resource_commands);
        let uniform_controllers = get_uniform_controllers(&resource_commands);

        CompilationResult {
            out_code,
            entry_group_sizes,
            bindings,
            uniform_controllers,
            resource_commands,
            call_commands,
            draw_commands,
            hashed_strings,
            uniform_size: get_uniform_size(&shader_reflection),
        }
    }
}

fn parameter_texture_view_dimension(parameter: &VariableLayout) -> wgpu::TextureViewDimension {
    match parameter.ty().unwrap().resource_shape() {
        ResourceShape::SlangTexture1d => wgpu::TextureViewDimension::D1,
        ResourceShape::SlangTexture2d | ResourceShape::SlangTexture2dMultisample => {
            wgpu::TextureViewDimension::D2
        }
        ResourceShape::SlangTexture3d => wgpu::TextureViewDimension::D3,
        ResourceShape::SlangTextureCube => wgpu::TextureViewDimension::Cube,
        ResourceShape::SlangTexture1dArray => panic!("wgpu does not support 1d array textures"),
        ResourceShape::SlangTexture2dArray | ResourceShape::SlangTexture2dMultisampleArray => {
            wgpu::TextureViewDimension::D2Array
        }
        ResourceShape::SlangTextureCubeArray => panic!("wgpu does not support cube array textures"),
        _ => panic!("Could not process resource shape"),
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

fn is_available_in_graphics(parameter: &VariableLayout) -> bool {
    match parameter.ty().unwrap().kind() {
        TypeKind::Resource => matches!(
            (
                parameter.ty().unwrap().resource_shape(),
                parameter.ty().unwrap().resource_access(),
            ),
            (
                ResourceShape::SlangTexture2d | ResourceShape::SlangStructuredBuffer,
                ResourceAccess::Read,
            )
        ),
        TypeKind::SamplerState | TypeKind::ConstantBuffer => true,
        _ => false,
    }
}

fn round_up_to_nearest(size: u64, arg: u64) -> u64 {
    size.div_ceil(arg) * arg
}

fn get_wgpu_format_from_slang_format(
    format: slang::ImageFormat,
    resource_type: &slang::reflection::Type,
) -> wgpu::TextureFormat {
    use slang::ImageFormat;
    use wgpu::TextureFormat;
    match format {
        ImageFormat::SLANGIMAGEFORMATUnknown => match resource_type.kind() {
            TypeKind::Vector => match (
                resource_type.element_type().scalar_type(),
                resource_type.element_count(),
            ) {
                (ScalarType::Float32, 2) => TextureFormat::Rg32Float,
                (ScalarType::Float32, 3) => TextureFormat::Rgba32Float,
                (ScalarType::Float32, 4) => TextureFormat::Rgba32Float,
                _ => panic!("Invalid resource type"),
            },
            TypeKind::Scalar => match resource_type.scalar_type() {
                ScalarType::Int32 => TextureFormat::R32Sint,
                ScalarType::Uint32 => TextureFormat::R32Uint,
                ScalarType::Float32 => TextureFormat::R32Float,
                _ => panic!("Invalid resource type"),
            },
            _ => panic!("Invalid resource type"),
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

fn load_strings(shader_reflection: &ProgramLayout) -> HashMap<u32, String> {
    (0..shader_reflection.hashed_string_count())
        .map(|i| shader_reflection.hashed_string(i).unwrap().to_string())
        .map(|s| (slang::reflection::compute_string_hash(s.as_str()), s))
        .collect()
}

fn get_layout_size(resource_result_type: &slang::reflection::TypeLayout) -> u32 {
    match resource_result_type.kind() {
        TypeKind::Scalar => match resource_result_type.scalar_type().unwrap() {
            slang::ScalarType::Int8 | slang::ScalarType::Uint8 => 1,
            slang::ScalarType::Int16 | slang::ScalarType::Uint16 | slang::ScalarType::Float16 => 2,
            slang::ScalarType::Int32 | slang::ScalarType::Uint32 | slang::ScalarType::Float32 => 4,
            slang::ScalarType::Int64 | slang::ScalarType::Uint64 | slang::ScalarType::Float64 => 8,
            _ => panic!("Unimplemented scalar type"),
        },
        TypeKind::Vector => {
            let count = resource_result_type
                .element_count()
                .unwrap()
                .next_power_of_two() as u32;
            count * get_layout_size(resource_result_type.element_type_layout())
        }
        TypeKind::Struct => resource_result_type
            .fields()
            .map(|f| get_layout_size(f.type_layout()))
            .fold(0, |a, f| (a + f).div_ceil(f) * f),
        TypeKind::Array => {
            get_layout_size(resource_result_type.element_type_layout())
                * resource_result_type.element_count().unwrap() as u32
        }
        ty => panic!("Unimplemented type {ty:?} for get_size"),
    }
}

fn get_size(resource_result_type: &slang::reflection::Type) -> u32 {
    match resource_result_type.kind() {
        TypeKind::Scalar => match resource_result_type.scalar_type() {
            slang::ScalarType::Int8 | slang::ScalarType::Uint8 => 1,
            slang::ScalarType::Int16 | slang::ScalarType::Uint16 | slang::ScalarType::Float16 => 2,
            slang::ScalarType::Int32 | slang::ScalarType::Uint32 | slang::ScalarType::Float32 => 4,
            slang::ScalarType::Int64 | slang::ScalarType::Uint64 | slang::ScalarType::Float64 => 8,
            _ => panic!("Unimplemented scalar type"),
        },
        TypeKind::Vector => {
            let count = resource_result_type.element_count().next_power_of_two() as u32;
            count * get_size(resource_result_type.element_type())
        }
        TypeKind::Struct => resource_result_type
            .fields()
            .map(|f| get_size(f.ty()))
            .fold(0, |a, f| (a + f).div_ceil(f) * f),
        _ => panic!("Unimplemented type for get_size"),
    }
}

fn parse_call_commands(reflection: &ProgramLayout) -> Vec<CallCommand> {
    let mut call_commands: Vec<CallCommand> = vec![];
    for entry_point in reflection.entry_points() {
        let fn_name = entry_point.name();
        let mut call_command = None;
        let mut call_once = false;
        for attribute in entry_point.function().user_attributes() {
            let Some(playground_attribute_name) = attribute.name().strip_prefix("playground_")
            else {
                continue;
            };
            if playground_attribute_name == "CALL_SIZE_OF" {
                if call_command.is_some() {
                    panic!("Cannot have multiple CALL attributes for the same function");
                }

                let resource_name = attribute
                    .argument_value_string(0)
                    .unwrap()
                    .trim_matches('"');

                let mut resource_reflection = reflection.global_params_type_layout();

                if matches!(resource_reflection.kind(), TypeKind::ConstantBuffer) {
                    resource_reflection = resource_reflection.element_type_layout();
                }
                let resource_reflection = resource_reflection
                    .fields()
                    .find(|param| param.variable().unwrap().name() == resource_name)
                    .unwrap();

                let mut element_size: Option<u32> = None;
                if resource_reflection.ty().unwrap().kind() == TypeKind::Resource
                    && resource_reflection.ty().unwrap().resource_shape()
                        == ResourceShape::SlangStructuredBuffer
                {
                    element_size = Some(get_size(
                        resource_reflection.ty().unwrap().resource_result_type(),
                    ));
                }

                call_command = Some(CallCommand {
                    function: fn_name.to_string(),
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

                let args: Vec<u32> = (0..attribute.argument_count())
                    .map(|i| attribute.argument_value_int(i).unwrap() as u32)
                    .collect();
                call_command = Some(CallCommand {
                    function: fn_name.to_string(),
                    call_once: false,
                    parameters: CallCommandParameters::FixedSize(args),
                });
            } else if playground_attribute_name == "CALL_INDIRECT" {
                if call_command.is_some() {
                    panic!("Cannot have multiple CALL attributes for the same function");
                }

                let resource_name = attribute
                    .argument_value_string(0)
                    .unwrap()
                    .trim_matches('"')
                    .to_string();
                let resource_reflection = reflection
                    .parameters()
                    .find(|param| param.variable().unwrap().name() == resource_name)
                    .unwrap();

                if resource_reflection.ty().unwrap().kind() != TypeKind::Resource
                    && resource_reflection.ty().unwrap().resource_shape()
                        != ResourceShape::SlangStructuredBuffer
                {
                    panic!("Invalid type for CALL_INDIRECT buffer");
                }

                let offset = attribute.argument_value_int(1).unwrap() as u32;

                call_command = Some(CallCommand {
                    function: fn_name.to_string(),
                    call_once: false,
                    parameters: CallCommandParameters::Indirect(resource_name, offset),
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

fn parse_draw_commands(reflection: &ProgramLayout) -> Vec<DrawCommand> {
    let mut draw_commands: Vec<DrawCommand> = vec![];
    for entry_point in reflection.entry_points() {
        let fn_name = entry_point.name();
        for attribute in entry_point.function().user_attributes() {
            let Some(playground_attribute_name) = attribute.name().strip_prefix("playground_")
            else {
                continue;
            };
            if playground_attribute_name == "DRAW" {
                let vertex_count = attribute.argument_value_int(0).unwrap();
                let fragment_entrypoint = attribute
                    .argument_value_string(1)
                    .unwrap()
                    .trim_matches('"');

                draw_commands.push(DrawCommand {
                    vertex_count: vertex_count as u32,
                    vertex_entrypoint: fn_name.to_string(),
                    fragment_entrypoint: fragment_entrypoint.to_string(),
                });
            }
        }
    }

    draw_commands
}

fn get_uniform_size(shader_reflection: &ProgramLayout) -> u64 {
    let mut size = 0;

    for parameter in shader_reflection.parameters() {
        if parameter.category_by_index(0) != ParameterCategory::Uniform {
            continue;
        }
        size = size.max(
            parameter.offset(ParameterCategory::Uniform) as u64
                + parameter.type_layout().size(ParameterCategory::Uniform) as u64,
        )
    }

    round_up_to_nearest(size, 16)
}

fn get_uniform_controllers(
    resource_commands: &HashMap<String, ResourceCommandData>,
) -> Vec<UniformController> {
    let mut controllers: Vec<UniformController> = vec![];
    for (resource_name, command_data) in resource_commands {
        match command_data {
            ResourceCommandData::Slider {
                default,
                min,
                max,
                offset,
                ..
            } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: Box::new(UniformSlider {
                    value: *default,
                    min: *min,
                    max: *max,
                }),
            }),
            ResourceCommandData::ColorPick {
                default, offset, ..
            } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: Box::new(UniformColorPick { value: *default }),
            }),
            ResourceCommandData::MousePosition { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: Box::new(UniformMousePosition),
            }),
            ResourceCommandData::Time { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: Box::new(UniformTime),
            }),
            ResourceCommandData::DeltaTime { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: Box::new(UniformDeltaTime),
            }),
            ResourceCommandData::KeyInput { key, offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: Box::new(UniformKeyInput { key: key.clone() }),
            }),
            _ => {}
        }
    }
    controllers
}
