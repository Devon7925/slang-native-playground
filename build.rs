use std::{
    fs::{self, File},
    io::Write,
};

use slang::{
    Downcast, GlobalSession, ParameterCategory, ResourceAccess, ResourceShape, ScalarType, Stage,
    TypeKind,
    reflection::{Shader, VariableLayout},
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

include!("src/slang_compiler.rs");

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
    // compile_target_map: { name: string, value: number }[] | null,
}

impl Default for SlangCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl SlangCompiler {
    pub fn new() -> Self {
        let global_slang_session = slang::GlobalSession::new().unwrap();
        // self.compile_target_map = slang::get_compile_targets();

        SlangCompiler {
            global_slang_session,
        }
    }

    fn add_components(
        &self,
        user_module: slang::Module,
        slang_session: &slang::Session,
        component_list: &mut Vec<slang::ComponentType>,
    ) {
        for imported_file in user_module.dependency_file_paths() {
            let module = slang_session.load_module(imported_file).unwrap();

            component_list.push(module.downcast().clone());

            for entry_point in module.entry_points() {
                component_list.push(entry_point.downcast().clone());
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
                    .unwrap();

                component_list.push(module.downcast().clone());

                for entry_point in module.entry_points() {
                    component_list.push(entry_point.downcast().clone());
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
        program_reflection: &Shader,
        parameter: &slang::reflection::VariableLayout,
    ) -> Option<wgpu::BindingType> {
        let global_layout = program_reflection.global_params_type_layout();

        let binding_type = global_layout.descriptor_set_descriptor_range_type(0, index as i64);

        match binding_type {
            slang::BindingType::Texture => Some(wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: parameter_texture_view_dimension(parameter),
            }),
            slang::BindingType::MutableTeture => {
                let format = global_layout
                    .element_type_layout()
                    .binding_range_image_format(index as i64 - 1);
                Some(wgpu::BindingType::StorageTexture {
                    access: match parameter.ty().resource_access() {
                        slang::ResourceAccess::Read => wgpu::StorageTextureAccess::ReadOnly,
                        slang::ResourceAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
                        slang::ResourceAccess::Write => wgpu::StorageTextureAccess::WriteOnly,
                        _ => panic!("Invalid resource access"),
                    },
                    format: get_wgpu_format_from_slang_format(
                        format,
                        parameter.ty().resource_result_type(),
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
        linked_program: &slang::ComponentType,
        resource_commands: &HashMap<String, ResourceCommandData>,
    ) -> HashMap<String, wgpu::BindGroupLayoutEntry> {
        let reflection = linked_program.layout(0).unwrap(); // assume target-index = 0

        let mut resource_descriptors = HashMap::new();
        let mut uniform_input = false;
        for parameter in reflection.parameters() {
            let name = parameter.variable().name().unwrap().to_string();
            if parameter.category() == ParameterCategory::Uniform {
                uniform_input = true;
                continue;
            }

            let resource_info =
                self.get_binding_descriptor(parameter.binding_index(), reflection, parameter);
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
        shader_reflection: &Shader,
    ) -> HashMap<String, ResourceCommandData> {
        let mut commands: HashMap<String, ResourceCommandData> = HashMap::new();

        for (parameter_idx, parameter) in shader_reflection
            .global_params_type_layout()
            .element_type_layout()
            .fields()
            .enumerate()
        {
            let offset = shader_reflection
                .global_params_type_layout()
                .element_type_layout()
                .field_binding_range_offset(parameter_idx as i64);
            for attribute in parameter.variable().user_attributes() {
                let Some(playground_attribute_name) = attribute.name().strip_prefix("playground_")
                else {
                    continue;
                };
                let command = if playground_attribute_name == "ZEROS" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangStructuredBuffer
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports buffers",
                            parameter.variable().name().unwrap()
                        )
                    }
                    let count = attribute.argument_value_int(0).unwrap();
                    if count < 0 {
                        panic!(
                            "{playground_attribute_name} count for {} cannot have negative size",
                            parameter.variable().name().unwrap()
                        )
                    }
                    Some(ResourceCommandData::Zeros {
                        count: count as u32,
                        element_size: get_size(parameter.ty().resource_result_type()),
                    })
                } else if playground_attribute_name == "SAMPLER" {
                    if parameter.ty().kind() != TypeKind::SamplerState {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports samplers",
                            parameter.variable().name().unwrap()
                        )
                    }
                    Some(ResourceCommandData::Sampler)
                } else if playground_attribute_name == "RAND" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangStructuredBuffer
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports buffers",
                            parameter.variable().name().unwrap()
                        )
                    }
                    if parameter.ty().resource_result_type().kind() != TypeKind::Scalar
                        || parameter.ty().resource_result_type().scalar_type()
                            != ScalarType::Float32
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float buffers",
                            parameter.variable().name().unwrap()
                        )
                    }
                    let count = attribute.argument_value_int(0).unwrap();
                    if count < 0 {
                        panic!(
                            "{playground_attribute_name} count for {} cannot have negative size",
                            parameter.variable().name().unwrap()
                        )
                    }
                    Some(ResourceCommandData::Rand(count as u32))
                } else if playground_attribute_name == "BLACK" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.variable().name().unwrap()
                        )
                    }

                    let width = attribute.argument_value_int(0).unwrap();
                    let height = attribute.argument_value_int(1).unwrap();
                    if width < 0 {
                        panic!(
                            "{playground_attribute_name} width for {} cannot have negative size",
                            parameter.variable().name().unwrap()
                        )
                    }
                    if height < 0 {
                        panic!(
                            "{playground_attribute_name} height for {} cannot have negative size",
                            parameter.variable().name().unwrap()
                        )
                    }

                    let format = shader_reflection
                        .global_params_type_layout()
                        .element_type_layout()
                        .binding_range_image_format(offset);

                    Some(ResourceCommandData::Black {
                        width: width as u32,
                        height: height as u32,
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "BLACK_3D" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangTexture3d
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports 3D textures",
                            parameter.variable().name().unwrap()
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
                                    parameter.variable().name().unwrap()
                                )
                            }
                        };
                    }

                    check_positive!(size_x);
                    check_positive!(size_y);
                    check_positive!(size_z);

                    let format = shader_reflection
                        .global_params_type_layout()
                        .element_type_layout()
                        .binding_range_image_format(offset);

                    Some(ResourceCommandData::Black3D {
                        size_x: size_x as u32,
                        size_y: size_y as u32,
                        size_z: size_z as u32,
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "BLACK_SCREEN" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.variable().name().unwrap()
                        )
                    }

                    let width_scale = attribute.argument_value_float(0).unwrap();
                    let height_scale = attribute.argument_value_float(1).unwrap();
                    if width_scale < 0.0 {
                        panic!(
                            "{playground_attribute_name} width for {} cannot have negative size",
                            parameter.variable().name().unwrap()
                        )
                    }
                    if height_scale < 0.0 {
                        panic!(
                            "{playground_attribute_name} height for {} cannot have negative size",
                            parameter.variable().name().unwrap()
                        )
                    }

                    let format = shader_reflection
                        .global_params_type_layout()
                        .element_type_layout()
                        .binding_range_image_format(offset);

                    Some(ResourceCommandData::BlackScreen {
                        width_scale,
                        height_scale,
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "URL" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "URL attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.variable().name().unwrap()
                        )
                    }

                    let format = shader_reflection
                        .global_params_type_layout()
                        .element_type_layout()
                        .binding_range_image_format(offset);

                    Some(ResourceCommandData::Url {
                        url: attribute
                            .argument_value_string(0)
                            .unwrap()
                            .trim_matches('"')
                            .to_string(),
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "MODEL" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangStructuredBuffer
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports buffers",
                            parameter.variable().name().unwrap()
                        )
                    }
                    if parameter.ty().element_type().kind() != TypeKind::Struct {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, inner type must be struct",
                            parameter.variable().name().unwrap()
                        )
                    }
                    let mut field_types = vec![];
                    for field in parameter.ty().element_type().fields() {
                        match field.name().unwrap() {
                            "position" => {
                                if field.ty().kind() != TypeKind::Vector
                                    || field.ty().element_count() != 3
                                    || field.ty().element_type().kind() != TypeKind::Scalar
                                    || field.ty().element_type().scalar_type()
                                        != ScalarType::Float32
                                {
                                    panic!(
                                        "Unsupported type for position field of MODEL struct for {}",
                                        parameter.variable().name().unwrap()
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
                                        parameter.variable().name().unwrap()
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
                                        parameter.variable().name().unwrap()
                                    )
                                }
                                field_types.push(ModelField::TexCoords)
                            }
                            field_name => panic!(
                                "{field_name} is not a valid field for MODEL attribute on {}, valid fields are: position, normal, uv",
                                parameter.variable().name().unwrap()
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
                    if parameter.ty().kind() != TypeKind::Resource
                        || !(parameter.ty().resource_shape() == ResourceShape::SlangTexture2d
                            || parameter.ty().resource_shape()
                                == ResourceShape::SlangStructuredBuffer)
                    {
                        panic!(
                            "REBIND_FOR_DRAW attribute cannot be applied to {}, it only supports 2D textures and structured buffers",
                            parameter.variable().name().unwrap()
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
                    if parameter.ty().kind() != TypeKind::Scalar
                        || parameter.ty().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().name().unwrap()
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
                    if parameter.ty().kind() != TypeKind::Vector
                        || parameter.ty().element_count() <= 2
                        || parameter.ty().element_type().kind() != TypeKind::Scalar
                        || parameter.ty().element_type().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float vectors",
                            parameter.variable().name().unwrap()
                        )
                    }

                    Some(ResourceCommandData::ColorPick {
                        default: [
                            attribute.argument_value_float(0).unwrap(),
                            attribute.argument_value_float(1).unwrap(),
                            attribute.argument_value_float(2).unwrap(),
                        ],
                        element_size: parameter.type_layout().size(ParameterCategory::Uniform)
                            / parameter.ty().element_count(),
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "MOUSEPOSITION" {
                    if parameter.ty().kind() != TypeKind::Vector
                        || parameter.ty().element_count() <= 3
                        || parameter.ty().element_type().kind() != TypeKind::Scalar
                        || parameter.ty().element_type().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float vectors",
                            parameter.variable().name().unwrap()
                        )
                    }

                    Some(ResourceCommandData::MousePosition {
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "KEY_INPUT" {
                    if parameter.ty().kind() != TypeKind::Scalar
                        || parameter.ty().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().name().unwrap()
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
                    if parameter.ty().kind() != TypeKind::Scalar
                        || parameter.ty().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().name().unwrap()
                        )
                    }

                    Some(ResourceCommandData::Time {
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else if playground_attribute_name == "DELTA_TIME" {
                    if parameter.ty().kind() != TypeKind::Scalar
                        || parameter.ty().scalar_type() != ScalarType::Float32
                        || parameter.category_by_index(0) != ParameterCategory::Uniform
                    {
                        panic!(
                            "{playground_attribute_name} attribute cannot be applied to {}, it only supports float uniforms",
                            parameter.variable().name().unwrap()
                        )
                    }

                    Some(ResourceCommandData::DeltaTime {
                        offset: parameter.offset(ParameterCategory::Uniform),
                    })
                } else {
                    None
                };

                if let Some(command) = command {
                    commands.insert(parameter.variable().name().unwrap().to_string(), command);
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

        // All compiler options are available through this builder.
        let session_options = slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::High)
            .matrix_layout_row(true);

        let target_desc = slang::TargetDesc::default()
            .format(slang::CompileTarget::Wgsl)
            .profile(self.global_slang_session.find_profile("spirv_1_6"));

        let targets = [target_desc];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_paths)
            .options(&session_options);

        let Some(slang_session) = self.global_slang_session.create_session(&session_desc) else {
            // let error = self.slang_wasm_module.get_last_error();
            // console.error(error.type + " error: " + error.message);
            // self.diagnostics_msg += (error.type + " error: " + error.message);
            // TODO
            panic!();
        };

        let mut components: Vec<slang::ComponentType> = vec![];

        let user_module = slang_session.load_module(entry_module_name).unwrap();

        // For now, we just don't allow user to define image_main or print_main as entry point name for simplicity
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

        self.add_components(user_module, &slang_session, &mut components);

        let program = slang_session
            .create_composite_component_type(components.as_slice())
            .unwrap();
        let linked_program = program.link().unwrap();

        let shader_reflection = linked_program.layout(0).unwrap();
        let hashed_strings = load_strings(shader_reflection);
        let out_code = linked_program.target_code(0).unwrap().as_slice().to_vec();
        let out_code = String::from_utf8(out_code).unwrap();

        let mut entry_group_sizes = HashMap::new();
        for entry in shader_reflection.entry_points() {
            let group_size = entry.compute_thread_group_size();
            //convert to string
            if entry.stage() == Stage::Compute {
                entry_group_sizes.insert(entry.name().to_string(), group_size);
            }
        }

        let resource_commands = self.resource_commands_from_attributes(shader_reflection);
        let call_commands = parse_call_commands(shader_reflection);
        let draw_commands = parse_draw_commands(shader_reflection);

        let bindings = self.get_resource_bindings(&linked_program, &resource_commands);

        let uniform_controllers = get_uniform_sliders(&resource_commands);

        CompilationResult {
            out_code,
            entry_group_sizes,
            bindings,
            uniform_controllers,
            resource_commands,
            call_commands,
            draw_commands,
            hashed_strings,
            uniform_size: get_uniform_size(shader_reflection),
        }
    }
}

fn parameter_texture_view_dimension(parameter: &VariableLayout) -> wgpu::TextureViewDimension {
    match parameter.ty().resource_shape() {
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

enum ModelField {
    Position,
    Normal,
    TexCoords,
}

fn is_available_in_compute(resource_command: &ResourceCommandData) -> bool {
    !matches!(resource_command, ResourceCommandData::RebindForDraw { .. })
}
fn is_available_in_graphics(parameter: &VariableLayout) -> bool {
    match parameter.ty().kind() {
        TypeKind::Resource => matches!(
            (
                parameter.ty().resource_shape(),
                parameter.ty().resource_access(),
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

fn load_strings(shader_reflection: &Shader) -> HashMap<u32, String> {
    (0..shader_reflection.hashed_string_count())
        .map(|i| shader_reflection.hashed_string(i).unwrap().to_string())
        .map(|s| (slang::reflection::compute_string_hash(s.as_str()), s))
        .collect()
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

fn parse_call_commands(reflection: &Shader) -> Vec<CallCommand> {
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
                let resource_reflection = reflection
                    .parameters()
                    .find(|param| param.variable().name().unwrap() == resource_name)
                    .unwrap();

                let mut element_size: Option<u32> = None;
                if resource_reflection.ty().kind() == TypeKind::Resource
                    && resource_reflection.ty().resource_shape()
                        == ResourceShape::SlangStructuredBuffer
                {
                    element_size = Some(get_size(resource_reflection.ty().resource_result_type()));
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
                    .find(|param| param.variable().name().unwrap() == resource_name)
                    .unwrap();

                if resource_reflection.ty().kind() != TypeKind::Resource
                    && resource_reflection.ty().resource_shape()
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

fn parse_draw_commands(reflection: &Shader) -> Vec<DrawCommand> {
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

fn get_uniform_size(shader_reflection: &Shader) -> u64 {
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

fn get_uniform_sliders(
    resource_commands: &HashMap<String, ResourceCommandData>,
) -> Vec<UniformController> {
    let mut controllers: Vec<UniformController> = vec![];
    for (resource_name, command_data) in resource_commands.iter() {
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
                controller: UniformControllerType::Slider {
                    value: *default,
                    min: *min,
                    max: *max,
                },
            }),
            ResourceCommandData::ColorPick {
                default, offset, ..
            } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::ColorPick { value: *default },
            }),
            ResourceCommandData::MousePosition { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::MousePosition,
            }),
            ResourceCommandData::Time { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::Time,
            }),
            ResourceCommandData::DeltaTime { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::DeltaTime,
            }),
            ResourceCommandData::KeyInput { key, offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::KeyInput { key: key.clone() },
            }),
            _ => {}
        }
    }

    controllers
}

fn get_uniform_update_code(uniform_controllers: &[UniformController]) -> String {
    let mut uniform_update_code = "{\n\
        let uniform_borrow = self.uniform_components.borrow_mut();\n\
    "
    .to_string();
    for (idx, controller) in uniform_controllers.iter().enumerate() {
        uniform_update_code += format!(
            "{{
    let UniformController {{
        buffer_offset,
        controller: _controller,
        ..
    }} = uniform_borrow.get({idx}).unwrap();"
        )
        .as_str();
        match controller.controller {
            UniformControllerType::Slider { .. } => {
                uniform_update_code += "
    let UniformControllerType::Slider { value, .. } = _controller else {
        panic!(\"Invalid generated code: Expected Slider got {:?}\", _controller);
    };
    let slice = [*value];";
            }
            UniformControllerType::ColorPick { .. } => {
                uniform_update_code += "
    let UniformControllerType::ColorPick {{ value, .. }} = _controller else {{
        panic!(\"Invalid generated code: Expected ColorPick got {:?}\", _controller);
    }};
    let slice = [*value];";
            }
            UniformControllerType::MousePosition => {
                uniform_update_code += "
    let mut slice = [0.0f32; 4];
    slice[0] = self.mouse_state.last_mouse_down_pos.x as f32;
    slice[1] = self.mouse_state.last_mouse_down_pos.y as f32;
    slice[2] = self.mouse_state.last_mouse_clicked_pos.x as f32;
    slice[3] = self.mouse_state.last_mouse_clicked_pos.y as f32;
    if self.mouse_state.is_mouse_down {
        slice[2] = -slice[2];
    }
    if self.mouse_state.mouse_clicked {
        slice[3] = -slice[3];
    }";
            }
            UniformControllerType::Time => {
                uniform_update_code += "
    let value = std::time::Instant::now()
        .duration_since(self.launch_time)
        .as_secs_f32();
    let slice = [value];";
            }
            UniformControllerType::DeltaTime => {
                uniform_update_code += "
    let slice = [self.delta_time];";
            }
            UniformControllerType::KeyInput { ref key } => {
                let keycode = match key.to_lowercase().as_str() {
                    "enter" => "Key::Named(NamedKey::Enter)".to_string(),
                    "space" => "Key::Named(NamedKey::Space)".to_string(),
                    "shift" => "Key::Named(NamedKey::Shift)".to_string(),
                    "ctrl" => "Key::Named(NamedKey::Control)".to_string(),
                    "escape" => "Key::Named(NamedKey::Escape)".to_string(),
                    "backspace" => "Key::Named(NamedKey::Backspace)".to_string(),
                    "tab" => "Key::Named(NamedKey::Tab)".to_string(),
                    "arrowup" => "Key::Named(NamedKey::ArrowUp)".to_string(),
                    "arrowdown" => "Key::Named(NamedKey::ArrowDown)".to_string(),
                    "arrowleft" => "Key::Named(NamedKey::ArrowLeft)".to_string(),
                    "arrowright" => "Key::Named(NamedKey::ArrowRight)".to_string(),
                    k => format!("Key::Character(\"{}\".into())", k),
                };
                uniform_update_code += format!(
                    "
    let value = if self.keyboard_state.pressed_keys.contains(&{keycode}) {{
        1.0f32
    }} else {{
        0.0f32
    }};
    let slice = [value];"
                )
                .as_str();
            }
        }
        uniform_update_code += "
    let uniform_data = bytemuck::cast_slice(&slice);
    buffer_data[*buffer_offset..(buffer_offset + uniform_data.len())].copy_from_slice(uniform_data);
}\n";
    }
    uniform_update_code += "\n}";

    uniform_update_code
}

fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo::rerun-if-changed=shaders");
    println!("cargo::rerun-if-changed=build.rs");

    let compiler = SlangCompiler::new();

    fs::create_dir_all("compiled_shaders").unwrap();

    let compilation = compiler.compile(vec!["shaders", "src/shaders"], "user.slang");
    let serialized =
        ron::ser::to_string_pretty(&compilation, ron::ser::PrettyConfig::default()).unwrap();
    let mut file = File::create("compiled_shaders/compiled.ron").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();

    let uniform_update_code = get_uniform_update_code(&compilation.uniform_controllers);
    let mut file = File::create("compiled_shaders/uniform_update_code.rs").unwrap();
    file.write_all(uniform_update_code.as_bytes()).unwrap();

    let rand_float_compilation = compiler.compile(vec!["demos"], "rand_float.slang");
    let serialized =
        ron::ser::to_string_pretty(&rand_float_compilation, ron::ser::PrettyConfig::default())
            .unwrap();
    let mut file = File::create("compiled_shaders/rand_float_compiled.ron").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}
