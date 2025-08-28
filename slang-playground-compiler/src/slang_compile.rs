use std::collections::HashMap;

use slang_reflector::{
    BoundParameter, BoundResource, Downcast, EntrypointReflection, GlobalSession,
    ProgramLayoutReflector, ProgramReflection, ResourceAccess, ScalarType, TextureType,
    UserAttributeParameter, VariableReflection, VariableReflectionType,
};

use crate::{
    CallCommand, CallCommandParameters, CompilationResult, DrawCommand, ResourceCommandData,
    UniformController, UniformControllerType, resource_commands::*, uniform_controllers::*,
};

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
    resource_command_constructors: HashMap<
        String,
        Box<
            dyn Fn(&BoundResource, &[UserAttributeParameter], &str) -> Box<dyn ResourceCommandData>,
        >,
    >,
}

impl Default for SlangCompiler {
    fn default() -> Self {
        let mut default_controllers = HashMap::new();

        UniformSlider::register(&mut default_controllers);
        UniformColorPick::register(&mut default_controllers);
        UniformTime::register(&mut default_controllers);
        UniformFrameId::register(&mut default_controllers);
        UniformScreenSize::register(&mut default_controllers);
        UniformMousePosition::register(&mut default_controllers);
        UniformDeltaTime::register(&mut default_controllers);
        UniformKeyInput::register(&mut default_controllers);
        UniformExternal::register(&mut default_controllers);

        let mut default_resource_commands = HashMap::new();

        ExternalResourceCommand::register(&mut default_resource_commands);
        ZerosResourceCommand::register(&mut default_resource_commands);
        RandResourceCommand::register(&mut default_resource_commands);
        BlackResourceCommand::register(&mut default_resource_commands);
        Black3DResourceCommand::register(&mut default_resource_commands);
        BlackScreenResourceCommand::register(&mut default_resource_commands);
        UrlResourceCommand::register(&mut default_resource_commands);
        ModelResourceCommand::register(&mut default_resource_commands);
        SamplerResourceCommand::register(&mut default_resource_commands);
        RebindForDrawResourceCommand::register(&mut default_resource_commands);

        Self::new(default_controllers, default_resource_commands)
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
        resource_command_constructors: HashMap<
            String,
            Box<
                dyn Fn(
                    &BoundResource,
                    &[UserAttributeParameter],
                    &str,
                ) -> Box<dyn ResourceCommandData>,
            >,
        >,
    ) -> Self {
        let global_slang_session = slang_reflector::GlobalSession::new().unwrap();
        SlangCompiler {
            global_slang_session,
            uniform_controller_constructors,
            resource_command_constructors,
        }
    }

    fn add_components(
        &self,
        slang_session: &slang_reflector::Session,
        used_files: impl Iterator<Item = String>,
        component_list: &mut Vec<slang_reflector::ComponentType>,
    ) {
        for imported_file in used_files {
            let re = regex::Regex::new(":[0-9A-F]+$").unwrap();
            let file = re.replace(&imported_file, "").to_string();

            let module = slang_session
                .load_module(&file)
                .unwrap_or_else(|e| panic!("Failed to load module {}: {:?}", file, e.to_string()));

            component_list.push(module.downcast().clone());

            for entry_point in module.entry_points() {
                component_list.push(entry_point.downcast().clone());
            }
        }
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
        resource_commands: &HashMap<String, Box<dyn ResourceCommandData>>,
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
                .map(|command| command.is_available_in_compute())
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
    ) -> HashMap<String, Box<dyn ResourceCommandData>> {
        let mut commands: HashMap<String, Box<dyn ResourceCommandData>> = HashMap::new();

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

                if let Some(constructor) = self
                    .resource_command_constructors
                    .get(playground_attribute_name)
                {
                    let command = constructor(resource, &attribute.parameters, name);
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

                if let Some(constructor) = self
                    .uniform_controller_constructors
                    .get(playground_attribute_name)
                {
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

        let targets = [target_desc];

        let session_desc = slang_reflector::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_paths)
            .options(&session_options);

        let Some(slang_session) = self.global_slang_session.create_session(&session_desc) else {
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

        self.add_components(
            &slang_session,
            user_module.dependency_file_paths().map(|p| p.to_string()),
            &mut components,
        );

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
                entry_group_sizes.insert(entry.name().unwrap().to_string(), group_size);
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

pub fn get_wgpu_format_from_slang_format(
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
