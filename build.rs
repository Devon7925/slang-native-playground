use std::{
    fs::{self, File},
    io::Write,
};

use slang::{
    reflection::{Shader, VariableLayout}, Downcast, GlobalSession, ParameterCategory, ResourceAccess, ResourceShape, ScalarType, Stage, TypeKind
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

impl SlangCompiler {
    pub fn new() -> Self {
        let global_slang_session = slang::GlobalSession::new().unwrap();
        // self.compile_target_map = slang::get_compile_targets();

        SlangCompiler {
            global_slang_session,
        }
    }

    fn add_components(&self, user_module: slang::Module, slang_session: &slang::Session, component_list: &mut Vec<slang::ComponentType>) {
        let count = user_module.entry_point_count();
        for i in 0..count {
            let entry_point = user_module.entry_point_by_index(i).unwrap();
            component_list.push(entry_point.downcast().clone());
        }

        let program = slang_session
            .create_composite_component_type(&[user_module.downcast().clone()])
            .unwrap();
        let linked_program = program.link().unwrap();
        let shader_reflection = linked_program.layout(0).unwrap();

        for st in ShaderType::iter().map(|st| st.get_entry_point_name().to_string()) {
            if shader_reflection
                .find_function_by_name(st.as_str())
                .is_some()
            {
                let module = slang_session.load_module(format!("{}.slang", st).as_str()).unwrap();

                component_list.push(module.downcast().clone());
        
                let count = module.entry_point_count();
                for i in 0..count {
                    let entry_point = module.entry_point_by_index(i).unwrap();
                    component_list.push(entry_point.downcast().clone());
                }
            }
        }
    }

    fn is_runnable_entry_point(entry_point_name: &String) -> bool {
        return ShaderType::iter().any(|st| st.get_entry_point_name() == entry_point_name);
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
                view_dimension: wgpu::TextureViewDimension::D2,
            }),
            slang::BindingType::MutableTeture => {
                let format = global_layout.element_type_layout().binding_range_image_format(index as i64 - 1);
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
                    view_dimension: wgpu::TextureViewDimension::D2,
                })
            },
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
            slang::BindingType::Sampler => {
                Some(
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering)
                )
            }
            a => {
                println!("cargo::warning=Could not generate binding for {:?}", a);
                None
            },
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
            println!("cargo::warning={}, bi={}, tk={:?}", name, parameter.binding_index(), parameter.ty().kind());
            if resource_commands.get(&name).map(|c| is_available_in_compute(c)).unwrap_or(true) {
                visibility |= wgpu::ShaderStages::COMPUTE;
                println!("cargo::warning=made computable {}", name);
            }
            if is_available_in_graphics(parameter) {
                visibility |= wgpu::ShaderStages::VERTEX_FRAGMENT;
                println!("cargo::warning=made visible {}", name);
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

        return resource_descriptors;
    }

    // fn load_module(&self, slang_session: Session, module_name: string, source: string, component_type_list: Module[]) {
    //     let module: Module | null = slang_session.load_module_from_source(source, module_name, "/" + module_name + ".slang");
    //     if (!module) {
    //         let error = self.slang_wasm_module.get_last_error();
    //         console.error(error.type + " error: " + error.message);
    //         self.diagnostics_msg += (error.type + " error: " + error.message);
    //         return false;
    //     }
    //     component_type_list.push(module);
    //     return true;
    // }

    fn resource_commands_from_attributes(
        &self,
        shader_reflection: &Shader,
    ) -> HashMap<String, ResourceCommandData> {
        let mut commands: HashMap<String, ResourceCommandData> = HashMap::new();

        for (parameter_idx, parameter) in shader_reflection
            .global_params_type_layout()
            .element_type_layout()
            .fields().enumerate() {
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
                    Some(ResourceCommandData::ZEROS {
                        count: count as u32,
                        element_size: get_size(parameter.ty().resource_result_type()),
                    })
                } else if playground_attribute_name == "SAMPLER" {
                    if parameter.ty().kind() != TypeKind::SamplerState
                    {
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
                        panic!("{playground_attribute_name} attribute cannot be applied to {}, it only supports float buffers", parameter.variable().name().unwrap())
                    }
                    let count = attribute.argument_value_int(0).unwrap();
                    if count < 0 {
                        panic!(
                            "{playground_attribute_name} count for {} cannot have negative size",
                            parameter.variable().name().unwrap()
                        )
                    }
                    Some(ResourceCommandData::RAND(count as u32))
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

                    Some(ResourceCommandData::BLACK {
                        width: width as u32,
                        height: height as u32,
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

                    Some(ResourceCommandData::URL {
                        url: attribute.argument_value_string(0).unwrap().trim_matches('"').to_string(),
                        format: get_wgpu_format_from_slang_format(
                            format,
                            parameter.ty().resource_result_type(),
                        ),
                    })
                } else if playground_attribute_name == "REBIND_FOR_DRAW" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "REBIND_FOR_DRAW attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.variable().name().unwrap()
                        )
                    }

                    Some(ResourceCommandData::RebindForDraw {
                        original_texture: attribute.argument_value_string(0).unwrap().trim_matches('"').to_string(),
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

                    Some(ResourceCommandData::SLIDER {
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

                    Some(ResourceCommandData::COLORPICK {
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

                    Some(ResourceCommandData::MOUSEPOSITION {
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

                    Some(ResourceCommandData::TIME {
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

        return commands;
    }

    pub fn compile(
        &self,
        search_path: &str,
        entry_module_name: &str,
    ) -> CompilationResult {
        let search_path = std::ffi::CString::new(search_path).unwrap();

        // All compiler options are available through this builder.
        let session_options = slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::High)
            .matrix_layout_row(true);

        let target_desc = slang::TargetDesc::default()
            .format(slang::CompileTarget::Wgsl)
            .profile(self.global_slang_session.find_profile("spirv_1_6"));

        let targets = [target_desc];
        let search_paths = [search_path.as_ptr()];

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

        return CompilationResult {
            out_code,
            entry_group_sizes,
            bindings,
            uniform_controllers: get_uniform_sliders(&resource_commands),
            resource_commands,
            call_commands,
            draw_commands,
            hashed_strings,
            uniform_size: get_uniform_size(shader_reflection),
        };
    }
}

fn is_available_in_compute(resource_command: &ResourceCommandData) -> bool {
    match resource_command {
        ResourceCommandData::RebindForDraw { .. } => false,
        _ => true
    }
}
fn is_available_in_graphics(parameter: &VariableLayout) -> bool {
    if parameter.ty().kind() == TypeKind::Resource {
        if parameter.ty().resource_shape() == ResourceShape::SlangTexture2d && parameter.ty().resource_access() != ResourceAccess::Write {
            return true
        }
    } else if parameter.ty().kind() == TypeKind::SamplerState {
        return true
    } else if parameter.ty().kind() == TypeKind::ConstantBuffer {
        return true
    }
    false
}

fn round_up_to_nearest(size: u64, arg: u64) -> u64 {
    (size + arg - 1) / arg * arg
}

fn get_wgpu_format_from_slang_format(
    format: slang::ImageFormat,
    resource_type: &slang::reflection::Type,
) -> wgpu::TextureFormat {
    use slang::ImageFormat;
    use wgpu::TextureFormat;
    match format {
        ImageFormat::SLANGIMAGEFORMATUnknown => match resource_type.kind() {
            TypeKind::Vector => match (resource_type.element_type().scalar_type(), resource_type.element_count()) {
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

                let resource_name = attribute.argument_value_string(0).unwrap().trim_matches('"');
                let resource_reflection = reflection
                    .parameters()
                    .find(|param| param.variable().name().unwrap() == resource_name)
                    .unwrap();

                let mut element_size: Option<u32> = None;
                if resource_reflection.ty().kind() == TypeKind::Resource
                    && resource_reflection.ty().resource_shape() == ResourceShape::SlangStructuredBuffer
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
                    parameters: CallCommandParameters::FixedSize(
                        args
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

    return call_commands;
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
                let fragment_entrypoint = attribute.argument_value_string(1).unwrap().trim_matches('"');

                draw_commands.push(DrawCommand {
                    vertex_count: vertex_count as u32,
                    vertex_entrypoint: fn_name.to_string(),
                    fragment_entrypoint: fragment_entrypoint.to_string(),
                });
            }
        }
    }

    return draw_commands;
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

    return round_up_to_nearest(size, 16);
}

fn get_uniform_sliders(resource_commands: &HashMap<String, ResourceCommandData>) -> Vec<UniformController> {
    let mut controllers: Vec<UniformController> = vec![];
    for (resource_name, command_data) in resource_commands.iter() {
        match command_data {
            ResourceCommandData::SLIDER {
                default,
                min,
                max,
                offset,
                ..
            } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::SLIDER {
                    value: *default,
                    min: *min,
                    max: *max,
                },
            }),
            ResourceCommandData::COLORPICK {
                default, offset, ..
            } => {
                controllers.push(UniformController {
                    name: resource_name.clone(),
                    buffer_offset: *offset,
                    controller: UniformControllerType::COLORPICK { value: *default },
                });
            }
            ResourceCommandData::MOUSEPOSITION { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::MOUSEPOSITION,
            }),
            ResourceCommandData::TIME { offset } => controllers.push(UniformController {
                name: resource_name.clone(),
                buffer_offset: *offset,
                controller: UniformControllerType::TIME,
            }),
            _ => {}
        }
    }
    return controllers;
}

fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo::rerun-if-changed=shaders");
    println!("cargo::rerun-if-changed=build.rs");

    let compiler = SlangCompiler::new();

    fs::create_dir_all("compiled_shaders").unwrap();

    let compilation = compiler.compile("shaders", "user.slang");
    let serialized =
        ron::ser::to_string_pretty(&compilation, ron::ser::PrettyConfig::default()).unwrap();
    let mut file = File::create("compiled_shaders/compiled.ron").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();

    let rand_float_compilation =
        compiler.compile("demos", "rand_float.slang");
    let serialized =
        ron::ser::to_string_pretty(&rand_float_compilation, ron::ser::PrettyConfig::default())
            .unwrap();
    let mut file = File::create("compiled_shaders/rand_float_compiled.ron").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}
