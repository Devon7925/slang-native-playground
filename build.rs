use std::{fs::{self, File}, io::Write};

use regex::Regex;
use slang::{
    reflection::Shader, Downcast, EntryPoint, GlobalSession, ParameterCategory, ResourceShape,
    ScalarType, TypeKind,
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

struct CompiledEntryPoint {
    module: slang::Module,
    entry_point: slang::EntryPoint,
}

impl SlangCompiler {
    pub fn new() -> Self {
        let global_slang_session = slang::GlobalSession::new().unwrap();
        // self.compile_target_map = slang::get_compile_targets();

        SlangCompiler {
            global_slang_session,
        }
    }

    // In our playground, we only allow to run shaders with two entry points: render_main and print_main
    fn find_runnable_entry_point(&self, module: &slang::Module) -> Option<EntryPoint> {
        for shader_type in ShaderType::iter() {
            let entry_point_name = shader_type.get_entry_point_name();
            if let Some(entry_point) = module.find_entry_point_by_name(entry_point_name) {
                return Some(entry_point);
            }
        }

        return None;
    }

    fn find_entry_point(
        &self,
        module: &slang::Module,
        entry_point_name: Option<&String>,
    ) -> Option<EntryPoint> {
        if entry_point_name.clone().map(|ep| ep == "").unwrap_or(true) {
            let entry_point = self.find_runnable_entry_point(module);
            if entry_point.is_none() {
                // self.diagnostics_msg += "Warning: The current shader code is not runnable because 'image_main' or 'print_main' functions are not found.\n";
                // self.diagnostics_msg += "Use the 'Compile' button to compile it to different targets.\n";
                // TODO
            }
            return entry_point;
        } else {
            let entry_point = module.find_entry_point_by_name(entry_point_name.unwrap().as_str());
            if entry_point.is_none() {
                // let error = slang::get_last_error();
                // panic!(error.type + " error: " + error.message);
                // self.diagnostics_msg += (error.type + " error: " + error.message);
                return None; // TODO
            }
            return entry_point;
        }
    }

    // If user code defines image_main or print_main, we will know the entry point name because they're
    // already defined in our pre-built module. So we will add those one of those entry points to the
    // dropdown list. Then, we will find whether user code also defines other entry points, if it has
    // we will also add them to the dropdown list.
    fn find_defined_entry_points(&self) -> Vec<String> {
        let mut result: Vec<String> = vec![];

        let search_path = std::ffi::CString::new("shaders").unwrap();

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
            return vec![];
        };
        let module = slang_session.load_module("user.slang").unwrap();

        let count = module.entry_point_count();
        for i in 0..count {
            let entry_point = module.entry_point_by_index(i).unwrap();
            result.push(entry_point.function_reflection().name().to_string());
        }

        let program = slang_session
            .create_composite_component_type(&[module.downcast().clone()])
            .unwrap();
        let linked_program = program.link().unwrap();
        let shader_reflection = linked_program.layout(0).unwrap();

        for st in ShaderType::iter().map(|st| st.get_entry_point_name().to_string()) {
            if shader_reflection
                .find_function_by_name(st.as_str())
                .is_some()
            {
                result.push(st);
            }
        }

        return result;
    }

    fn is_runnable_entry_point(entry_point_name: &String) -> bool {
        return ShaderType::iter().any(|st| st.get_entry_point_name() == entry_point_name);
    }

    // Since we will not let user to change the entry point code, we can precompile the entry point module
    // and reuse it for every compilation.

    fn compile_entry_point_module(
        &self,
        slang_session: &slang::Session,
        module_name: &String,
    ) -> Result<CompiledEntryPoint, slang::Error> {
        let module = slang_session.load_module(&module_name)?;

        // we use the same entry point name as module name
        let Some(entry_point) = self.find_entry_point(&module, Some(module_name)) else {
            panic!("Could not find entry point {}", module_name);
        };

        return Ok(CompiledEntryPoint {
            module,
            entry_point,
        });
    }

    fn get_precompiled_program(
        &self,
        slang_session: &slang::Session,
        module_name: &String,
    ) -> Option<CompiledEntryPoint> {
        if !SlangCompiler::is_runnable_entry_point(&module_name) {
            return None;
        }

        let main_module = self.compile_entry_point_module(slang_session, module_name);

        return Some(main_module.unwrap());
    }

    fn add_active_entry_points(
        &self,
        slang_session: &slang::Session,
        entry_point_name: &Option<String>,
        user_module: slang::Module,
        component_list: &mut Vec<slang::ComponentType>,
    ) -> bool {
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
                // self.diagnostics_msg += "error: Entry point name ${name} is reserved";
                // TODO
                return false;
            }
        }

        // If entry point is provided, we know for sure this is not a whole program compilation,
        // so we will just go to find the correct module to include in the compilation.
        if let Some(entry_point_name) = entry_point_name {
            if SlangCompiler::is_runnable_entry_point(&entry_point_name) {
                // we use the same entry point name as module name
                let Some(main_program) =
                    self.get_precompiled_program(&slang_session, entry_point_name)
                else {
                    return false;
                };

                component_list.push(main_program.module.downcast().clone());
                component_list.push(main_program.entry_point.downcast().clone());
            } else {
                // we know the entry point is from user module
                let Some(entry_point) = self.find_entry_point(&user_module, Some(entry_point_name))
                else {
                    return false;
                };

                component_list.push(entry_point.downcast().clone());
            }
        }
        // otherwise, it's a whole program compilation, we will find all active entry points in the user code
        // and pre-built modules.
        else {
            let results = self.find_defined_entry_points();
            for result in results {
                if SlangCompiler::is_runnable_entry_point(&result) {
                    let Some(main_program) = self.get_precompiled_program(&slang_session, &result)
                    else {
                        return false;
                    };
                    component_list.push(main_program.module.downcast().clone());
                    component_list.push(main_program.entry_point.downcast().clone());
                } else {
                    let Some(entry_point) = self.find_entry_point(&user_module, Some(&result))
                    else {
                        return false;
                    };

                    component_list.push(entry_point.downcast().clone());
                }
            }
        }
        return true;
    }

    fn get_binding_descriptor(
        &self,
        index: u32,
        program_reflection: &Shader,
        parameter: &slang::reflection::VariableLayout,
    ) -> Option<wgpu::BindingType> {
        let global_layout = program_reflection.global_params_type_layout();

        let binding_type = global_layout.descriptor_set_descriptor_range_type(0, index as i64);

        // Special case.. TODO: Remove this as soon as the reflection API properly reports write-only textures.
        if parameter.variable().name().unwrap() == "outputTexture" {
            return Some(wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            });
        }

        match binding_type {
            slang::BindingType::Texture => Some(wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
            }),
            slang::BindingType::MutableTeture => Some(wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            }),
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
            _ => None,
        }
    }

    fn get_resource_bindings(
        &self,
        linked_program: &slang::ComponentType,
    ) -> HashMap<String, wgpu::BindGroupLayoutEntry> {
        let reflection = linked_program.layout(0).unwrap(); // assume target-index = 0

        let mut resource_descriptors = HashMap::new();
        for parameter in reflection.parameters() {
            let name = parameter.variable().name().unwrap().to_string();
            if parameter.category() == ParameterCategory::Uniform { continue }


            let resource_info =
                self.get_binding_descriptor(parameter.binding_index(), reflection, parameter);
            let binding = wgpu::BindGroupLayoutEntry {
                ty: resource_info.unwrap(),
                binding: parameter.binding_index(),
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
            };

            resource_descriptors.insert(name, binding);
        }
        resource_descriptors.insert("uniformInput".to_string(), wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
        });

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
    ) -> Vec<ResourceCommand> {
        let mut commands: Vec<ResourceCommand> = vec![];

        for parameter in shader_reflection.parameters() {
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

                    let bi = parameter.binding_index();
                    let format = shader_reflection
                        .global_params_type_layout()
                        .element_type_layout()
                        .binding_range_image_format(bi as i64 - 1);

                    Some(ResourceCommandData::BLACK {
                        width: width as u32,
                        height: height as u32,
                        format: get_wgpu_format_from_slang_format(format, parameter.ty().resource_result_type()),
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

                    let bi = parameter.binding_index();
                    let format = shader_reflection
                        .global_params_type_layout()
                        .element_type_layout()
                        .binding_range_image_format(bi as i64 - 1);

                    Some(ResourceCommandData::URL {
                        url: attribute.argument_value_string(0).unwrap().to_string(),
                        format: get_wgpu_format_from_slang_format(format, parameter.ty().resource_result_type()),
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
                } else {
                    None
                };

                if let Some(command) = command {
                    commands.push(ResourceCommand {
                        resource_name: parameter.variable().name().unwrap().to_string(),
                        command_data: command,
                    });
                }
            }
        }

        return commands;
    }

    pub fn compile(
        &self,
        search_path: &str,
        entry_point_name: Option<String>,
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
        let user_source = fs::read_to_string(user_module.file_path()).unwrap();
        self.add_active_entry_points(
            &slang_session,
            &entry_point_name,
            user_module,
            &mut components,
        );
        let program = slang_session
            .create_composite_component_type(components.as_slice())
            .unwrap();
        let linked_program = program.link().unwrap();

        let bindings = self.get_resource_bindings(&linked_program);

        let shader_reflection = linked_program.layout(0).unwrap();
        let hashed_strings = load_strings(shader_reflection);
        let out_code = linked_program.target_code(0).unwrap().as_slice().to_vec();
        let out_code = String::from_utf8(out_code).unwrap();

        let mut entry_group_sizes = HashMap::new();
        for entry in shader_reflection.entry_points() {
            let group_size = entry.compute_thread_group_size();
            //convert to string
            entry_group_sizes.insert(entry.name().to_string(), group_size);
        }

        let resource_commands = self.resource_commands_from_attributes(shader_reflection);
        let call_commands = parse_call_commands(user_source, shader_reflection);

        return CompilationResult {
            out_code,
            entry_group_sizes,
            bindings,
            uniform_controllers: get_uniform_sliders(&resource_commands),
            resource_commands,
            call_commands,
            hashed_strings,
            uniform_size: get_uniform_size(shader_reflection),
        };
    }
}

fn round_up_to_nearest(size: u64, arg: u64) -> u64 {
    (size + arg - 1) / arg * arg
}

fn get_wgpu_format_from_slang_format(format: slang::ImageFormat, resource_type: &slang::reflection::Type) -> wgpu::TextureFormat {
    use slang::ImageFormat;
    use wgpu::TextureFormat;
    match format {
        ImageFormat::SLANGIMAGEFORMATUnknown => {
            match resource_type.kind() {
                TypeKind::Vector => todo!(),
                TypeKind::Scalar => {
                    match resource_type.scalar_type() {
                        ScalarType::Int32 => TextureFormat::R32Sint,
                        ScalarType::Uint32 => TextureFormat::R32Uint,
                        ScalarType::Float32 => TextureFormat::R32Float,
                        _ => panic!("Invalid resource type"),
                    }
                },
                _ => panic!("Invalid resource type"),
            }
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
        .map(|s| (slang::reflection::get_string_hash(s.as_str()), s))
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

fn parse_call_commands(user_source: String, reflection: &Shader) -> Vec<CallCommand> {
    // Look for commands of the form:
    //
    // 1. //! CALL(fn-name, SIZE_OF(<resource-name>)) ==> Dispatch a compute pass with the given
    //                                                    function name and using the resource size
    //                                                    to determine the work-group size.
    // 2. //! CALL(fn-name, 512, 512) ==> Dispatch a compute pass with the given function name and
    //                                    the provided work-group size.
    //

    let mut call_commands: Vec<CallCommand> = vec![];
    let lines = user_source.split('\n');
    let call_regex = Regex::new(r"\/\/!\s+CALL\((\w+),\s*(.*)\)").unwrap();
    for line in lines {
        let Some(call_matches) = call_regex.captures(line) else {
            continue;
        };
        let fn_name = call_matches.get(1).unwrap().as_str();
        let args: Vec<&str> = call_matches
            .get(2)
            .unwrap()
            .as_str()
            .split(',')
            .map(|arg| arg.trim())
            .collect();

        if let Some(resource_name) = args[0]
            .strip_prefix("SIZE_OF(")
            .and_then(|rest| rest.strip_suffix(")"))
        {
            let Some(resource_reflection) = reflection
                .parameters()
                .find(|param| param.variable().name().unwrap() == resource_name)
            else {
                panic!(
                    "Cannot find resource {} for {} CALL command",
                    resource_name, fn_name
                )
            };
            let mut element_size: Option<u32> = None;
            if resource_reflection.ty().kind() == TypeKind::Resource
                && resource_reflection.ty().resource_shape() == ResourceShape::SlangStructuredBuffer
            {
                element_size = Some(get_size(resource_reflection.ty().resource_result_type()));
            }
            call_commands.push(CallCommand {
                function: fn_name.to_string(),
                parameters: CallCommandParameters::ResourceBased(
                    resource_name.to_string(),
                    element_size,
                ),
            });
        } else {
            call_commands.push(CallCommand {
                function: fn_name.to_string(),
                parameters: CallCommandParameters::FixedSize(
                    args.iter().map(|arg| arg.parse::<u32>().unwrap()).collect(),
                ),
            });
        }
    }

    return call_commands;
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

fn get_uniform_sliders(resource_commands: &Vec<ResourceCommand>) -> Vec<UniformController> {
    let mut controllers: Vec<UniformController> = vec![];
    for resource_command in resource_commands.iter() {
        match resource_command.command_data {
            ResourceCommandData::SLIDER {
                default,
                min,
                max,
                offset,
                ..
            } => controllers.push(UniformController::SLIDER {
                name: resource_command.resource_name.clone(),
                value: default,
                min,
                max,
                buffer_offset: offset,
            }),
            ResourceCommandData::COLORPICK {
                default, offset, ..
            } => {
                controllers.push(UniformController::COLORPICK {
                    name: resource_command.resource_name.clone(),
                    value: default,
                    buffer_offset: offset,
                });
            }
            _ => {}
        }
    }
    return controllers;
}


fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo::rerun-if-changed=shaders");

    let compiler = SlangCompiler::new();

    fs::create_dir_all("compiled_shaders").unwrap();
    
    let compilation = compiler.compile("shaders", None, "user.slang");
    let serialized = ron::ser::to_string_pretty(&compilation, ron::ser::PrettyConfig::default()).unwrap();
    let mut file = File::create("compiled_shaders/compiled.ron").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
    
    let rand_float_compilation = compiler.compile("demos", Some("computeMain".to_string()), "rand_float.slang");
    let serialized = ron::ser::to_string_pretty(&rand_float_compilation, ron::ser::PrettyConfig::default()).unwrap();
    let mut file = File::create("compiled_shaders/rand_float_compiled.ron").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}