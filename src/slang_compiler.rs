use std::{collections::HashMap, fs};

use regex::Regex;
use slang::{
    reflection::Shader, Downcast, EntryPoint, GlobalSession, ResourceShape, ScalarType, TypeKind,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use wgpu::BindGroupLayoutEntry;

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

pub enum ResourceCommandData {
    ZEROS { count: u32, element_size: u32 },
    RAND(u32),
    BLACK(u32, u32),
    URL(String),
}

pub struct ResourceCommand {
    pub resource_name: String,
    pub command_data: ResourceCommandData,
}

pub struct CompilationResult {
    pub out_code: HashMap<String, (String, [u64; 3])>,
    pub bindings: HashMap<String, BindGroupLayoutEntry>,
    pub resource_commands: Vec<ResourceCommand>,
    pub call_commands: Vec<CallCommand>,
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
        let Some(entry_point) = self.find_entry_point(
            &module,
            Some(module_name),
        ) else {
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
                let Some(entry_point) = self.find_entry_point(
                    &user_module,
                    Some(entry_point_name),
                ) else {
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
                    return true;
                } else {
                    let Some(entry_point) = self.find_entry_point(
                        &user_module,
                        Some(&result),
                    ) else {
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

        let count = reflection.parameter_count();

        let mut resource_descriptors = HashMap::new();
        for i in 0..count {
            let parameter = reflection.parameter_by_index(i).unwrap();
            let name = parameter.variable().name().unwrap().to_string();

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
                            "ZEROS attribute cannot be applied to {}, it only supports buffers",
                            parameter.semantic_name().unwrap()
                        )
                    }
                    let count = attribute.argument_value_int(0).unwrap();
                    if count < 0 {
                        panic!(
                            "ZEROS count for {} cannot have negative size",
                            parameter.semantic_name().unwrap()
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
                            "RAND attribute cannot be applied to {}, it only supports buffers",
                            parameter.semantic_name().unwrap()
                        )
                    }
                    if parameter.ty().resource_result_type().kind() != TypeKind::Scalar
                        || parameter.ty().resource_result_type().scalar_type()
                            != ScalarType::Float32
                    {
                        panic!("RAND attribute cannot be applied to {}, it only supports float buffers", parameter.semantic_name().unwrap())
                    }
                    let count = attribute.argument_value_int(0).unwrap();
                    if count < 0 {
                        panic!(
                            "RAND count for {} cannot have negative size",
                            parameter.semantic_name().unwrap()
                        )
                    }
                    Some(ResourceCommandData::RAND(count as u32))
                } else if playground_attribute_name == "BLACK" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "BLACK attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.semantic_name().unwrap()
                        )
                    }

                    let width = attribute.argument_value_int(0).unwrap();
                    let height = attribute.argument_value_int(1).unwrap();
                    if width < 0 {
                        panic!(
                            "BLACK width for {} cannot have negative size",
                            parameter.semantic_name().unwrap()
                        )
                    }
                    if height < 0 {
                        panic!(
                            "BLACK height for {} cannot have negative size",
                            parameter.semantic_name().unwrap()
                        )
                    }

                    Some(ResourceCommandData::BLACK(width as u32, height as u32))
                } else if playground_attribute_name == "URL" {
                    if parameter.ty().kind() != TypeKind::Resource
                        || parameter.ty().resource_shape() != ResourceShape::SlangTexture2d
                    {
                        panic!(
                            "URL attribute cannot be applied to {}, it only supports 2D textures",
                            parameter.semantic_name().unwrap()
                        )
                    }
                    Some(ResourceCommandData::URL(
                        attribute.argument_value_string(0).unwrap().to_string(),
                    ))
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
        // let hashed_strings = linked_program.load_strings(); TODO

        let bindings = self.get_resource_bindings(&linked_program);

        let shader_reflection = linked_program.layout(0).unwrap();

        let mut out_code = HashMap::new();
        for (i, entry) in shader_reflection.entry_points().enumerate() {
            let entry_out_code = linked_program
                .entry_point_code(
                    i as i64, /* entry_point_index */
                    0,        /* target_index */
                )
                .unwrap()
                .as_slice()
                .to_vec();
            let group_size = entry.compute_thread_group_size();
            //convert to string
            let entry_out_code = String::from_utf8(entry_out_code).unwrap();
            out_code.insert(entry.name().to_string(), (entry_out_code, group_size));
        }

        let resource_commands = self.resource_commands_from_attributes(shader_reflection);
        let call_commands = parse_call_commands(user_source, shader_reflection);

        return CompilationResult {
            out_code,
            bindings,
            resource_commands,
            call_commands,
        };
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
        _ => panic!("Unimplemented type for get_size"),
    }
}

pub enum CallCommandParameters {
    ResourceBased(String, Option<u32>),
    FixedSize(Vec<u32>),
}

pub struct CallCommand {
    pub function: String,
    pub parameters: CallCommandParameters,
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
