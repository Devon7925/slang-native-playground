use std::collections::HashMap;

use slang::{reflection::Shader, Downcast, EntryPoint, GlobalSession, SessionDesc};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use wgpu::BindGroupLayoutEntry;

fn is_whole_program_target(compile_target: String) -> bool {
    return compile_target == "METAL" || compile_target == "SPIRV";
}

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

pub struct CompilationResult {
    pub out_code: String,
    pub bindings: HashMap<String, BindGroupLayoutEntry>,
}

impl SlangCompiler {
    const SLANG_STAGE_VERTEX: u32 = 1;
    const SLANG_STAGE_FRAGMENT: u32 = 5;
    const SLANG_STAGE_COMPUTE: u32 = 6;

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

    fn find_entry_point(&self, module: &slang::Module, entry_point_name: Option<&String>, stage: u32) -> Option<EntryPoint> {
        if entry_point_name.clone().map(|ep| ep == "").unwrap_or(true) {
            let entry_point = self.find_runnable_entry_point(module);
            if entry_point.is_none() {
                // self.diagnostics_msg += "Warning: The current shader code is not runnable because 'image_main' or 'print_main' functions are not found.\n";
                // self.diagnostics_msg += "Use the 'Compile' button to compile it to different targets.\n";
                // TODO
            }
            return entry_point;
        }
        else {
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
        let module = slang_session.load_module("imageMain.slang").unwrap();

        let count = module.get_defined_entry_point_count();
        for i in 0..count {
            let entry_point = module.get_defined_entry_point(i).unwrap();
            result.push(entry_point.get_function_reflection().name().to_string());
        }

        return result;
    }

    fn is_runnable_entry_point(entry_point_name: &String) -> bool {
        return ShaderType::iter().any(|st| st.get_entry_point_name() == entry_point_name);
    }

    // Since we will not let user to change the entry point code, we can precompile the entry point module
    // and reuse it for every compilation.

    fn compile_entry_point_module(&self, slang_session: &slang::Session, module_name: &String) -> Result<CompiledEntryPoint, slang::Error> {
        let module = slang_session.load_module(&module_name)?;

        // we use the same entry point name as module name
        let Some(entry_point) = self.find_entry_point(&module, Some(module_name), SlangCompiler::SLANG_STAGE_COMPUTE) else {
            panic!(); //TODO
        };

        return Ok(CompiledEntryPoint { module, entry_point });

    }

    fn get_precompiled_program(&self, slang_session: &slang::Session, module_name: &String) -> Option<CompiledEntryPoint> {
        if !SlangCompiler::is_runnable_entry_point(&module_name) {
            return None;
        }

        let main_module = self.compile_entry_point_module(slang_session, module_name);

        return Some(main_module.unwrap());
    }

    fn add_active_entry_points(&self, slang_session: &slang::Session, entry_point_name: &String, user_module: slang::Module, component_list: &mut Vec<slang::ComponentType>) -> bool {
        // For now, we just don't allow user to define image_main or print_main as entry point name for simplicity
        let count = user_module.get_defined_entry_point_count();
        for i in 0..count {
            let name = user_module.get_defined_entry_point(i).unwrap().get_function_reflection().name().to_string();
            if SlangCompiler::is_runnable_entry_point(&name) {
                // self.diagnostics_msg += "error: Entry point name ${name} is reserved";
                // TODO
                return false;
            }
        }

        // If entry point is provided, we know for sure this is not a whole program compilation,
        // so we will just go to find the correct module to include in the compilation.
        if entry_point_name != "" {
            if SlangCompiler::is_runnable_entry_point(&entry_point_name) {
                // we use the same entry point name as module name
                let Some(main_program) = self.get_precompiled_program(&slang_session, entry_point_name) else {
                    return false;
                };

                component_list.push(main_program.module.downcast().clone());
                component_list.push(main_program.entry_point.downcast().clone());
            }
            else {
                // we know the entry point is from user module
                let Some(entry_point) = self.find_entry_point(&user_module, Some(entry_point_name), SlangCompiler::SLANG_STAGE_COMPUTE) else {
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
                    let Some(main_program) = self.get_precompiled_program(&slang_session, &result) else {
                        return false;
                    };
                    component_list.push(main_program.module.downcast().clone());
                    component_list.push(main_program.entry_point.downcast().clone());
                    return true;
                }
                else {
                    let Some(entry_point) = self.find_entry_point(&user_module, Some(&result), SlangCompiler::SLANG_STAGE_COMPUTE) else {
                        return false;
                    };

                    component_list.push(entry_point.downcast().clone());
                }
            }
        }
        return true;
    }

    fn get_binding_descriptor(&self, index: u32, program_reflection: &Shader, parameter: &slang::reflection::VariableLayout) -> Option<wgpu::BindingType> {
        let global_layout = program_reflection.global_params_type_layout();

        let binding_type = global_layout.descriptor_set_descriptor_range_type(0, index as i64);

        // Special case.. TODO: Remove this as soon as the reflection API properly reports write-only textures.
        if parameter.variable().name().unwrap() == "outputTexture" {
            return Some(wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 });
        }

        match binding_type {
            slang::BindingType::Texture => Some(wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
            }),
            slang::BindingType::MutableTeture => Some(wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::ReadWrite, format: wgpu::TextureFormat::R32Float, view_dimension: wgpu::TextureViewDimension::D2 }),
            slang::BindingType::ConstantBuffer => Some(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            slang::BindingType::MutableTypedBuffer
            | slang::BindingType::MutableRawBuffer => Some(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            }),
            _ => None,
        }
    }

    fn get_resource_bindings(&self, linked_program: &slang::ComponentType) -> HashMap<String, wgpu::BindGroupLayoutEntry> {
        let reflection = linked_program.layout(0).unwrap(); // assume target-index = 0

        let count = reflection.parameter_count();

        let mut resource_descriptors = HashMap::new();
        for i in 0..count {
            let parameter = reflection.parameter_by_index(i).unwrap();
            let name = parameter.variable().name().unwrap().to_string();

            let resource_info = self.get_binding_descriptor(parameter.binding_index(), reflection, parameter);
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

    pub fn compile(&self, entry_point_name: String) -> CompilationResult {
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
            // let error = self.slang_wasm_module.get_last_error();
            // console.error(error.type + " error: " + error.message);
            // self.diagnostics_msg += (error.type + " error: " + error.message);
            // TODO
            panic!();
        };

        let mut components: Vec<slang::ComponentType> = vec![];

        let user_module = slang_session.load_module("user.slang").unwrap();
        self.add_active_entry_points(&slang_session, &entry_point_name,  user_module, &mut components);
        let program = slang_session.create_composite_component_type(components.as_slice()).unwrap();
        let linked_program = program.link().unwrap();
        // let hashed_strings = linked_program.load_strings(); TODO

        let out_code = linked_program.entry_point_code(0 /* entry_point_index */, 0 /* target_index */).unwrap()
        .as_slice()
        .to_vec();
    //convert to string
        let out_code = String::from_utf8(out_code).unwrap();

        let bindings = self.get_resource_bindings(&linked_program);

        // Also read the shader work-group size.
        let entry_point_reflection = linked_program.layout(0).unwrap().find_entry_point_by_name(entry_point_name.as_str()).unwrap();
        // let thread_group_size = entry_point_reflection.get_compute_thread_group_size(); TODO

        // let reflection_json = linked_program.layout(0).to_json_object(); TODO

        return CompilationResult {
            out_code,
            bindings,
        };
    }
}