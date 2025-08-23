use std::path::Path;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use slang_playground_compiler::{
    slang_compile::SlangCompiler, slang_reflector::{ScalarType, VariableReflectionType}, CompilationResult, UniformController
};
use syn::{
    LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

fn compilation_result_tokens(compilation_result: CompilationResult) -> proc_macro2::TokenStream {
    let out_code = compilation_result.out_code;
    let entry_group_sizes = compilation_result.entry_group_sizes;
    let bindings = compilation_result.bindings;
    let uniform_controllers = compilation_result.uniform_controllers;
    let resource_commands = compilation_result.resource_commands;
    let call_commands = compilation_result.call_commands;
    let draw_commands = compilation_result.draw_commands;
    let hashed_strings = compilation_result.hashed_strings;
    let uniform_size = compilation_result.uniform_size;

    // Create tokens for each map entry individually
    let entry_group_size_tokens = entry_group_sizes.into_iter().map(|(key, val)| {
        quote! {
            m.insert(#key.to_string(), [#(#val),*]);
        }
    });

    let binding_tokens = bindings.into_iter().map(|(key, entry)| {
        let binding = entry.binding;
        let visibility = entry.visibility.bits();
        let ty = ron::to_string(&entry.ty).unwrap();
        let count_value = entry.count.map(|c| c.get());
        let count = match count_value {
            Some(v) => quote! { Some(::std::num::NonZeroU32::new(#v).unwrap()) },
            None => quote! { None },
        };
        quote! {
            m.insert(#key.to_string(), wgpu::BindGroupLayoutEntry {
                binding: #binding,
                visibility: wgpu::ShaderStages::from_bits_truncate(#visibility),
                ty: ron::from_str(#ty).unwrap(),
                count: #count,
            });
        }
    });

    let resource_command_tokens = resource_commands.into_iter().map(|(key, cmd)| {
        let cmd_str = ron::to_string(&cmd).unwrap();
        quote! {
            m.insert(#key.to_string(), ron::from_str(#cmd_str).unwrap());
        }
    });

    let call_command_tokens = call_commands.into_iter().map(|cmd| {
        let cmd_str = ron::to_string(&cmd).unwrap();
        quote! { ron::from_str(#cmd_str).unwrap() }
    });

    let draw_command_tokens = draw_commands.into_iter().map(|cmd| {
        let cmd_str = ron::to_string(&cmd).unwrap();
        quote! { ron::from_str(#cmd_str).unwrap() }
    });

    let uniform_controller_tokens = uniform_controllers.into_iter().map(|ctrl| {
        let ctrl_str = ron::to_string(&ctrl).unwrap();
        quote! { ron::from_str(#ctrl_str).unwrap() }
    });

    let hashed_string_tokens = hashed_strings.into_iter().map(|(key, value)| {
        quote! {
            m.insert(#key, #value.to_string());
        }
    });

    quote! {
        CompilationResult {
            out_code: #out_code.to_string(),
            entry_group_sizes: {
                let mut m = ::std::collections::HashMap::new();
                #(#entry_group_size_tokens)*
                m
            },
            bindings: {
                let mut m = ::std::collections::HashMap::new();
                #(#binding_tokens)*
                m
            },
            uniform_controllers: vec![#(#uniform_controller_tokens),*],
            resource_commands: {
                let mut m = ::std::collections::HashMap::new();
                #(#resource_command_tokens)*
                m
            },
            call_commands: vec![#(#call_command_tokens),*],
            draw_commands: vec![#(#draw_command_tokens),*],
            hashed_strings: {
                let mut m = ::std::collections::HashMap::new();
                #(#hashed_string_tokens)*
                m
            },
            uniform_size: #uniform_size,
        }
    }
}

struct TypeData {
    definitions: Vec<proc_macro2::TokenStream>,
    usage: proc_macro2::TokenStream,
}

fn scalar_type_to_usage(slang_scalar_type: &ScalarType) -> proc_macro2::TokenStream {
    match slang_scalar_type {
        ScalarType::None => quote! { () },
        ScalarType::Void => quote! { () },
        ScalarType::Bool => quote! { bool },
        ScalarType::Int32 => quote! { i32 },
        ScalarType::Uint32 => quote! { u32 },
        ScalarType::Int64 => quote! { i64 },
        ScalarType::Uint64 => quote! { u64 },
        ScalarType::Float16 => todo!(),
        ScalarType::Float32 => quote! { f32 },
        ScalarType::Float64 => quote! { f64 },
        ScalarType::Int8 => quote! { i8 },
        ScalarType::Uint8 => quote! { u8 },
        ScalarType::Int16 => quote! { i16 },
        ScalarType::Uint16 => quote! { u16 },
        ScalarType::Intptr => todo!(),
        ScalarType::Uintptr => todo!(),
    }
}

fn variable_reflection_to_type_data(binding: &VariableReflectionType) -> TypeData {
    match binding {
        VariableReflectionType::Struct(struct_name, items) => {
            let (mut definitions, fields) = items
                .iter()
                .map(|(field_name, field_type)| {
                    let field_ident = format_ident!("{}", field_name);
                    let field_type_data = variable_reflection_to_type_data(field_type);
                    (field_ident, field_type_data)
                })
                .fold(
                    (Vec::new(), Vec::new()),
                    |(mut defs, mut fields), (field_ident, field_type_data)| {
                        defs.extend(field_type_data.definitions);
                        fields.push((field_ident, field_type_data.usage));
                        (defs, fields)
                    },
                );
            let struct_ident = format_ident!("{}", struct_name);
            let field_defs = fields.iter().map(|(field_ident, field_type)| {
                quote! { pub #field_ident: #field_type }
            });
            let struct_def = quote! {
                #[repr(C)]
                #[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
                pub struct #struct_ident {
                    #(#field_defs),*
                }
            };
            definitions.push(struct_def);

            TypeData {
                definitions,
                usage: quote! { #struct_ident },
            }
        }
        VariableReflectionType::Scalar(slang_scalar_type) => TypeData {
            definitions: Vec::new(),
            usage: scalar_type_to_usage(slang_scalar_type),
        },
        VariableReflectionType::Vector(slang_scalar_type, count) => {
            let scalar_usage = scalar_type_to_usage(slang_scalar_type);
            TypeData {
                definitions: Vec::new(),
                usage: quote! { [#scalar_usage; #count] },
            }
        }
        VariableReflectionType::Array(variable_reflection_type, count) => {
            let element_type_data = variable_reflection_to_type_data(variable_reflection_type);
            let element_usage = element_type_data.usage;
            TypeData {
                definitions: element_type_data.definitions,
                usage: quote! { [#element_usage; #count] },
            }
        }
    }
}

struct CompileShaderInput {
    shader_path: LitStr,
    search_paths: Vec<LitStr>,
}

impl Parse for CompileShaderInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let shader_path = input.parse()?;

        input.parse::<Token![,]>()?;
        let paths_content;
        syn::bracketed!(paths_content in input);
        let paths = paths_content.parse_terminated(|input| input.parse::<LitStr>(), Token![,])?;
        let search_paths = paths.into_iter().collect();

        Ok(CompileShaderInput {
            shader_path,
            search_paths,
        })
    }
}

#[proc_macro]
pub fn compile_shader(input: TokenStream) -> TokenStream {
    let CompileShaderInput {
        shader_path,
        search_paths,
    } = parse_macro_input!(input as CompileShaderInput);
    let shader_path = shader_path.value();

    let compiler = SlangCompiler::default();

    let search_paths: Vec<String> = search_paths.iter().map(|s| s.value()).collect();
    let mut search_paths = search_paths.iter().map(|s| s.as_str()).collect::<Vec<_>>();

    let this_file = file!();
    let shaders_path = Path::new(this_file)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("shaders");
    let shaders_path = shaders_path.canonicalize().unwrap();
    search_paths.push(shaders_path.to_str().unwrap());

    let compilation_result = compiler.compile(search_paths, &shader_path);

    let tokens = compilation_result_tokens(compilation_result);

    TokenStream::from(tokens)
}

#[proc_macro]
pub fn shader_module(input: TokenStream) -> TokenStream {
    let CompileShaderInput {
        shader_path,
        search_paths,
    } = parse_macro_input!(input as CompileShaderInput);
    let shader_path = shader_path.value();

    let compiler = SlangCompiler::default();

    let search_paths: Vec<String> = search_paths.iter().map(|s| s.value()).collect();
    let mut search_paths = search_paths.iter().map(|s| s.as_str()).collect::<Vec<_>>();

    let this_file = file!();
    let shaders_path = Path::new(this_file)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("shaders");
    let shaders_path = shaders_path.canonicalize().unwrap();
    search_paths.push(shaders_path.to_str().unwrap());

    let compilation_result = compiler.compile(search_paths, &shader_path);
    let resource_bound_data = compilation_result
        .resource_commands
        .iter()
        .filter_map(|(bind_name, cmd)| cmd.generate_binding().map(|binding| (bind_name.clone(), binding)))
        .map(|(name, binding)| (name, variable_reflection_to_type_data(&binding)))
        .collect::<Vec<_>>();
    let uniform_bound_data = compilation_result
        .uniform_controllers
        .iter()
        .filter_map(|UniformController { name, controller, .. }| controller.generate_binding().map(|binding| (name.clone(), binding)))
        .map(|(name, binding)| (name, variable_reflection_to_type_data(&binding)))
        .collect::<Vec<_>>();
    

    let renderer_integration = {
        #[cfg(not(feature = "renderer-integration"))]
        quote! {}
        #[cfg(feature = "renderer-integration")]
        {
            use heck::ToSnakeCase;
            let resource_functions = resource_bound_data.iter().map(|(name, data)| {
                let fn_name = format_ident!("set_{}", name.to_snake_case());
                let usage = data.usage.clone();
                quote! {
                    pub fn #fn_name(renderer: &mut Renderer, value: #usage) {
                        renderer.update_resource(#name, bytemuck::bytes_of(&value));
                    }
                }
            });
            let uniform_functions = uniform_bound_data.iter().map(|(name, data)| {
                let fn_name = format_ident!("set_{}", name.to_snake_case());
                let usage = data.usage.clone();
                quote! {
                    pub fn #fn_name(renderer: &mut Renderer, value: #usage) {
                        renderer.update_uniform(#name, bytemuck::bytes_of(&value));
                    }
                }
            });
            quote! {
                use slang_renderer::Renderer;
                #(#resource_functions)*
                #(#uniform_functions)*
            }
        }
    };

    let resource_definitions = resource_bound_data.into_iter().flat_map(|(_, data)| data.definitions);
    let uniform_definitions = uniform_bound_data.into_iter().flat_map(|(_, data)| data.definitions);

    let result_tokens = compilation_result_tokens(compilation_result);

    let tokens = quote! {
        use once_cell::sync::Lazy;
        use slang_playground_compiler::CompilationResult;
        #renderer_integration
        #(#resource_definitions)*
        #(#uniform_definitions)*
        pub static COMPILATION_RESULT: Lazy<CompilationResult> = Lazy::new(|| #result_tokens);
    };

    TokenStream::from(tokens)
}
