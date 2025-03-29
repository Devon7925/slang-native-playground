use std::path::Path;

use proc_macro::TokenStream;
use quote::quote;
use slang_compiler::SlangCompiler;
use syn::{parse::{Parse, ParseStream}, LitStr, Token, parse_macro_input};
mod slang_compiler;

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
    let CompileShaderInput { shader_path, search_paths } = parse_macro_input!(input as CompileShaderInput);
    let shader_path = shader_path.value();
    
    let compiler = SlangCompiler::new();
    
    let search_paths: Vec<String> = search_paths.iter().map(|s| s.value()).collect();
    let mut search_paths = search_paths.iter().map(|s| s.as_str()).collect::<Vec<_>>();

    let this_file = file!();
    let shaders_path = Path::new(this_file).parent().unwrap().parent().unwrap().join("shaders");
    let shaders_path = shaders_path.canonicalize().unwrap();
    search_paths.push(shaders_path.to_str().unwrap());

    let compilation_result = compiler.compile(search_paths, &shader_path);

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

    let tokens = quote! {
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
    };

    TokenStream::from(tokens)
}
