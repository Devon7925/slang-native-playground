use std::collections::HashMap;

use crate::{GPUResource, GraphicsAPI, ResourceCommandData, ResourceMetadata, safe_set};
use rand::Rng;
use serde::{Deserialize, Serialize};
#[cfg(feature = "compilation")]
use slang_reflector::{
    BoundResource, ScalarType, TextureType, UserAttributeParameter, VariableReflectionType,
};
#[cfg(feature = "compilation")]
use url::Url;
use wgpu::{BufferDescriptor, TextureFormat};
use winit::dpi::PhysicalSize;

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ZerosResourceCommand {
    count: u32,
    element_size: u32,
}

#[typetag::serde]
impl ResourceCommandData for ZerosResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "ZEROS".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        let BoundResource::StructuredBuffer {
            resource_result: element_type,
            ..
        } = resource
        else {
            panic!(
                "{} attribute cannot be applied to {variable_name}, it only supports buffers",
                Self::playground_name(),
            )
        };
        let [UserAttributeParameter::Int(count)] = parameters else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name(),
            )
        };
        assert!(
            *count >= 0,
            "{} count for {variable_name} cannot have negative size",
            Self::playground_name(),
        );
        Box::new(ZerosResourceCommand {
            count: *count as u32,
            element_size: element_type.get_size(),
        })
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource ${resource_name} is not defined in the bindings.");
        };

        if !matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
            panic!("Resource ${resource_name} is an invalid type for ZEROS");
        }

        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        if resource_metadata
            .get(resource_name)
            .unwrap_or(&vec![])
            .contains(&ResourceMetadata::Indirect)
        {
            usage |= wgpu::BufferUsages::INDIRECT;
        }

        let buffer = api.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&resource_name),
            mapped_at_creation: false,
            size: (self.count * self.element_size) as u64,
            usage,
        });

        // Initialize the buffer with zeros.
        let zeros = vec![0u8; (self.count * self.element_size) as usize];
        api.queue.write_buffer(&buffer, 0, &zeros);

        Ok(GPUResource::Buffer(buffer))
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct RandResourceCommand {
    count: u32,
}

#[typetag::serde]
impl ResourceCommandData for RandResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "RAND".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        assert!(
            matches!(resource, BoundResource::StructuredBuffer{ resource_result: element, ..} if matches!(element, VariableReflectionType::Scalar(ScalarType::Float32))),
            "{} attribute cannot be applied to {variable_name}, it only supports float buffers",
            Self::playground_name()
        );
        let [UserAttributeParameter::Int(count)] = parameters else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };
        assert!(
            *count >= 0,
            "{} count for {variable_name} cannot have negative size",
            Self::playground_name(),
        );
        Box::new(RandResourceCommand {
            count: *count as u32,
        })
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let element_size = 4; // RAND is only valid for floats

        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource ${resource_name} is not defined in the bindings.");
        };

        if !matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
            panic!("Resource ${resource_name} is an invalid type for ZEROS");
        }

        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        if resource_metadata
            .get(resource_name)
            .unwrap_or(&vec![])
            .contains(&ResourceMetadata::Indirect)
        {
            usage |= wgpu::BufferUsages::INDIRECT;
        }

        let buffer = api.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&resource_name),
            mapped_at_creation: false,
            size: (self.count * element_size) as u64,
            usage,
        });

        // Initialize the buffer.
        let rng = rand::rng();
        let data = rng
            .random_iter::<f32>()
            .take(self.count as usize)
            .collect::<Vec<_>>();

        api.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data));

        Ok(GPUResource::Buffer(buffer))
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BlackResourceCommand {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

#[typetag::serde]
impl ResourceCommandData for BlackResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "BLACK".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        let BoundResource::Texture {
            tex_type: TextureType::Dim2,
            resource_result: resource_type,
            format,
            ..
        } = resource
        else {
            panic!(
                "{} attribute cannot be applied to {variable_name}, it only supports 2D textures",
                Self::playground_name(),
            )
        };

        let [
            UserAttributeParameter::Int(width),
            UserAttributeParameter::Int(height),
        ] = parameters[..]
        else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        assert!(
            width >= 0,
            "{} width for {variable_name} cannot have negative size",
            Self::playground_name(),
        );
        assert!(
            height >= 0,
            "{} height for {variable_name} cannot have negative size",
            Self::playground_name(),
        );

        Box::new(BlackResourceCommand {
            width: width as u32,
            height: height as u32,
            format: crate::slang_compile::get_wgpu_format_from_slang_format(format, resource_type),
        })
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        _resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let size = self.width * self.height;
        let element_size = self.format.block_copy_size(None).unwrap();
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource {} is not defined in the bindings.", resource_name);
        };

        if !matches!(
            binding_info.ty,
            wgpu::BindingType::StorageTexture { .. } | wgpu::BindingType::Texture { .. }
        ) {
            panic!("Resource {} is an invalid type for BLACK", resource_name);
        }
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let texture = api.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            format: self.format,
            usage,
            view_formats: &[],
        });

        // Initialize the texture with zeros.
        let zeros = vec![0; (size * element_size) as usize];
        api.queue.write_texture(
            texture.as_image_copy(),
            &zeros,
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(self.width * element_size),
                offset: 0,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        Ok(GPUResource::Texture(texture))
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Black3DResourceCommand {
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
    pub format: TextureFormat,
}

#[typetag::serde]
impl ResourceCommandData for Black3DResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "BLACK_3D".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        let BoundResource::Texture {
            tex_type: TextureType::Dim3,
            resource_result: resource_type,
            format,
            ..
        } = resource
        else {
            panic!(
                "{} attribute cannot be applied to {variable_name}, it only supports 3D textures",
                Self::playground_name(),
            )
        };

        let [
            UserAttributeParameter::Int(size_x),
            UserAttributeParameter::Int(size_y),
            UserAttributeParameter::Int(size_z),
        ] = parameters[..]
        else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        macro_rules! check_positive {
            ($id:ident) => {
                if $id < 0 {
                    panic!(
                        "{} {} for {variable_name} cannot have negative size",
                        Self::playground_name(),
                        stringify!($id),
                    )
                }
            };
        }

        check_positive!(size_x);
        check_positive!(size_y);
        check_positive!(size_z);

        Box::new(Black3DResourceCommand {
            size_x: size_x as u32,
            size_y: size_y as u32,
            size_z: size_z as u32,
            format: crate::slang_compile::get_wgpu_format_from_slang_format(format, resource_type),
        })
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        _resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let size = self.size_x * self.size_y * self.size_z;
        let element_size = self.format.block_copy_size(None).unwrap();
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource {} is not defined in the bindings.", resource_name);
        };

        if !matches!(
            binding_info.ty,
            wgpu::BindingType::StorageTexture { .. } | wgpu::BindingType::Texture { .. }
        ) {
            panic!("Resource {} is an invalid type for BLACK_3D", resource_name);
        }
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let texture = api.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D3,
            mip_level_count: 1,
            sample_count: 1,
            size: wgpu::Extent3d {
                width: self.size_x,
                height: self.size_y,
                depth_or_array_layers: self.size_z,
            },
            format: self.format,
            usage,
            view_formats: &[],
        });

        // Initialize the texture with zeros.
        let zeros = vec![0; (size * element_size) as usize];
        api.queue.write_texture(
            texture.as_image_copy(),
            &zeros,
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(self.size_x * element_size),
                offset: 0,
                rows_per_image: Some(self.size_y),
            },
            wgpu::Extent3d {
                width: self.size_x,
                height: self.size_y,
                depth_or_array_layers: self.size_z,
            },
        );

        Ok(GPUResource::Texture(texture))
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BlackScreenResourceCommand {
    pub width_scale: f32,
    pub height_scale: f32,
    pub format: TextureFormat,
}

#[typetag::serde]
impl ResourceCommandData for BlackScreenResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "BLACK_SCREEN".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        let BoundResource::Texture {
            tex_type: TextureType::Dim2,
            resource_result: resource_type,
            format,
            ..
        } = resource
        else {
            panic!(
                "{} attribute cannot be applied to {variable_name}, it only supports 2D textures",
                Self::playground_name(),
            )
        };

        let [
            UserAttributeParameter::Float(width_scale),
            UserAttributeParameter::Float(height_scale),
        ] = parameters[..]
        else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        assert!(
            width_scale >= 0.0,
            "{} width for {variable_name} cannot have negative size",
            Self::playground_name(),
        );
        assert!(
            height_scale >= 0.0,
            "{} height for {variable_name} cannot have negative size",
            Self::playground_name(),
        );

        Box::new(BlackScreenResourceCommand {
            width_scale,
            height_scale,
            format: crate::slang_compile::get_wgpu_format_from_slang_format(format, resource_type),
        })
    }

    fn handle_resize(&self, api: GraphicsAPI, resource_name: &String, new_size: PhysicalSize<u32>) {
        let width = (self.width_scale * new_size.width as f32) as u32;
        let height = (self.height_scale * new_size.height as f32) as u32;
        let size = width * height;
        let element_size = self.format.block_copy_size(None).unwrap();
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource {} is not defined in the bindings.", resource_name);
        };

        if !matches!(
            binding_info.ty,
            wgpu::BindingType::StorageTexture { .. } | wgpu::BindingType::Texture { .. }
        ) {
            panic!("Resource {} is an invalid type for BLACK", resource_name);
        }
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let texture = api.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(resource_name),
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: self.format,
            usage,
            view_formats: &[],
        });

        // Initialize the texture with zeros.
        let zeros = vec![0; (size * element_size) as usize];
        api.queue.write_texture(
            texture.as_image_copy(),
            &zeros,
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(width * element_size),
                offset: 0,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let mut encoder = api.device.create_command_encoder(&Default::default());
        // copy old texture to new texture
        let Some(GPUResource::Texture(old_texture)) = api.allocated_resources.get(resource_name)
        else {
            panic!("Resource {} is not a Texture", resource_name);
        };
        encoder.copy_texture_to_texture(
            old_texture.as_image_copy(),
            texture.as_image_copy(),
            wgpu::Extent3d {
                width: width.min(old_texture.width()),
                height: height.min(old_texture.height()),
                depth_or_array_layers: 1,
            },
        );
        api.queue.submit(Some(encoder.finish()));

        safe_set(
            api.allocated_resources,
            resource_name.to_string(),
            GPUResource::Texture(texture),
        );
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        _resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let width = (self.width_scale * window_size.width as f32) as u32;
        let height = (self.height_scale * window_size.height as f32) as u32;
        let size = width * height;
        let element_size = self.format.block_copy_size(None).unwrap();
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource {} is not defined in the bindings.", resource_name);
        };

        if !matches!(
            binding_info.ty,
            wgpu::BindingType::StorageTexture { .. } | wgpu::BindingType::Texture { .. }
        ) {
            panic!("Resource {} is an invalid type for BLACK", resource_name);
        }
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        if matches!(binding_info.ty, wgpu::BindingType::StorageTexture { .. }) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let texture = api.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&resource_name),
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: self.format,
            usage,
            view_formats: &[],
        });

        // Initialize the texture with zeros.
        let zeros = vec![0; (size * element_size) as usize];
        api.queue.write_texture(
            texture.as_image_copy(),
            &zeros,
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(width * element_size),
                offset: 0,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        api.queue.submit(None);

        Ok(GPUResource::Texture(texture))
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct UrlResourceCommand {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

#[typetag::serde]
impl ResourceCommandData for UrlResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "URL".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        let BoundResource::Texture {
            tex_type: TextureType::Dim2,
            resource_result: resource_type,
            format,
            ..
        } = resource
        else {
            panic!(
                "{} attribute cannot be applied to {variable_name}, it only supports 2D textures",
                Self::playground_name(),
            )
        };

        let format = crate::slang_compile::get_wgpu_format_from_slang_format(format, resource_type);

        let [UserAttributeParameter::String(url)] = &parameters[..] else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        let parsed_url = Url::parse(&url);
        let image_bytes = if let Err(url::ParseError::RelativeUrlWithoutBase) = parsed_url {
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
            TextureFormat::Rgba8Unorm => image.to_rgba8().to_vec(),
            TextureFormat::R8Unorm => image
                .to_rgba8()
                .to_vec()
                .iter()
                .enumerate()
                .filter(|(i, _)| (*i % 4) < 1)
                .map(|(_, c)| c)
                .cloned()
                .collect(),
            TextureFormat::Rg8Unorm => image
                .to_rgba8()
                .to_vec()
                .iter()
                .enumerate()
                .filter(|(i, _)| (*i % 4) < 2)
                .map(|(_, c)| c)
                .cloned()
                .collect(),
            TextureFormat::Rgba32Float => image
                .to_rgba32f()
                .to_vec()
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .collect(),
            TextureFormat::Rg32Float => image
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

        Box::new(UrlResourceCommand {
            data,
            width: image.width(),
            height: image.height(),
            format,
        })
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        _resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        // Load image from URL and wait for it to be ready.
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource {} is not defined in the bindings.", resource_name);
        };

        let element_size = self.format.block_copy_size(None).unwrap();

        if !matches!(binding_info.ty, wgpu::BindingType::Texture { .. }) {
            panic!("Resource ${resource_name} is not a texture.");
        }
        let texture = api.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            format: self.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });
        api.queue.write_texture(
            texture.as_image_copy(),
            &self.data,
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(self.width * element_size),
                offset: 0,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(GPUResource::Texture(texture))
    }
}

#[cfg(feature = "compilation")]
pub enum ModelField {
    Position,
    Normal,
    TexCoords,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ModelResourceCommand {
    pub data: Vec<u8>,
}

#[typetag::serde]
impl ResourceCommandData for ModelResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "MODEL".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        let BoundResource::StructuredBuffer {
            resource_result: element_type,
            ..
        } = resource
        else {
            panic!(
                "{} attribute cannot be applied to {variable_name}, it only supports buffers",
                Self::playground_name(),
            )
        };
        let VariableReflectionType::Struct(fields) = element_type else {
            panic!(
                "{} attribute cannot be applied to {variable_name}, inner type must be struct",
                Self::playground_name(),
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
                            "Unsupported type for {field_name} field of MODEL struct for {variable_name}"
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
                            "Unsupported type for {field_name} field of MODEL struct for {variable_name}"
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
                            "Unsupported type for {field_name} field of MODEL struct for {variable_name}"
                        )
                    }
                    field_types.push(ModelField::TexCoords)
                }
                field_name => panic!(
                    "{field_name} is not a valid field for MODEL attribute on {variable_name}, valid fields are: position, normal, uv",
                ),
            }
        }

        let [UserAttributeParameter::String(path)] = &parameters[..] else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
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
                                &mesh.positions[3 * mesh.indices[i] as usize].to_le_bytes(),
                            );
                            data.extend_from_slice(
                                &mesh.positions[3 * mesh.indices[i] as usize + 1].to_le_bytes(),
                            );
                            data.extend_from_slice(
                                &mesh.positions[3 * mesh.indices[i] as usize + 2].to_le_bytes(),
                            );
                            data.extend_from_slice(&1.0f32.to_le_bytes());
                        }
                        ModelField::Normal => {
                            data.extend_from_slice(
                                &mesh.normals[3 * mesh.normal_indices[i] as usize].to_le_bytes(),
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
                                &mesh.texcoords[2 * mesh.texcoord_indices[i] as usize + 1]
                                    .to_le_bytes(),
                            );
                            data.extend_from_slice(&1.0f32.to_le_bytes());
                            data.extend_from_slice(&1.0f32.to_le_bytes());
                        }
                    }
                }
            }
        }

        Box::new(ModelResourceCommand { data })
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        _resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource ${resource_name} is not defined in the bindings.");
        };

        if !matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
            panic!("Resource ${resource_name} is an invalid type for MODEL");
        }

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let buffer = api.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&resource_name),
            mapped_at_creation: false,
            size: self.data.len() as u64,
            usage,
        });

        // Initialize the buffer with zeros.
        api.queue.write_buffer(&buffer, 0, &self.data);

        Ok(GPUResource::Buffer(buffer))
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct RebindForDrawResourceCommand {
    pub original_resource: String,
}

#[typetag::serde]
impl ResourceCommandData for RebindForDrawResourceCommand {
    fn is_available_in_compute(&self) -> bool {
        false
    }

    fn get_rebind_original_resource(&self) -> Option<&String> {
        Some(&self.original_resource)
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "REBIND_FOR_DRAW".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        assert!(
            matches!(
                resource,
                BoundResource::Texture {
                    tex_type: TextureType::Dim2,
                    ..
                } | BoundResource::StructuredBuffer { .. }
            ),
            "{} attribute cannot be applied to {variable_name}, it only supports 2D textures and structured buffers",
            Self::playground_name(),
        );

        let [UserAttributeParameter::String(original_resource)] = &parameters[..] else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        Box::new(RebindForDrawResourceCommand {
            original_resource: original_resource.clone(),
        })
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        _resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource {} is not defined in the bindings.", resource_name);
        };

        let resource = if matches!(binding_info.ty, wgpu::BindingType::Texture { .. }) {
            let Some(GPUResource::Texture(tex)) = api.allocated_resources.get(&self.original_resource)
            else {
                return Err(());
            };

            let texture = api.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                size: wgpu::Extent3d {
                    width: tex.width(),
                    height: tex.height(),
                    depth_or_array_layers: 1,
                },
                format: tex.format(),
                usage: tex.usage(),
                view_formats: &[],
            });
            GPUResource::Texture(texture)
        } else if matches!(binding_info.ty, wgpu::BindingType::Buffer { .. }) {
            let Some(GPUResource::Buffer(buf)) = api.allocated_resources.get(&self.original_resource)
            else {
                return Err(());
            };

            let buffer = api.device.create_buffer(&BufferDescriptor {
                label: None,
                size: buf.size(),
                usage: buf.usage(),
                mapped_at_creation: false,
            });
            GPUResource::Buffer(buffer)
        } else {
            panic!(
                "Resource {} is an invalid type for REBIND_FOR_DRAW",
                resource_name
            )
        };

        Ok(resource)
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct SamplerResourceCommand;

#[typetag::serde]
impl ResourceCommandData for SamplerResourceCommand {
    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "SAMPLER".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        resource: &BoundResource,
        _parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn ResourceCommandData> {
        if !matches!(resource, BoundResource::Sampler) {
            panic!(
                "{} attribute cannot be applied to {variable_name}, it only supports samplers",
                Self::playground_name(),
            )
        }
        Box::new(SamplerResourceCommand)
    }

    fn assign_resources(
        &self,
        api: GraphicsAPI,
        _resource_metadata: &HashMap<String, Vec<ResourceMetadata>>,
        resource_name: &String,
        _window_size: PhysicalSize<u32>,
    ) -> Result<GPUResource, ()> {
        let Some(binding_info) = api.resource_bindings.get(resource_name) else {
            panic!("Resource ${resource_name} is not defined in the bindings.");
        };

        if !matches!(binding_info.ty, wgpu::BindingType::Sampler { .. }) {
            panic!("Resource ${resource_name} is an invalid type for Sampler");
        }

        let sampler = api.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            ..Default::default()
        });

        Ok(GPUResource::Sampler(sampler))
    }
}