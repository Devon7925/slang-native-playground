use serde::{Deserialize, Serialize};
#[cfg(feature = "compilation")]
use slang_reflector::{ScalarType, UserAttributeParameter, VariableReflectionType};

use crate::{UniformControllerType, UniformSourceData};

#[derive(Deserialize, Serialize)]
pub struct UniformSlider {
    pub value: f32,
    pub min: f32,
    pub max: f32,
}

#[typetag::serde]
impl UniformControllerType for UniformSlider {
    fn get_data(&self, _uniform_source_data: &UniformSourceData) -> Vec<u8> {
        self.value.to_le_bytes().to_vec()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.label(name);
        ui.add(egui::Slider::new(&mut self.value, self.min..=self.max));
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "SLIDER".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        uniform_type: &VariableReflectionType,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType> {
        assert!(
            matches!(
                uniform_type,
                VariableReflectionType::Scalar(ScalarType::Float32)
            ),
            "{} attribute cannot be applied to {variable_name}, it only supports float uniforms",
            Self::playground_name()
        );

        let [
            UserAttributeParameter::Float(value),
            UserAttributeParameter::Float(min),
            UserAttributeParameter::Float(max),
        ] = parameters
        else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        Box::new(UniformSlider {
            value: *value,
            min: *min,
            max: *max,
        })
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformColorPick {
    pub value: [f32; 3],
}

#[typetag::serde]
impl UniformControllerType for UniformColorPick {
    fn get_data(&self, _uniform_source_data: &UniformSourceData) -> Vec<u8> {
        self.value
            .iter()
            .map(|x| x.to_le_bytes())
            .flatten()
            .collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.label(name);
        ui.color_edit_button_rgb(&mut self.value);
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "COLOR_PICK".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        uniform_type: &VariableReflectionType,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType> {
        assert!(
            matches!(
                uniform_type,
                VariableReflectionType::Vector(ScalarType::Float32, 3 | 4)
            ),
            "{} attribute cannot be applied to {variable_name}, it only supports float3 or float4 vectors",
            Self::playground_name()
        );

        let [
            UserAttributeParameter::Float(red),
            UserAttributeParameter::Float(green),
            UserAttributeParameter::Float(blue),
        ] = parameters else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        Box::new(UniformColorPick {
            value: [*red, *green, *blue],
        })
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformMousePosition;

#[typetag::serde]
impl UniformControllerType for UniformMousePosition {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        let mut data = vec![0.0f32; 4];
        data[0] = uniform_source_data.last_mouse_down_pos[0];
        data[1] = uniform_source_data.last_mouse_down_pos[1];
        data[2] = uniform_source_data.last_mouse_clicked_pos[0];
        data[3] = uniform_source_data.last_mouse_clicked_pos[1];
        if uniform_source_data.mouse_down {
            data[2] = -data[2];
        }
        if uniform_source_data.mouse_clicked {
            data[3] = -data[3];
        }
        data.iter().map(|x| x.to_le_bytes()).flatten().collect()
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "MOUSE_POSITION".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        uniform_type: &VariableReflectionType,
        _parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType> {
        assert!(
            matches!(
                uniform_type,
                VariableReflectionType::Vector(ScalarType::Float32, 4)
            ),
            "{} attribute cannot be applied to {variable_name}, it only supports float4 vectors",
            Self::playground_name()
        );

        Box::new(UniformMousePosition)
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformTime;

#[typetag::serde]
impl UniformControllerType for UniformTime {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        let value = web_time::Instant::now()
            .duration_since(uniform_source_data.launch_time)
            .as_secs_f32();
        value.to_le_bytes().to_vec()
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "TIME".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        uniform_type: &VariableReflectionType,
        _parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType> {
        assert!(
            matches!(
                uniform_type,
                VariableReflectionType::Scalar(ScalarType::Float32)
            ),
            "{} attribute cannot be applied to {variable_name}, it only supports float uniforms",
            Self::playground_name()
        );

        Box::new(UniformTime)
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformDeltaTime;

#[typetag::serde]
impl UniformControllerType for UniformDeltaTime {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        uniform_source_data.delta_time.to_le_bytes().to_vec()
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "DELTA_TIME".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        uniform_type: &VariableReflectionType,
        _parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType> {
        assert!(
            matches!(
                uniform_type,
                VariableReflectionType::Scalar(ScalarType::Float32)
            ),
            "{} attribute cannot be applied to {variable_name}, it only supports float uniforms",
            Self::playground_name()
        );

        Box::new(UniformDeltaTime)
    }
}

#[derive(Deserialize, Serialize)]
pub struct UniformKeyInput {
    pub key: String,
}

#[typetag::serde]
impl UniformControllerType for UniformKeyInput {
    fn get_data(&self, uniform_source_data: &UniformSourceData) -> Vec<u8> {
        let value = if uniform_source_data.pressed_keys.contains(&self.key) {
            1.0f32
        } else {
            0.0f32
        };
        value.to_le_bytes().to_vec()
    }

    #[cfg(feature = "compilation")]
    fn playground_name() -> String {
        "KEY".to_string()
    }

    #[cfg(feature = "compilation")]
    fn construct(
        uniform_type: &VariableReflectionType,
        parameters: &[UserAttributeParameter],
        variable_name: &str,
    ) -> Box<dyn UniformControllerType> {
        assert!(
            matches!(
                uniform_type,
                VariableReflectionType::Scalar(ScalarType::Float32)
            ),
            "{} attribute cannot be applied to {variable_name}, it only supports float uniforms",
            Self::playground_name()
        );

        let [UserAttributeParameter::String(key)] = parameters else {
            panic!(
                "Invalid attribute parameter type for {} attribute on {variable_name}",
                Self::playground_name()
            )
        };

        Box::new(UniformKeyInput {
            key: key.clone()
        })
    }
}
