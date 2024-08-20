use crate::central::*;
use crate::nn::layers::*;

pub trait Model {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
}

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    #[allow(unused)]
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }

    #[allow(unused)]
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        for layer in &mut self.layers {
            layer.set_requires_grad(requires_grad);
        }
    }
}

impl Model for Sequential {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut output = x.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        for layer in &self.layers {
            parameters.extend(layer.get_parameters());
        }
        parameters
    }
}

impl From<Vec<Box<dyn Module>>> for Sequential {
    fn from(modules: Vec<Box<dyn Module>>) -> Sequential {
        Sequential::new(modules)
    }
}

#[test]
fn linear_model() {
    let mut linear_layer_config = LinearLayerConfig::default();
    linear_layer_config.number_of_inputs = 3;
    linear_layer_config.number_of_weights = 1;

    let mut linear_model = Sequential::new(vec![Box::new(LinearLayer::new(linear_layer_config))]);

    let inputs = vec![
        vec![2.0f32, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    let inputs_as_tensor = Tensor::from_vec(
        inputs.iter().flatten().map(|x| *x).collect(),
        vec![4, 3].into(),
    );

    let outputs = vec![1.0f32, -1.0, -1.0, 1.0];

    let outputs_as_tensor =
        Tensor::from_vec(outputs.iter().map(|x| *x).collect(), vec![4, 1].into());

    for _ in 0..50 {
        zero_all_grads();
        let prediction = linear_model.forward(&inputs_as_tensor);
        let loss = (prediction - outputs_as_tensor).pow(2.0);
        loss.backward();
        update_parameters(-0.01);
    }
}
