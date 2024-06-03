use crate::central::Tensor;
use crate::central::module::Module;

pub trait Model {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
}

pub struct LinearModel {
    layers: Vec<Box<dyn Module>>,
}

impl LinearModel {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        LinearModel { layers }
    }
}

impl Model for LinearModel {
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