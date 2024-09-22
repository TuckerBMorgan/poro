use serde::de;

// This could maybe be its own lib, along with model.rs
use crate::{central::Tensor, Shape};
use log::info;

pub trait Module {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
    fn set_requires_grad(&mut self, requires_grad: bool) {
        for mut parameter in self.get_parameters() {
            parameter.set_requires_grad(requires_grad);
        }
    }
}
#[derive(Debug, Default, Copy, Clone)]
pub struct LinearLayerConfig {
    pub number_of_inputs: usize,
    pub number_of_weights: usize,
}

impl LinearLayerConfig {
    pub fn new(number_of_inputs: usize, number_of_weights: usize) -> LinearLayerConfig {
        LinearLayerConfig { number_of_inputs, number_of_weights }
    }
}
pub struct LinearLayer {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl LinearLayer {

    pub fn new(config: LinearLayerConfig) -> LinearLayer {
        let weights = Tensor::randn(Shape::new(vec![config.number_of_inputs, config.number_of_weights]));
        let bias = Tensor::ones(Shape::new(vec![config.number_of_weights]));
        LinearLayer { weights, bias }
    }

    pub fn from_weights_and_bias(weights: Tensor, bias: Tensor) -> LinearLayer {
        LinearLayer { weights, bias }
    }
}

impl Module for LinearLayer {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let middle_product = *x << self.weights;
        middle_product + self.bias
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

impl From<LinearLayer> for Box<dyn Module> {
    fn from(layer: LinearLayer) -> Box<dyn Module> {
        Box::new(layer)
    }
}
