// This could maybe be its own lib, along with model.rs
use crate::{central::Tensor, Shape};

pub trait Module {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
    fn set_requires_grad(&mut self, requires_grad: bool) {
        for mut parameter in self.get_parameters() {
            parameter.set_requires_grad(requires_grad);
        }
    }
}

pub struct LinearLayer {
    weights: Tensor,
    bias: Tensor,
}

impl LinearLayer {
    #[allow(unused)]
    pub fn new(number_of_inputs: usize, number_of_weights: usize) -> LinearLayer {
        // Initialize the weights and bias tensors
        let weights = Tensor::randn(Shape::new(vec![number_of_inputs, number_of_weights]));
        let bias = Tensor::ones(Shape::new(vec![number_of_weights]));
        LinearLayer { weights, bias }
    }
}

impl Module for LinearLayer {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        // Perform the forward pass: x * weights + bias
        (*x << self.weights) + self.bias
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
