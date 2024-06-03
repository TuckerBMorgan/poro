// include tensor
use crate::{central::Tensor, Shape};

pub trait Module {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
}

pub struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Linear {
    pub fn new(number_of_inputs: usize, number_of_weights: usize) -> Linear {
        // Initialize the weights and bias tensors
        let weights = Tensor::randn(Shape::new(vec![number_of_inputs, number_of_weights]));
        let bias = Tensor::ones(Shape::new(vec![number_of_weights]));
        Linear { weights, bias }
    }
}

impl Module for Linear {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        // Perform the forward pass: x * weights + bias
        (*x << self.weights) + self.bias
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

pub struct BatchNorm {
    gain: Tensor,
    bias: Tensor,
}

impl Module for BatchNorm {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        // Perform the forward pass: x * gain + bias
        let batch_mean = x.mean(0);
        let batch_variance = x.variance(0);
        let normalized = (*x - batch_mean) / batch_variance;
        normalized * self.gain + self.bias
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.gain.clone(), self.bias.clone()]
    }
}

/*
impl FnOnce<(&Tensor,)> for Linear {
    type Output = Tensor;

    extern "rust-call" fn call_once(self, args: (&Tensor,)) -> Tensor {
        panic!("This should never be called");
    }
}

impl FnMut<(&Tensor,)> for Linear {
    extern "rust-call" fn call_mut(&mut self, args: (&Tensor,)) -> Tensor {
        self.forward(args.0)
    }
}

impl Fn<(&Tensor,)> for Linear {
    extern "rust-call" fn call(&self, args: (&Tensor,)) -> Tensor {
        panic!("This should never be called");
    }
}
 */
