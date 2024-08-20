use crate::nn::Module;
use crate::Tensor;

pub struct Dropout {
    p: f32
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p }
    }
}

impl Module for Dropout {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // TODO: This is really slow, as it allocates new tensors for each operation
        let mask = Tensor::randn(input.shape);
        let mask = mask * self.p;
        let mask = mask / (1.0 - self.p);
        let mask = mask * *input;
        return mask;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}