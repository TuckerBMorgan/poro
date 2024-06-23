use super::Module;
use crate::central::Tensor;

pub struct Tanh {}

impl Tanh {
    #[allow(unused)]
    pub fn new() -> Tanh {
        Tanh {}
    }
}

impl Module for Tanh {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        x.tanh()
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

impl From<Tanh> for Box<dyn Module> {
    fn from(layer: Tanh) -> Box<dyn Module> {
        Box::new(layer)
    }
}
