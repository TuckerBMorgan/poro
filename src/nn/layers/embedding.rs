use crate::{Shape, Tensor};

use super::Module;

pub struct Embedding {
    tensor: Tensor
}

impl Embedding {
    pub fn new(vocab_size: usize, model_dimension: usize) -> Embedding {
        let tensor = Tensor::randn(Shape::new(vec![vocab_size, model_dimension]));
        Embedding { tensor }
    }

    pub fn from_tensor(tensor: Tensor) -> Embedding {
        Embedding { tensor }
    }
}

impl Module for Embedding {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.tensor.view(crate::Indexable::FromTensor(input.tensor_id))
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.tensor.clone()]
    }
}