use crate::{Shape, Tensor};

use super::Module;

pub struct Embedding {
    pub tensor: Tensor,
    vocab_size: usize,
    model_dimension: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, model_dimension: usize) -> Embedding {
        let tensor = Tensor::randn(Shape::new(vec![vocab_size, model_dimension]));
        Embedding { tensor, model_dimension, vocab_size }
    }

    pub fn from_pretrained(pretrained: Tensor) -> Embedding {
        Embedding { tensor: pretrained, model_dimension: pretrained.shape.indices[1], vocab_size: pretrained.shape.indices[0] }
    }

    pub fn from_tensor(tensor: Tensor) -> Embedding {
        Embedding { tensor, model_dimension: tensor.shape.indices[1], vocab_size: tensor.shape.indices[0] }
    }
}

impl Module for Embedding {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![input.shape.indices[0], input.shape.indices[1], self.model_dimension]));
        let data = self.tensor.item();
        for b in 0..input.shape.indices[0] {
            for t in 0..input.shape.indices[1] {
                let view = input.view([b, t].into());
                let index = view.item()[[0, 0]] as usize;
                for i in 0..self.model_dimension {

                    let datum = data[[index, i]];

                    test_index_tensor.set_index(
                        [b,t, i].into(),
                        vec![datum]
                    );
                }
            }
        }
        return test_index_tensor;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.tensor.clone()]
    }
}