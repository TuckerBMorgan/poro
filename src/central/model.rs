use std::collections::HashMap;
// include tensor
use crate::tensor::Tensor;

pub trait Model {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
}