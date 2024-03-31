use crate::{Indexable, Shape};

use super::tensor::TensorID;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Operation {
    /// No operation, this will not pass any gradient 
    Nop,
    Add(TensorID, TensorID),
    Mul(TensorID, TensorID),
    Exp(TensorID),
    Pow(TensorID, TensorID),
    MatMul(TensorID, TensorID),
    Sum(TensorID),
    Broadcast(TensorID, Shape),
    Log(TensorID),
    View(TensorID, Indexable),
    Mean(TensorID),
}

impl Operation {
    pub fn get_tensor_id(&self) -> Option<TensorID> {
        match self {
            Operation::Nop => None,
            Operation::Add(a, _) => Some(*a),
            Operation::Mul(a, _) => Some(*a),
            Operation::Exp(a) => Some(*a),
            Operation::Pow(a, _) => Some(*a),
            Operation::MatMul(a, _) => Some(*a),
            Operation::Sum(a) => Some(*a),
            Operation::Broadcast(a, _) => Some(*a),
            Operation::Log(a) => Some(*a),
            Operation::View(a, _index) => Some(*a),
            Operation::Mean(a) => Some(*a),
        }
    }
}