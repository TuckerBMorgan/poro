use crate::{Indexable, Shape};

use std::fmt;
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
    Sum(TensorID, usize),
    Broadcast(TensorID, Shape),
    Log(TensorID),
    View(TensorID, Indexable),
    Mean(TensorID),
    Concat(TensorID, TensorID),
    Reshape(TensorID, Shape),
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
            Operation::Sum(a, _) => Some(*a),
            Operation::Broadcast(a, _) => Some(*a),
            Operation::Log(a) => Some(*a),
            Operation::View(a, _index) => Some(*a),
            Operation::Mean(a) => Some(*a),
            Operation::Concat(a, _) => Some(*a),
            Operation::Reshape(a, _) => Some(*a),
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operation::Nop => write!(f, "Nop"),
            Operation::Add(a, b) => write!(f, "Add()"),
            Operation::Mul(a, b) => write!(f, "Mul()"),
            Operation::Exp(a) => write!(f, "Exp()"),
            Operation::Pow(a, b) => write!(f, "Pow()"),
            Operation::MatMul(a, b) => write!(f, "MatMul()"),
            Operation::Sum(a, _) => write!(f, "Sum()"),
            Operation::Broadcast(a, shape) => write!(f, "Broadcast()"),
            Operation::Log(a) => write!(f, "Log()"),
            Operation::View(a, index) => write!(f, "View()"),
            Operation::Mean(a) => write!(f, "Mean()"),
            Operation::Concat(a, b) => write!(f, "Concat()"),
            Operation::Reshape(a, shape) => write!(f, "Reshape()"),
        }
    }
}