use crate::{Indexable, Shape};

use super::tensor::TensorID;
use std::fmt;

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
    Mean(TensorID, usize),
    Concat(TensorID, TensorID),
    Reshape(TensorID, Shape),
    Tanh(TensorID),
    Transpose(TensorID, usize, usize),
    Sin(TensorID),
    Cos(TensorID),
    MaskedFill(TensorID, TensorID, isize),
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
            Operation::Mean(a, _) => Some(*a),
            Operation::Concat(a, _) => Some(*a),
            Operation::Reshape(a, _) => Some(*a),
            Operation::Tanh(a) => Some(*a),
            Operation::Transpose(a, _, _) => Some(*a),
            Operation::Sin(a) => Some(*a),
            Operation::Cos(a) => Some(*a),
            Operation::MaskedFill(a, _, _) => Some(*a),
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operation::Nop => write!(f, "Nop"),
            Operation::Add(_a, _b) => write!(f, "Add()"),
            Operation::Mul(_a, _b) => write!(f, "Mul()"),
            Operation::Exp(_a) => write!(f, "Exp()"),
            Operation::Pow(_a, _b) => write!(f, "Pow()"),
            Operation::MatMul(_a, _b) => write!(f, "MatMul()"),
            Operation::Sum(_a, _) => write!(f, "Sum()"),
            Operation::Broadcast(_a, _shape) => write!(f, "Broadcast()"),
            Operation::Log(_a) => write!(f, "Log()"),
            Operation::View(_a, _index) => write!(f, "View()"),
            Operation::Mean(_a, _) => write!(f, "Mean()"),
            Operation::Concat(_a, _b) => write!(f, "Concat()"),
            Operation::Reshape(_a, _shape) => write!(f, "Reshape()"),
            Operation::Tanh(_a) => write!(f, "Tanh()"),
            Operation::Transpose(_a, _, _) => write!(f, "Transpose()"),
            Operation::Sin(_a) => write!(f, "Sin()"),
            Operation::Cos(_a) => write!(f, "Cos()"),
            Operation::MaskedFill(_a, _b, _c) => write!(f, "MaskedFill()"),
        }
    }
}
