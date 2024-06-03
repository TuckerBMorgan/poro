use super::{operation::Operation, shape::Shape, tensor::TensorID};

pub struct InternalTensor {
    pub tensor_id: TensorID,
    pub shape: Shape,
    pub operation: Operation,
    pub data_start_index: usize,
    pub grad_start_index: usize,
    pub requires_grad: bool,
}

impl InternalTensor {
    pub fn new(
        tensor_id: TensorID,
        shape: Shape,
        operation: Operation,
        data_start_index: usize,
        grad_start_index: usize,
    ) -> InternalTensor {
        InternalTensor {
            tensor_id,
            shape,
            operation,
            data_start_index,
            grad_start_index,
            requires_grad: false,
        }
    }

    pub fn dependencies(&self) -> Vec<TensorID> {
        match self.operation {
            Operation::Nop => vec![],
            Operation::Add(a, b) => vec![a, b],
            Operation::Mul(a, b) => vec![a, b],
            Operation::Exp(a) => vec![a],
            Operation::Pow(base, power) => vec![base, power],
            Operation::MatMul(a, b) => vec![a, b],
            Operation::Sum(a, _) => vec![a],
            Operation::Broadcast(a, _) => vec![a],
            Operation::Log(a) => vec![a],
            Operation::View(a, _index) => vec![a],
            Operation::Mean(a) => vec![a],
            Operation::Concat(a, b) => vec![a, b],
            Operation::Reshape(a, _) => vec![a],
        }
    }
}
