use super::{operation::Operation, shape::Shape, tensor::TensorID};

pub struct InternalTensor {
    pub id: TensorID,
    pub shape: Shape,
    pub operation: Operation,
    pub data_start_index: usize,
    pub grad_start_index: usize
}

impl InternalTensor {
    pub fn new(id: TensorID, shape: Shape, operation: Operation, data_start_index: usize, grad_start_index: usize) -> InternalTensor {
        InternalTensor {
            id,
            shape,
            operation,
            data_start_index,
            grad_start_index
        }
    }

    pub fn dependencies(&self) -> Vec<TensorID> {
        match self.operation {
            Operation::Nop => vec![],
            Operation::Add(a, b) => vec![a, b]
        }
    }
}