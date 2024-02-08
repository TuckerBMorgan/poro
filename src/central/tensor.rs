use std::ops::Add;
use ndarray::prelude::*;

use super::{operation::Operation, shape::Shape};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TensorID {
    pub id: usize
}

impl TensorID {
    pub fn new(id: usize) -> TensorID {
        TensorID {
            id
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Tensor {
    pub tensor_id: TensorID,
    pub shape: Shape,
    pub operation: Operation
}

impl Tensor {


    pub fn get_id(&self) -> usize {
        self.tensor_id.id
    }

    pub fn zeroes(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_zero_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop
        }
    }

    pub fn ones(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_ones_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop
        }
    }

    pub fn item(&self) -> ArrayD<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone();
        let data = data.as_slice();
        let data = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
        data
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let result_data = self.item() + rhs.item();

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

        let data_as_vec = result_data.into_iter().collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Add(self.tensor_id, rhs.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Add(self.tensor_id, rhs.tensor_id)
        }
    }
}

