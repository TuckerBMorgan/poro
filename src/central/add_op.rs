use crate::central::get_equation;
use crate::central::operation::Operation;
#[allow(unused_imports)]
use crate::central::shape::Shape;
use crate::central::tensor::Tensor;
use crate::central::BackpropagationPacket;
pub use ndarray::prelude::*;
use std::ops::{Add, Sub};

pub fn backward(backprop_packet: BackpropagationPacket) {
    if let Operation::Add(a, b) = backprop_packet.operation {
        // Get each of current tensor's gradient
        let left_hand_grad = backprop_packet.equation.get_tensor_grad(a);
        let right_hand_grad = backprop_packet.equation.get_tensor_grad(b);

        // derivative of a + b is a' + b' * global_grad
        backprop_packet
            .equation
            .set_tensor_grad(a, left_hand_grad + backprop_packet.grad.clone());
        backprop_packet
            .equation
            .set_tensor_grad(b, right_hand_grad + backprop_packet.grad);
    } else {
        panic!("Invalid operation type for backward pass");
    }
}

/// Overload the add operator for the Tensor struct
/// This will allow us to add two tensors together
/// If the tensors are not the same shape, we will broadcast the right hand side tensor
/// to the shape of the left hand side tensor
/// If the tensors are the same shape, we will just add them together
impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            // check if we need to broadcast the tensors, and then do so
            // will only broadcast the right hand side tensor
            let right_hand_broadcasted = rhs.broadcast(self.shape);
            let mut singleton = get_equation();

            let result_data =
                singleton.element_wise_add(self.tensor_id, right_hand_broadcasted.tensor_id);
            let tensor_id = singleton.allocate_tensor_from_operation(
                self.shape.clone(),
                result_data,
                Operation::Add(self.tensor_id, right_hand_broadcasted.tensor_id),
            );
            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Add(self.tensor_id, right_hand_broadcasted.tensor_id),
                name: ['a'; 10],
            }
        } else {
            let mut singleton = get_equation();
            // If they are the same size, preform the add and then return the result tensor
            let result_data = singleton.element_wise_add(self.tensor_id, rhs.tensor_id);

            let tensor_id = singleton.allocate_tensor_from_operation(
                self.shape.clone(),
                result_data,
                Operation::Add(self.tensor_id, rhs.tensor_id),
            );

            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Add(self.tensor_id, rhs.tensor_id),
                name: ['a'; 10],
            }
        }
    }
}

/// Overload the add operator for the Tensor struct
/// This will allow us to add a tensor and a f32 together
/// it will turn the f32 into a tensor and then add them together
impl Add<f32> for Tensor {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self + right_hand_as_tesnor
    }
}

/// Overload the sub operator for the Tensor struct
/// This will allow us to subtract two tensors together
/// it does it by negating the right hand side tensor and then adding them together
impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

// Overload the sub operator for the Tensor struct
// This will allow us to subtract a tensor and a f32 together
// it will turn the f32 into a tensor and then subtract them together
impl Sub<Tensor> for f32 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(rhs.shape.clone(), self);
        right_hand_as_tesnor - rhs
    }
}

// Overload the sub operator for the Tensor struct
// This will allow us to subtract a tensor and a f32 together
// it will turn the f32 into a tensor and then subtract them together
impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self - right_hand_as_tesnor
    }
}

// Overload the add operator for the f32 struct
// This will allow us to add a f32 and a tensor together
// it will turn the f32 into a tensor and then add them together
impl Add<Tensor> for f32 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        rhs + self
    }
}

#[test]
fn add_test() {
    // Test the add operation
    let a = Tensor::ones(Shape::new(vec![2, 2]));
    let b = Tensor::zeroes(Shape::new(vec![2, 2]));
    let c = a + b;
    let result = c.item();
    assert!(result == arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
}

#[test]
fn add_test_2() {
    let a = Tensor::zeroes(Shape::new(vec![2, 2]));
    let b = Tensor::zeroes(Shape::new(vec![2, 2]));
    let c = a + b;

    // Test using the result of the add operation
    let d = c.clone() + c;
    let result = d.item();
    assert!(result == arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn());
}

#[test]
fn backward_add_test() {
    // Basic verification of the backward pass
    let a = Tensor::ones(Shape::new(vec![1, 1]));
    let b = Tensor::element(Shape::new(vec![1, 1]), 2.0);
    let c = a + b;
    c.backward();
    let result = c.grad();
    assert!(result == arr2(&[[1.0]]).into_dyn());
    let result = a.grad();
    assert!(result == arr2(&[[1.0]]).into_dyn());
    let result = b.grad();
    assert!(result == arr2(&[[1.0]]).into_dyn());
}

#[test]
fn sub_test() {
    // sub is the same as add, but with a negative sign
    let a = Tensor::ones(Shape::new(vec![2, 2]));
    let b = Tensor::zeroes(Shape::new(vec![2, 2]));
    let c = a - b;
    let result = c.item();
    assert!(result == arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
}

#[test]
fn sub_test_2() {
    let a = Tensor::zeroes(Shape::new(vec![2, 2]));
    let b = Tensor::zeroes(Shape::new(vec![2, 2]));
    let c = a - b;
    let d = c.clone() - c;
    let result = d.item();
    assert!(result == arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn());
}

#[test]
fn backward_sub_test() {
    let a = Tensor::ones(Shape::new(vec![1, 1]));
    let b = Tensor::element(Shape::new(vec![1, 1]), 2.0);
    let c = a - b;
    c.backward();
    let result = c.grad();
    assert!(result == arr2(&[[1.0]]).into_dyn());
    let result = a.grad();
    assert!(result == arr2(&[[1.0]]).into_dyn());
    let result = b.grad();
    assert!(result == arr2(&[[-1.0]]).into_dyn());
}
