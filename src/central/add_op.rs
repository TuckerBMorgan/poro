use std::ops::{Add, Sub};
use crate::central::tensor::Tensor;
use crate::central::operation::Operation;
#[allow(unused_imports)]
use crate::central::shape::Shape;
pub use ndarray::prelude::*;
use crate::central::BackpropagationPacket;
use crate::central::get_equation;

pub fn backward(backprop_packet: BackpropagationPacket) {
    if let Operation::Add(a, b) = backprop_packet.operation {
        
        // Get each of current tensor's gradient
        let left_hand_grad = backprop_packet.equation.get_tensor_grad(a);
        let right_hand_grad = backprop_packet.equation.get_tensor_grad(b);

        // derivative of a + b is a' + b' * global_grad  
        backprop_packet.equation.set_tensor_grad(a, left_hand_grad + backprop_packet.grad.clone());
        backprop_packet.equation.set_tensor_grad(b, right_hand_grad + backprop_packet.grad);
    }
    else {
        panic!("Invalid operation type for backward pass");
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        
        // grab the data so we can do some shape checks
        let left_hand_item = self.item();
        let right_hand_item = rhs.item();

        if left_hand_item.shape() != right_hand_item.shape() {
            // check if we need to broadcast the tensors, and then do so
            // will only broadcast the right hand side tensor
            let right_hand_broadcasted = rhs.broadcast(self.shape);
            let result_data = self.item() + right_hand_broadcasted.item();

            let mut singleton = get_equation();

            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(
                self.shape.clone(),
                data_as_vec,
                Operation::Add(self.tensor_id, right_hand_broadcasted.tensor_id),
            );

            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Add(self.tensor_id, right_hand_broadcasted.tensor_id),
                name: ['a'; 10],
            }
        } else {
            // If they are the same size, preform the add and then return the result tensor
            let result_data = self.item() + rhs.item();
            let mut singleton = get_equation();

            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(
                self.shape.clone(),
                data_as_vec,
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

impl Add<f32> for Tensor {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self + right_hand_as_tesnor
    }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}


impl Sub<Tensor> for f32 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(rhs.shape.clone(), self);
        right_hand_as_tesnor - rhs
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self - right_hand_as_tesnor
    }
}

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