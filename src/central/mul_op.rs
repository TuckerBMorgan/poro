use crate::central::get_equation;
use crate::central::operation::Operation;
use crate::central::tensor::Tensor;
use crate::central::BackpropagationPacket;
use std::ops::{Div, Mul, Neg};

use super::tensor::NAME_LENGTH;

/// This function is used to perform the backward pass for the multiplication operation.
/// It takes in a `BackpropagationPacket` and then sets the gradients of the left and right hand side tensors.
/// # Arguments
///     
/// * `backprop_packet` - A `BackpropagationPacket` that contains the information needed to perform the backward pass.
///
/// # Panics
/// This function will panic if the operation in the `BackpropagationPacket` is not a multiplication operation.
pub fn backward(backprop_packet: BackpropagationPacket) {
    if let Operation::Mul(a, b) = backprop_packet.operation {
        if backprop_packet.advanced_logging == true {
            println!("A: {:?}", a);
            println!("B: {:?}", b);
        }

        // Get the gradients of the left and right hand side tensors
        let left_hand_grad = backprop_packet.equation.get_tensor_grad(a);
        let right_hand_grad = backprop_packet.equation.get_tensor_grad(b);
        if backprop_packet.advanced_logging == true {
            println!("Left Hand Grad: {:?}", left_hand_grad);
            println!("Right Hand Grad: {:?}", right_hand_grad);
        }

        // Get the data of the left and right hand side tensors
        let left_hand_data = backprop_packet.equation.get_tensor_data(a);
        let right_hand_data = backprop_packet.equation.get_tensor_data(b);
        if backprop_packet.advanced_logging == true {
            println!("Left Hand Data: {:?}", left_hand_data);
            println!("Right Hand Data: {:?}", right_hand_data);
        }

        // The derivative of a * b is a' * b + a * b'
        let new_left_hand_grad = right_hand_data * backprop_packet.grad.clone();
        let new_right_hand_grad = left_hand_data * backprop_packet.grad;
        if backprop_packet.advanced_logging == true {
            println!("New Left Hand Grad: {:?}", new_left_hand_grad);
            println!("New Right Hand Grad: {:?}", new_right_hand_grad);
        }

        // Add the new gradients to the old gradients
        // and then set the new gradients
        let right_hand_grad = right_hand_grad + new_right_hand_grad;
        let left_hand_grad = left_hand_grad + new_left_hand_grad;
        if backprop_packet.advanced_logging == true {
            println!("Right Hand Grad: {:?}", right_hand_grad);
            println!("Left Hand Grad: {:?}", left_hand_grad);
        }

        // Set the new gradients
        backprop_packet.equation.set_tensor_grad(a, left_hand_grad);
        backprop_packet.equation.set_tensor_grad(b, right_hand_grad);
    } else {
        panic!("Invalid operation type for backward pass");
    }
}

/// Overload the multiplication operator for the Tensor struct
/// This will allow us to multiply two tensors together
///
/// If the tensors are not the same shape, we will broadcast the right hand side tensor
/// to the shape of the left hand side tensor
impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            // we need to broadcast the tensors
            let broaded_casted_rhs = rhs.broadcast(self.shape);
            let result_data = self.item() * broaded_casted_rhs.item();

            let mut singleton = get_equation();

            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(
                self.shape.clone(),
                data_as_vec,
                Operation::Mul(self.tensor_id, broaded_casted_rhs.tensor_id),
            );

            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Mul(self.tensor_id, broaded_casted_rhs.tensor_id),
                name: ['a'; NAME_LENGTH],
            }
        } else {
            let result_data = self.item() * rhs.item();

            let mut singleton = get_equation();

            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(
                self.shape.clone(),
                data_as_vec,
                Operation::Mul(self.tensor_id, rhs.tensor_id),
            );

            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Mul(self.tensor_id, rhs.tensor_id),
                name: ['a'; NAME_LENGTH],
            }
        }
    }
}

/// Overload the multiplication operator for the Tensor struct
/// This will allow us to multiply a tensor by a scalar
/// we will convert the scalar into a tensor and then multiply the two tensors together
impl Mul<f32> for Tensor {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self * right_hand_as_tesnor
    }
}

/// Convinence function to allow us to handle the element wise negation of a tensor
impl Neg for Tensor {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

/// Overload the Division operator for the Tensor struct
/// we are going to use the fact that a/b = a * b^-1
impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        // we take advantage of the fact that a/b = a * b^-1
        // to let us keep the code simplier
        let intermidiate = rhs.pow(-1.0);
        self * intermidiate
    }
}

/// Overload the Division operator for the Tensor struct
/// This will allow us to divide a tensor by a scalar
/// we will convert the scalar into a tensor and then divide the two tensors together
impl Div<f32> for Tensor {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self / right_hand_as_tesnor
    }
}

/// Overload the Multiplication operator for the f32 struct
/// This will allow us to multiply a scalar by a tensor
/// we will convert the scalar into a tensor and then multiply the two tensors together
impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs * self
    }
}

mod test {
    #[allow(unused_imports)]
    use crate::central::shape::Shape;
    #[allow(unused_imports)]
    use crate::central::tensor::Tensor;
    #[allow(unused_imports)]
    use ndarray::prelude::*;

    #[test]
    fn mul_test() {
        let a = Tensor::ones(Shape::new(vec![2, 2]));
        let b = Tensor::ones(Shape::new(vec![2, 2]));
        let c = a * b;
        let result = c.item();
        assert!(result == arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
    }

    #[test]
    fn mul_test_2() {
        let a = Tensor::ones(Shape::new(vec![2, 2]));
        let b = Tensor::element(Shape::new(vec![2, 2]), 2.0);
        let c = a * b;
        let result = c.item();
        assert!(result == arr2(&[[2.0, 2.0], [2.0, 2.0]]).into_dyn());
    }

    #[test]
    fn backward_mul_test() {
        let a = Tensor::ones(Shape::new(vec![1, 1]));
        let b = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let c = a * b;
        c.backward();
        let result = c.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[2.0]]).into_dyn());
        let result = b.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
    }

    #[test]
    fn basic_div_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a / 2.0;
        let result = b.item();
        assert!(result == arr2(&[[1.0]]).into_dyn());
    }

    #[test]
    fn backward_div_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a / 2.0;
        b.backward();
        let result = b.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[0.5]]).into_dyn());
    }
}
