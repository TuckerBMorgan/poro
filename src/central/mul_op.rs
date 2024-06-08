use std::ops::{Mul, Div, Neg};
use crate::central::tensor::Tensor;
use crate::central::operation::Operation;
use crate::central::BackpropagationPacket;
use crate::central::get_equation;

pub fn backward(backprop_packet: BackpropagationPacket) {
    if let Operation::Mul(a, b ) = backprop_packet.operation {
        if backprop_packet.advanced_logging == true {
            println!("A: {:?}", a);
            println!("B: {:?}", b);
        }
        let left_hand_grad = backprop_packet.equation.get_tensor_grad(a);
        let right_hand_grad = backprop_packet.equation.get_tensor_grad(b);
        if backprop_packet.advanced_logging == true {
            println!("Left Hand Grad: {:?}", left_hand_grad);
            println!("Right Hand Grad: {:?}", right_hand_grad);
        }
    
        let left_hand_data = backprop_packet.equation.get_tensor_data(a);
        let right_hand_data = backprop_packet.equation.get_tensor_data(b);
        if backprop_packet.advanced_logging == true {
            println!("Left Hand Data: {:?}", left_hand_data);
            println!("Right Hand Data: {:?}", right_hand_data);
        }
    
        let new_left_hand_grad = right_hand_data * backprop_packet.grad.clone();
        let new_right_hand_grad = left_hand_data * backprop_packet.grad;
        if backprop_packet.advanced_logging == true {
            println!("New Left Hand Grad: {:?}", new_left_hand_grad);
            println!("New Right Hand Grad: {:?}", new_right_hand_grad);
        }
        let right_hand_grad = right_hand_grad + new_right_hand_grad;
        let left_hand_grad = left_hand_grad + new_left_hand_grad;
        if backprop_packet.advanced_logging == true {
            println!("Right Hand Grad: {:?}", right_hand_grad);
            println!("Left Hand Grad: {:?}", left_hand_grad);
        }
    
        backprop_packet.equation.set_tensor_grad(a, left_hand_grad);
        backprop_packet.equation.set_tensor_grad(b, right_hand_grad);
    }
    else {
        panic!("Invalid operation type for backward pass");
    }

}

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
                name: ['a'; 10],
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
                name: ['a'; 10],
            }
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self * right_hand_as_tesnor
    }
}

impl Neg for Tensor {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}


impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        // we take advantage of the fact that a/b = a * b^-1
        // to let us keep the code simplier
        let intermidiate = rhs.pow(-1.0);
        self * intermidiate
    }
}

impl Div<f32> for Tensor {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self / right_hand_as_tesnor
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs * self
    }
}
