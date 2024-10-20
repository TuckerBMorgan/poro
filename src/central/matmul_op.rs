use crate::central::get_equation;
use crate::central::operation::Operation;
use crate::central::tensor::Tensor;
use crate::central::BackpropagationPacket;
use ndarray::prelude::*;
use std::ops::Shl;

use super::tensor::{name_from_string, NAME_LENGTH};

pub fn backward(backprop_packet: BackpropagationPacket) {
    if let Operation::MatMul(a, b) = backprop_packet.operation {
        // Handle the case when the gradient is a 2D matrix
        if backprop_packet.grad.ndim() == 1 {
            // Convert the gradient to a 2D matrix
            let out_grad = backprop_packet.grad.clone();
            // Get the data of the right-hand operand of the MatMul operation
            let right_hand_data = backprop_packet.equation.get_tensor_data(b);
            let right_hand_data_tranpose = right_hand_data.t();

            // Transpose the right-hand data

            // Get the gradient of the left-hand operand of the MatMul operation
            let left_hand_grad = backprop_packet.equation.get_tensor_grad(a);
            // Update the gradient of the left-hand operand
            // Gradient of A in A*B with respect to the loss is given by (dL/dZ) * B^T
            let hold = left_hand_grad
                + backprop_packet
                    .equation
                    .matmul(&out_grad, &right_hand_data_tranpose.to_owned());
            //let hold = left_hand_grad + out_grad.clone().dot(&other).into_dyn();
            backprop_packet.equation.set_tensor_grad(a, hold);

            // Get the data of the left-hand operand of the MatMul operation
            let left_hand_data = backprop_packet.equation.get_tensor_data(a);

            // Get the gradient of the right-hand operand of the MatMul operation
            let right_hand_grad = backprop_packet.equation.get_tensor_grad(b);
            // Transpose the left-hand data
            let other = left_hand_data.t();
            // Update the gradient of the right-hand operand
            // Gradient of B in A*B with respect to the loss is given by A^T * (dL/dZ)

            let other_len = other.len();
            let out_grad_len = out_grad.len();
            let other_reshape = other.into_shape((other_len, 1)).unwrap().to_owned().into_dyn();
            let out_grad_reshape = out_grad.into_shape((1, out_grad_len)).unwrap().to_owned().into_dyn();

            let temp = right_hand_grad
                + backprop_packet
                    .equation
                    .matmul(&other_reshape, &out_grad_reshape);

            backprop_packet.equation.set_tensor_grad(b, temp);
        }
        else if backprop_packet.grad.ndim() == 2 {
            // Convert the gradient to a 2D matrix
            let out_grad = backprop_packet.grad.clone();
            // Get the data of the right-hand operand of the MatMul operation
            let right_hand_data = backprop_packet.equation.get_tensor_data(b);
            let right_hand_data_tranpose = right_hand_data.t();

            // Transpose the right-hand data

            // Get the gradient of the left-hand operand of the MatMul operation
            let left_hand_grad = backprop_packet.equation.get_tensor_grad(a);
            // Update the gradient of the left-hand operand
            // Gradient of A in A*B with respect to the loss is given by (dL/dZ) * B^T
            let hold = left_hand_grad
                + backprop_packet
                    .equation
                    .matmul(&out_grad, &right_hand_data_tranpose.to_owned());
            //let hold = left_hand_grad + out_grad.clone().dot(&other).into_dyn();
            backprop_packet.equation.set_tensor_grad(a, hold);

            // Get the data of the left-hand operand of the MatMul operation
            let left_hand_data = backprop_packet.equation.get_tensor_data(a);
            // Get the gradient of the right-hand operand of the MatMul operation
            let right_hand_grad = backprop_packet.equation.get_tensor_grad(b);
            // Transpose the left-hand data
            let other = left_hand_data.t();
            // Update the gradient of the right-hand operand
            // Gradient of B in A*B with respect to the loss is given by A^T * (dL/dZ)
            let temp = right_hand_grad
                + backprop_packet
                    .equation
                    .matmul(&other.to_owned(), &out_grad);
            //            let temp = right_hand_grad + other.dot(&out_grad).into_dyn();
            backprop_packet.equation.set_tensor_grad(b, temp);
        } else if backprop_packet.grad.ndim() == 3 {
            // Handle the case when the gradient is a 3D tensor
            // Convert the gradient to a 3D tensor
            let out_grad = backprop_packet
                .grad
                .clone()
                .into_dimensionality::<Ix3>()
                .unwrap();
            // Get the data of the right-hand operand of the MatMul operation
            let right_hand_data = backprop_packet.equation.get_tensor_data(b);
            let right_hand_data_tranpose = right_hand_data.t();

            // Transpose the right-hand data
            let other = right_hand_data_tranpose
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Get the gradient of the left-hand operand of the MatMul operation
            let left_hand_grad = backprop_packet.equation.get_tensor_grad(a);
            let mut result = Array3::zeros((
                left_hand_grad.shape()[0],
                left_hand_grad.shape()[1],
                left_hand_grad.shape()[2],
            ));
            // Update the gradient of the left-hand operand for each slice
            for i in 0..out_grad.shape()[0] {
                let hold = out_grad
                    .slice(s![i, .., ..])
                    .dot(&other.slice(s![.., ..]))
                    .into_dyn();
                result.slice_mut(s![i, .., ..]).assign(&hold);
            }

            backprop_packet
                .equation
                .set_tensor_grad(a, result.into_dyn() + left_hand_grad);

            // Get the data of the left-hand operand of the MatMul operation
            let left_hand_data = backprop_packet.equation.get_tensor_data(a);
            // Get the gradient of the right-hand operand of the MatMul operation
            let right_hand_grad = backprop_packet.equation.get_tensor_grad(b);
            // Convert the left-hand data to a 3D tensor
            let other = left_hand_data.into_dimensionality::<Ix3>().unwrap();
            let mut result =
                Array2::zeros((right_hand_grad.shape()[0], right_hand_grad.shape()[1])).into_dyn();

            // Update the gradient of the right-hand operand for each slice
            for i in 0..out_grad.shape()[0] {
                let hold = other
                    .slice(s![i, .., ..])
                    .t()
                    .dot(&out_grad.slice(s![i, .., ..]))
                    .into_dyn();
                result = result + hold;
            }
            backprop_packet
                .equation
                .set_tensor_grad(b, result.into_dyn() + right_hand_grad);
        }     
        else {
            
            panic!("Not implementerd for dim {}", backprop_packet.grad.ndim());
        }
    } else {
        panic!("Invalid operation type for backward pass");
    }
}

/*
pub fn backward(backprop_packet: BackpropagationPacket) {
    if let Operation::MatMul(a, b) = backprop_packet.operation {


        if backprop_packet.grad.ndim() == 2 {
            let out_grad = backprop_packet.grad.clone().into_dimensionality::<Ix2>().unwrap();
            let right_hand_data = backprop_packet.equation.get_tensor_data(b);
            let right_hand_data_tranpose = right_hand_data.t();

            let other = right_hand_data_tranpose
                .into_dimensionality::<Ix2>()
                .unwrap();
            let left_hand_grad =  backprop_packet.equation.get_tensor_grad(a);
            let hold = left_hand_grad + out_grad.clone().dot(&other).into_dyn();
            backprop_packet.equation.set_tensor_grad(a, hold);

            let left_hand_data =  backprop_packet.equation.get_tensor_data(a);
            let right_hand_grad =  backprop_packet.equation.get_tensor_grad(b);
            let other = left_hand_data.t().into_dimensionality::<Ix2>().unwrap();
            let temp = right_hand_grad + other.dot(&out_grad).into_dyn();
            backprop_packet.equation.set_tensor_grad(b, temp);
        }
        else if backprop_packet.grad.ndim() == 3 {
            let out_grad = backprop_packet.grad.clone().into_dimensionality::<Ix3>().unwrap();
            let right_hand_data =  backprop_packet.equation.get_tensor_data(b);
            let right_hand_data_tranpose = right_hand_data.t();


            let other = right_hand_data_tranpose
                .into_dimensionality::<Ix2>()
                .unwrap();

            let left_hand_grad =  backprop_packet.equation.get_tensor_grad(a);
            let mut result = Array3::zeros((left_hand_grad.shape()[0], left_hand_grad.shape()[1], left_hand_grad.shape()[2]));
            for i in 0..out_grad.shape()[0] {
                let hold = out_grad.slice(s![i, .., ..]).dot(&other.slice(s![.., ..])).into_dyn();
                result.slice_mut(s![i, .., ..]).assign(&hold);
            }



            backprop_packet.equation.set_tensor_grad(a, result.into_dyn() + left_hand_grad);

            let left_hand_data =  backprop_packet.equation.get_tensor_data(a);
            let right_hand_grad =  backprop_packet.equation.get_tensor_grad(b);
            let other = left_hand_data.into_dimensionality::<Ix3>().unwrap();
            let mut result = Array2::zeros((right_hand_grad.shape()[0], right_hand_grad.shape()[1])).into_dyn();

            for i in 0..out_grad.shape()[0] {
                let hold = other.slice(s![i,..,..]).t().dot(&out_grad.slice(s![i, .., ..])).into_dyn();
                result = result + hold;
            }
            backprop_packet.equation.set_tensor_grad(b, result.into_dyn() + right_hand_grad);
        }
        else {
            panic!("Not implemented");
        }

    }
    else {
        panic!("Invalid operation type for backward pass");
    }
}
 */
// SIN: reusing the Shl opeartor to do the matmul operations
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {
        let mut singleton = get_equation();
        let a_data = singleton.get_tensor_data(self.tensor_id);
        let b_data = singleton.get_tensor_data(rhs.tensor_id);
        let result_data = singleton.matmul(&a_data, &b_data);

        let resultant_shape = self.shape.matmul_shape(&rhs.shape);
        let tensor_id = singleton.allocate_tensor_from_operation(
            resultant_shape,
            result_data.into_raw_vec(),
            Operation::MatMul(self.tensor_id, rhs.tensor_id),
        );
        let matmul_shape = self.shape.matmul_shape(&rhs.shape);

        Tensor {
            tensor_id,
            shape: matmul_shape,
            operation: Operation::MatMul(self.tensor_id, rhs.tensor_id),
            name: name_from_string("MatMul"),
        }
    }
}

mod tests {
    #[allow(unused_imports)]
    use crate::central::shape::Shape;
    #[allow(unused_imports)]
    use crate::central::tensor::Tensor;
    #[test]
    fn two_dimension_matmul_test() {
        let a = Tensor::ones(Shape::new(vec![2, 2]));
        let b = Tensor::ones(Shape::new(vec![2, 2]));
        let c = a << b;
        let result = c.item();
        println!("{:?}", result);
    }

    #[test]
    fn three_dimension_matmul_test() {
        let a = Tensor::randn(Shape::new(vec![3, 2, 2]));
        let b = Tensor::randn(Shape::new(vec![2, 2]));
        let c = a << b;
        let result = c.item();
        println!("{:?}", result);
    }
}
