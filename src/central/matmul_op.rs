use crate::central::operation::Operation;
use crate::central::tensor::Tensor;
use crate::central::get_equation;
use crate::central::BackpropagationPacket;
use std::ops::Shl;
use ndarray::prelude::*;

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



// SIN: reusing the Shl opeartor to do the matmul operations
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {
        let mut singleton = get_equation();

        let result_data = singleton.matmul(self.tensor_id, self.shape, rhs.tensor_id, rhs.shape);
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
            name: ['a'; 10],
        }
    }
}