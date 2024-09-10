use crate::central::get_equation;
use crate::central::indexable::Indexable;
use crate::central::operation::Operation;
use crate::central::shape::Shape;
use crate::central::tensor::Tensor;
use crate::central::BackpropagationPacket;
use ndarray::prelude::*;
use ndarray::ArrayD;

use super::tensor::NAME_LENGTH;

/// Backward pass for the view operation
/// This function will take in a `BackpropagationPacket` and then set the gradients of the source tensor.
/// # Arguments
/// * `backprop_packet` - A `BackpropagationPacket` that contains the information needed to perform the backward pass.
///
/// # Panics
/// This function will panic if the operation in the `BackpropagationPacket` is not a view operation.
/// This function will panic if the number of dimensions of the source tensor is not 1 or 2.
/// This function will panic if the number of dimensions of the source tensor is greater than 2.
/// This function will panic if the number of dimensions of the view tensor is not 1 or 2.
pub fn backward(backprop_packet: BackpropagationPacket) {
    if let Operation::View(source_tensor, origin_index) = backprop_packet.operation {
        // All this should do is take the grad and put it in the right place
        let source_grad = backprop_packet.equation.get_tensor_grad(source_tensor);
        // View grad is not the same shape as source grad
        // so we want to allocate a zero tensor of the same shape as source grad
        // and then copy the view grad into the right place
        let source_shape = backprop_packet
            .equation
            .internal_tensor_store
            .get(&source_tensor)
            .unwrap()
            .shape;
        let mut new_view_grad = ArrayD::zeros(source_shape.as_ndarray_shape());
        // Indexable::Single and Indexable::Double are simple versions that let us use just usize to index
        // Indexable::FromTensor is a more complex version that lets us use a tensor to index
        match origin_index {
            Indexable::Single(i) => {
                let number_of_dimensions = new_view_grad.shape().len();
                if number_of_dimensions == 1 {
                    new_view_grad[[i]] = backprop_packet.grad[0];
                } else if number_of_dimensions == 2 {
                    new_view_grad
                        .slice_mut(s![i, ..])
                        .assign(&backprop_packet.grad.clone());
                } else {
                    panic!("Not implemented");
                }
            }
            Indexable::Double(i, j) => {
                new_view_grad[[i, j]] = backprop_packet.grad[0];
            }
            Indexable::Triple(i, j, k) => {
                new_view_grad[[i, j, k]] = backprop_packet.grad[0];
            }
            Indexable::FromTensor(tensor) => {
                // Get the indices from the tensor
                let indices = backprop_packet.equation.get_tensor_data(tensor);

                // This is a hack to get the shape of the source tensor
                // This is because the source tensor is not stored in the backprop packet
                // and so we need to get the shape of the source tensor from the equation
                let this_shape = source_grad.shape();
                let other_shape = indices.shape();
                let mut new_shape_dims = Vec::new();
                for i in 0..other_shape.len() {
                    new_shape_dims.push(other_shape[i]);
                }
                // HACK: this is to get this to work for tha 2 indexing 2 case
                new_shape_dims.push(this_shape[source_grad.ndim() - 1]);
                let new_shape = Shape::new(new_shape_dims);
                // REcreate the shape of the output

                // This is just running the operation in reverse
                for i in 0..new_shape.indices[0] {
                    for j in 0..new_shape.indices[1] {
                        for k in 0..new_shape.indices[2] {
                            new_view_grad[[indices[[i, j]] as usize, k]] =
                                backprop_packet.grad[[i, j, k]];
                        }
                    }
                }
            }
        }

        if backprop_packet.equation.advanced_logging {
            println!("New View Grad: {:?}", new_view_grad);
            println!("Source Grad: {:?}", source_grad);
        }
        backprop_packet
            .equation
            .set_tensor_grad(source_tensor, source_grad + new_view_grad);
    } else {
        panic!("Invalid operation type for backward pass");
    }
}

/// View operation for the tensor struct
/// This function will take in an indexable and then return a new tensor that is a view of the original tensor
/// # Arguments
/// * `index` - An indexable that will be used to create the view tensor.
/// # Returns
/// A new tensor that is a view of the original tensor.
impl Tensor {
    pub fn view(&self, index: Indexable) -> Tensor {
        // Allocate a new tensor the size of the view
        // and then set the data of the new tensor to the data of the old tensor
        let mut singleton = get_equation();
        let data: Vec<f32> = singleton.get_item(self.tensor_id);
        // I now need to get the index subset of data from the old tensor
        let new_shape = self.shape.subshape_from_indexable(index);

        match index {
            Indexable::Single(i) => {
                // If the number of indices is 1, then we can just take the data from the old tensor
                // and then allocate a new tensor with the data from the old tensor
                if self.shape.number_of_indices == 1 {
                    let data = data[i];
                    let tensor_id = singleton.allocate_element_tensor(
                        new_shape,
                        data,
                        Operation::View(self.tensor_id, index),
                    );
                    return Tensor {
                        tensor_id,
                        shape: new_shape,
                        operation: Operation::View(self.tensor_id, index),
                        name: ['a'; NAME_LENGTH],
                    };
                // If the number of indices is 2, then we need to take a slice of the data
                // and then allocate a new tensor with the data from the old tensor
                } else if self.shape.number_of_indices == 2 {
                    let offset = i * self.shape.indices[1];
                    let data = data[offset..offset + self.shape.indices[1]].to_vec();
                    let tensor_id = singleton.allocate_tensor_from_operation(
                        new_shape,
                        data,
                        Operation::View(self.tensor_id, index),
                    );
                    return Tensor {
                        tensor_id,
                        shape: new_shape,
                        operation: Operation::View(self.tensor_id, index),
                        name: ['a'; NAME_LENGTH],
                    };
                } else {
                    panic!("Indexing not supported for tensors with more than 2 dimensions");
                }
            }
            Indexable::Double(a, b) => {
                // If the number of indices is 1, then we can just take the data from the old tensor
                // and then allocate a new tensor with the data from the old tensor
                let offset = a * self.shape.indices[1] + b;
                let data = data[offset];
                let tensor_id = singleton.allocate_element_tensor(
                    new_shape,
                    data,
                    Operation::View(self.tensor_id, index),
                );
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a'; NAME_LENGTH],
                };
            }
            Indexable::Triple(a, b, c) => {
                // If the number of indices is 1, then we can just take the data from the old tensor
                // and then allocate a new tensor with the data from the old tensor
                let offset = a * self.shape.indices[1] * self.shape.indices[2]
                    + b * self.shape.indices[2]
                    + c;
                let data = data[offset];
                let tensor_id = singleton.allocate_element_tensor(
                    new_shape,
                    data,
                    Operation::View(self.tensor_id, index),
                );
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a'; NAME_LENGTH],
                };
            }
            Indexable::FromTensor(a) => {

                let indices = singleton.get_tensor_data(a);
                println!("Indices: {:?}", self.shape.clone().indices);
                println!("Shape: {:?}", indices.shape());

                let this_shape = self.shape.clone().indices;
                let other_shape = indices.shape();
                let mut new_shape_dims = Vec::new();
                for i in 0..other_shape.len() {
                    new_shape_dims.push(other_shape[i]);
                }
                // HACK: this is to get this to work for tha 2 indexing 2 case
                new_shape_dims.push(this_shape[self.shape.number_of_indices - 1]);
                let new_shape = Shape::new(new_shape_dims);

                assert!(indices.ndim() <= self.shape.number_of_indices);

                let data = singleton.get_item(self.tensor_id).clone();
                let data = data.as_slice();
                let data_as_array =
                    ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
                let mut return_tensor = ArrayD::<f32>::zeros(new_shape.as_ndarray_shape());

                let return_shape = return_tensor.shape().to_vec();

                println!("Return Shape: {:?}", return_shape);

                if return_shape.len() == 2 {
                    for i in 0..return_shape[0] {
                        for j in 0..return_shape[1] {
                            return_tensor[[i, j]] = data_as_array[[indices[[i]] as usize, j]];
                        }
                    }  
                }
                else if return_shape.len() == 3  {
                    for i in 0..return_shape[0] {
                        for j in 0..return_shape[1] {
                            for k in 0..return_shape[2] {
                                return_tensor[[i, j, k]] = data_as_array[[indices[[i, j]] as usize, k]];
                            }
                        }
                    }    
                }

                let tensor_id = singleton.allocate_tensor_from_operation(
                    new_shape.clone().into(),
                    return_tensor.into_raw_vec(),
                    Operation::View(self.tensor_id, index),
                );
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a'; NAME_LENGTH],
                };
            }
        }
    }
}
