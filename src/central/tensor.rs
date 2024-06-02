use core::slice;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Shl, Sub};
use ndarray::parallel::prelude::{IntoParallelRefIterator, ParallelIterator};
use ndarray::{prelude::*, Slice};
use serde::{Deserialize, Serialize};
use serde_json::Result;
use serde_json::Value;
use std::{fs, vec};
use std::path::Path;
use super::{operation::Operation, shape::Shape, indexable::Indexable};

use ndarray::{ArrayD, Axis};
use std::cmp::Ordering;
#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub tensor_id: TensorID,
    pub shape: Shape,
    pub operation: Operation,
    pub name: [char;10]
}

impl Tensor {

    pub fn randn(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_randn_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn load_from_weight_file<P: AsRef<Path>>(path: P) ->Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        
        // Load the file at path into a string, as it is a JSON file
        let file_content = fs::read_to_string(path).unwrap();
        
        // Parse the JSON file content
        let json: Value = serde_json::from_str(&file_content).unwrap();
        
        let data_array = json["data"].as_array().ok_or("Expected 'data' field to be an array").unwrap();
        let data: Vec<f32> = data_array
            .iter()
            .map(|x| x.as_f64().ok_or("Expected floating point numbers"))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.unwrap() as f32)
            .collect();

        let shape_aray = json["shape"].as_array().ok_or("Expected 'data' field to be an array").unwrap();
        let shape: Vec<usize> = shape_aray
            .iter()
            .map(|x| x.as_f64().ok_or("Expected floating point numbers"))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.unwrap() as usize)
            .collect();

        // Allocate tensor from the loaded data
        let tensor_id = singleton.allocate_tensor_from_operation(shape.clone().into(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape: shape.into(),
            operation: Operation::Nop,
            name: ['a'; 10], // Note: Adjust as necessary
        }
    }
 
    pub fn get_id(&self) -> usize {
        self.tensor_id.id
    }

    pub fn set_name(&mut self, name: [char;10]) {
        self.name = name;
    }

    pub fn zeroes(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_zero_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn ones(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_ones_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn element(shape: Shape, data: f32) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_element_tensor(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_tensor_from_operation(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn item(&self) -> ArrayD<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone();
        let data = data.as_slice();
        let data = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
        data
    }

    pub fn backward(&self) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.backward(self.tensor_id);
    }

    pub fn grad(&self) -> ArrayD<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_tensor_grad(self.tensor_id).clone();
        let data = data.as_slice().unwrap();
        let data = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
        data
    }

    pub fn exp(&self) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone().par_iter().map(|x| x.exp()).collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data, Operation::Exp(self.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Exp(self.tensor_id),
            name: ['a';10]
        }
    }


    pub fn pow(&self, power: f32) -> Tensor {
        let power_as_tensor = Tensor::element(vec![1].into(), power);
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone().par_iter().map(|x| x.powf(power)).collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data, Operation::Pow(self.tensor_id, power_as_tensor.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Pow(self.tensor_id, power_as_tensor.tensor_id),
            name: ['a';10]
        }
    }
    pub fn tanh(&self) -> Tensor {
        let hold = Tensor::element(Shape::new(vec![1]), 2.0);
        let k = *self * hold;
        let e = k.exp();
        let denom_invert = (e + 1.0).pow(-1.0);
        let o = (e - 1.0) * denom_invert;
        return o;
    }

    pub fn sum(&self, axis: usize) -> Tensor {

        // I want to sum along the first axis
        let data = self.item().sum_axis(Axis(axis)).clone().into_iter().map(|x| x).collect();
        
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

        let mut new_shape_indicies = self.shape.indices.clone();
        new_shape_indicies[axis] = 1;

        let tensor_id = singleton.allocate_tensor(Shape::new(vec![new_shape_indicies[0], new_shape_indicies[1]]), data, Operation::Sum(self.tensor_id, axis));

        Tensor {
            tensor_id,
            shape: Shape::new(vec![ self.shape.indices[0], new_shape_indicies[1]]),
            operation: Operation::Sum(self.tensor_id, axis),
            name: ['a';10]
        }
    }
    
    pub fn mean(&self, axis: usize) -> Tensor {

        let item = self.item();
        let sum = item.sum_axis(Axis(axis));
        let mean = sum / item.shape()[axis] as f32;

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = mean.into_iter().collect();
        let mut new_shape_indices = self.shape.indices.clone();
        new_shape_indices[axis] = 1;
        let tensor_id = singleton.allocate_tensor_from_operation(Shape::new(vec![new_shape_indices[0], new_shape_indices[1]]), data, Operation::Mean(self.tensor_id));

        Tensor {
            tensor_id,
            shape: Shape::new(vec![new_shape_indices[0], new_shape_indices[1]]),
            operation: Operation::Mean(self.tensor_id),
            name: ['a';10]
        }

    }

    pub fn t_mean(tensors: &Vec<Tensor>) -> Tensor {
        for tensor in tensors {
            if tensor.shape.size() != 1 {
                panic!("All tensors must be of size 1");
            }
        }

        let mut tensor = Tensor::element(vec![1].into(), 0.0);
        for sum_tensor in tensors {
            tensor = tensor + *sum_tensor;
        }

        return tensor / tensors.len() as f32;
    }

    pub fn multi_concat(tensors: &Vec<Tensor>) -> Tensor {
        let mut tensor = tensors[0];
        for sum_tensor in tensors.iter().skip(1) {
            tensor = tensor.concat(sum_tensor.clone());
        }


        return tensor;
    }

    pub fn log(&self) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone().par_iter().map(|x| x.ln()).collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data, Operation::Log(self.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Log(self.tensor_id),
            name: ['a';10]
        }
    }

    pub fn set_index(&mut self, indexable: Indexable, data: Vec<f32>) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.set_subset_of_tensor_data(self.tensor_id, indexable, data);
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.set_requires_grad(self.tensor_id, requires_grad);
    }

    pub fn broadcast(&self, shape: Shape) -> Tensor {

        let data: Vec<f32> = self.item().broadcast(shape.as_ndarray_shape()).unwrap().iter().map(|x|*x).collect();

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        // We need to dubpilcate the data to match the new shape

        let tensor_id = singleton.allocate_tensor_from_operation(shape.clone(), data, Operation::Broadcast(self.tensor_id, shape));

        Tensor {
            tensor_id,
            shape,
            operation: Operation::Broadcast(self.tensor_id, shape),
            name: ['a';10]
        }
    }

    pub fn max(&self, axis: usize) -> Tensor {
        let mut shape = self.shape.clone().indices;
        shape[axis] = 1;
        // Find the max value along the second axis (axis 1)
        let max_values = self.item().map_axis(Axis(axis), |row| *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
        // Reshape the result to [32, 1]

        let result = max_values.into_shape((shape[0], shape[1])).unwrap();

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        // HACK: this should be generic to any sized shape, but I am not sure how to do that
        let new_shape = Shape::new(vec![shape[0], shape[1]]);
        println!(   "{:?}", new_shape);
        let tensor_id = singleton.allocate_tensor_from_operation(new_shape, result.iter().map(|x| *x).collect(), Operation::Nop);

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn view(&self, index: Indexable) -> Tensor {
        // Allocate a new tensor the size of the view
        // and then set the data of the new tensor to the data of the old tensor
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data : Vec<f32> = singleton.get_item(self.tensor_id);
        // I now need to get the index subset of data from the old tensor
        let new_shape = self.shape.subshape_from_indexable(index);

        match index {
            Indexable::Single(i) => {
                if self.shape.number_of_indices == 1 {
                    let data = data[i];
                    let tensor_id = singleton.allocate_element_tensor(new_shape, data, Operation::View(self.tensor_id, index));
                    return Tensor {
                        tensor_id,
                        shape: new_shape,
                        operation: Operation::View(self.tensor_id, index),
                        name: ['a';10]
                        }
                    }
                    else if self.shape.number_of_indices == 2 {
                        let offset = i * self.shape.indices[1];
                        let data = data[offset..offset + self.shape.indices[1]].to_vec();
                        let tensor_id = singleton.allocate_tensor_from_operation(new_shape, data, Operation::View(self.tensor_id, index));
                        return Tensor {
                            tensor_id,
                            shape: new_shape,
                            operation: Operation::View(self.tensor_id, index),
                            name: ['a';10]
                        }
                    }
                    else {
                        panic!("Indexing not supported for tensors with more than 2 dimensions");
                    }                                
            },
            Indexable::Double(a, b) => {
                let offset = a * self.shape.indices[1] + b;
                let data = data[offset];
                let tensor_id = singleton.allocate_element_tensor(new_shape, data, Operation::View(self.tensor_id, index));
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a';10]
                }
            },
            Indexable::Mixed(a, b) => {
                // Look up the A and B vectors, and then use the B vector to pick the indices from the A vector
                let a_data = singleton.get_item(a);
                let b_data = singleton.get_item(b);
                let mut new_data = Vec::new();
                for i in 0..b_data.len() {
                    let index = b_data[i] as usize;
                    new_data.push(a_data[index]);
                }
                let tensor_id = singleton.allocate_tensor_from_operation(new_shape, new_data, Operation::View(self.tensor_id, index));
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a';10]
                }
            },
            Indexable::FromTensor(a) => {
                let indices = singleton.get_tensor_data(a);

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
                let data_as_array = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
                let mut return_tensor = ArrayD::<f32>::zeros(new_shape.as_ndarray_shape());

                let return_shape = return_tensor.shape().to_vec();

                for i in 0..return_shape[0] {
                    for j in 0..return_shape[1] {
                        for k in 0..return_shape[2] {
                            return_tensor[[i, j, k]] = data_as_array[[indices[[i, j]] as usize, k]];
                        }
                    }
                }


                
                let tensor_id = singleton.allocate_tensor_from_operation(new_shape.clone().into(), return_tensor.into_raw_vec(), Operation::View(self.tensor_id, index));
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a';10]
                }
            }

        }
    }

    pub fn reshape(&self, new_shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone();
        let tensor_id = singleton.allocate_tensor_from_operation(new_shape.clone(), data, Operation::Reshape(self.tensor_id, new_shape));

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Reshape(self.tensor_id, new_shape),
            name: ['a';10]
        }

    }

    
    pub fn concat(&self, other: Tensor) -> Tensor {
        if other.shape.number_of_indices != 1 {
            panic!("The tensor to concat must be a 1D tensor");
        }

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone();
        let other_data = singleton.get_item(other.tensor_id).clone();
        let mut new_data = Vec::new();
        for i in 0..data.len() {
            new_data.push(data[i]);
        }
        for i in 0..other_data.len() {
            new_data.push(other_data[i]);
        }
        let new_shape = Shape::new(vec![data.len() + other_data.len()]);
        let tensor_id = singleton.allocate_tensor_from_operation(new_shape.clone(), new_data, Operation::Concat(self.tensor_id, other.tensor_id));

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Concat(self.tensor_id, other.tensor_id),
            name: ['a';10]
        }

        
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let left_hand_item = self.item();
        let right_hand_item = rhs.item();
        if left_hand_item.shape() != right_hand_item.shape() {
            let right_hand_broadcasted = rhs.broadcast(self.shape);
            let result_data = self.item() + right_hand_broadcasted.item();

            let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Add(self.tensor_id, right_hand_broadcasted.tensor_id));

            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Add(self.tensor_id, right_hand_broadcasted.tensor_id),
                name: ['a';10]
            }
        }
        else {

            let result_data = self.item() + rhs.item();

            let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
    
            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Add(self.tensor_id, rhs.tensor_id));
    
            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Add(self.tensor_id, rhs.tensor_id),
                name: ['a';10]
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

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            // we need to broadcast the tensors
            let broaded_casted_rhs = rhs.broadcast(self.shape);
            let result_data = self.item() * broaded_casted_rhs.item();

            let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
    
            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Mul(self.tensor_id, broaded_casted_rhs.tensor_id));
    
            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Mul(self.tensor_id, broaded_casted_rhs.tensor_id),
                name: ['a';10]
            }
        }
        else {
            let result_data = self.item() * rhs.item();

            let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
    
            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Mul(self.tensor_id, rhs.tensor_id));
    
            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Mul(self.tensor_id, rhs.tensor_id),
                name: ['a';10]
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

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}


impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {

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

// SIN: reusing the Shl opeartor to do the matmul operations
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {

        

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

        let result_data = singleton.matmul(self.tensor_id, self.shape, rhs.tensor_id, rhs.shape);
        let resultant_shape = self.shape.matmul_shape(&rhs.shape);
        let tensor_id = singleton.allocate_tensor_from_operation(resultant_shape, result_data.into_raw_vec(), Operation::MatMul(self.tensor_id, rhs.tensor_id));
        let matmul_shape = self.shape.matmul_shape(&rhs.shape);
        
        Tensor {
            tensor_id,
            shape: matmul_shape,
            operation: Operation::MatMul(self.tensor_id, rhs.tensor_id),
            name: ['a';10]
        }
    }
}
