use super::{indexable::Indexable, operation::Operation, shape::Shape};
use crate::central::get_equation;
use ndarray::parallel::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde_json::Value;
use std::path::Path;
use std::{fs, vec};
use std::io::{Read, Result};
use std::convert::TryInto;
use std::io::{self};
use std::mem::size_of;
use log::info;

use ndarray::{ArrayD, Axis};
#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash, Debug)]
pub struct TensorID {
    pub id: usize,
}

impl TensorID {
    pub fn new(id: usize) -> TensorID {
        TensorID { id }
    }
}

pub const NAME_LENGTH: usize = 20;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub tensor_id: TensorID,
    pub shape: Shape,
    pub operation: Operation,
    pub name: [char; NAME_LENGTH],
}

pub fn name_from_string(name: &str) -> [char;NAME_LENGTH] {
    assert!(name.len() <= NAME_LENGTH);    
    let mut name_array = ['a'; NAME_LENGTH];
    for (i, c) in name.chars().enumerate() {
        name_array[i] = c;
    }
    name_array

} 

impl Tensor {
    pub fn arange(start: usize, stop: usize, step: usize) -> Tensor {
        let mut data = Vec::new();
        for i in (start..stop).step_by(step) {
            data.push(i as f32);
        }
        let data_length = data.len();
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_tensor_from_operation(
            Shape::new(vec![data.len()]),
            data,
            Operation::Nop,
        );
        Tensor {
            tensor_id,
            shape: Shape::new(vec![data_length]),
            operation: Operation::Nop,
            name: name_from_string("arange"),
        }
    }

    pub fn tril(shape: Shape) -> Tensor {
        assert!(shape.number_of_indices == 2);
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_tril_tensor(shape.clone());
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: name_from_string("tril"),
        }
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor {
        let mut singleton = get_equation();
        let data = singleton.get_item(self.tensor_id).clone();
        let mask_data = singleton.get_item(mask.tensor_id).clone();
        let mut new_data = Vec::new();
        for i in 0..data.len() {
            if mask_data[i] == 0.0 {
                new_data.push(value);
            } else {
                new_data.push(data[i]);
            }
        }
        let tensor_id = singleton.allocate_tensor_from_operation(
            self.shape.clone(),
            new_data,
            Operation::MaskedFill(self.tensor_id, mask.tensor_id, value as isize),
        );

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::MaskedFill(self.tensor_id, mask.tensor_id, value as isize),
            name: name_from_string("masked_fill"),
        }
    }

    pub fn randn(shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_randn_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: name_from_string("randn"),
        }
    }

    pub fn idenitity(shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_idenitity_tensor(shape.clone());
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: name_from_string("idenitity"),
        }
    }

    pub fn load_from_weight_file<P: AsRef<Path>>(path: P) -> Tensor {
        let mut singleton = get_equation();

        // Load the file at path into a string, as it is a JSON file
        let file_content = fs::read_to_string(path).unwrap();

        // Parse the JSON file content
        let json: Value = serde_json::from_str(&file_content).unwrap();

        let data_array = json["data"]
            .as_array()
            .ok_or("Expected 'data' field to be an array")
            .unwrap();
        let data: Vec<f32> = data_array
            .iter()
            .map(|x| x.as_f64().ok_or("Expected floating point numbers"))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.unwrap() as f32)
            .collect();

        let shape_aray = json["shape"]
            .as_array()
            .ok_or("Expected 'data' field to be an array")
            .unwrap();
        let shape: Vec<usize> = shape_aray
            .iter()
            .map(|x| x.as_f64().ok_or("Expected floating point numbers"))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.unwrap() as usize)
            .collect();

        // Allocate tensor from the loaded data
        let tensor_id =
            singleton.allocate_tensor_from_operation(shape.clone().into(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape: shape.into(),
            operation: Operation::Nop,
            name: ['a'; NAME_LENGTH], // Note: Adjust as necessary
        }
    }

    pub fn get_id(&self) -> usize {
        self.tensor_id.id
    }

    pub fn set_name(&mut self, name: [char; NAME_LENGTH]) {
        self.name = name;
    }

    pub fn zeroes(shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_zero_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn ones(shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_ones_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn element(shape: Shape, data: f32) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_element_tensor(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; NAME_LENGTH],
        }
    }


    pub fn from_bytestream<R: Read>(reader: &mut R, reverse_dimensions: bool) -> io::Result<Self> {
        // Read the length of the shape (number of dimensions)
        let mut shape_len_buf = [0u8; 4];
        reader.read_exact(&mut shape_len_buf)?;
        let shape_len = i32::from_le_bytes(shape_len_buf) as usize;
    
        // Read the shape dimensions in a single read
        let mut shape_buf = vec![0u8; shape_len * size_of::<i32>()];
        reader.read_exact(&mut shape_buf)?;

        let shape : Vec<usize> = shape_buf
            .chunks(size_of::<i32>())
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as usize).collect();;
            // Reverse the shape to match the order of the dimensions
            // since that is what pytorch uses

        let shape = if reverse_dimensions {
            shape.into_iter().rev().collect()
        } else {
            shape
        };

    
        // Calculate the number of elements in the tensor
        let num_elements: usize = shape.iter().product();

        info!("Reading in a tensor with shape: {:?}", shape);
        info!("Reading in a tensor with num_elements: {:?}", num_elements);
    
        // Read all tensor values in a single read
        let mut data_buf = vec![0u8; num_elements * size_of::<f32>()];
        reader.read_exact(&mut data_buf)?;
    
        // Convert the read bytes to f32 values
        let data: Vec<f32> = data_buf
            .chunks(size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
    
        Ok(Tensor::from_vec(data, shape.into()))
    }


    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id =
            singleton.allocate_tensor_from_operation(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn item(&self) -> ArrayD<f32> {
        let singleton = get_equation();
        let data = singleton.get_item(self.tensor_id).clone();
        let data = data.as_slice();
        let data = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
        data
    }

    pub fn backward(&self) {
        let mut singleton = get_equation();
        singleton.backward(self.tensor_id);
    }

    pub fn grad(&self) -> ArrayD<f32> {
        let singleton = get_equation();
        let data = singleton.get_tensor_grad(self.tensor_id).clone();
        let data = data.as_slice().unwrap();
        let data = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
        data
    }

    pub fn exp(&self) -> Tensor {
        let mut singleton = get_equation();
        let data = singleton
            .get_item(self.tensor_id)
            .clone()
            .par_iter()
            .map(|x| x.exp())
            .collect();
        let tensor_id = singleton.allocate_tensor_from_operation(
            self.shape.clone(),
            data,
            Operation::Exp(self.tensor_id),
        );

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Exp(self.tensor_id),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn pow(&self, power: f32) -> Tensor {
        let power_as_tensor = Tensor::element(vec![1].into(), power);
        let mut singleton = get_equation();
        let data = singleton
            .get_item(self.tensor_id)
            .clone()
            .par_iter()
            .map(|x| x.powf(power))
            .collect();
        let tensor_id = singleton.allocate_tensor_from_operation(
            self.shape.clone(),
            data,
            Operation::Pow(self.tensor_id, power_as_tensor.tensor_id),
        );

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Pow(self.tensor_id, power_as_tensor.tensor_id),
            name: ['a'; NAME_LENGTH],
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

    pub fn sin(&self) -> Tensor {
        let mut singleton = get_equation();
        let data = singleton
            .get_item(self.tensor_id)
            .clone()
            .par_iter()
            .map(|x| x.sin())
            .collect();
        let tensor_id = singleton.allocate_tensor_from_operation(
            self.shape.clone(),
            data,
            Operation::Sin(self.tensor_id),
        );

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Sin(self.tensor_id),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn cos(&self) -> Tensor {
        let mut singleton = get_equation();
        let data = singleton
            .get_item(self.tensor_id)
            .clone()
            .par_iter()
            .map(|x| x.cos())
            .collect();
        let tensor_id = singleton.allocate_tensor_from_operation(
            self.shape.clone(),
            data,
            Operation::Cos(self.tensor_id),
        );

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Cos(self.tensor_id),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn tanh_mapped(&self) -> Tensor {
        let mut singleton = get_equation();

        let data = singleton
            .get_item(self.tensor_id)
            .clone()
            .par_iter()
            .map(|x| x.tanh())
            .collect();

        let tensor_id = singleton.allocate_tensor_from_operation(
            self.shape.clone(),
            data,
            Operation::Tanh(self.tensor_id),
        );

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Tanh(self.tensor_id),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn sum(&self, axis: usize) -> Tensor {
        // I want to sum along the first axis
        let data = self
            .item()
            .sum_axis(Axis(axis))
            .clone()
            .into_iter()
            .map(|x| x)
            .collect();

        let mut singleton = get_equation();

        let mut new_shape_indices = vec![];

        for i in 0..self.shape.number_of_indices {
            if i != axis {
                new_shape_indices.push(self.shape.indices[i]);
            }
            else if i == axis {
                new_shape_indices.push(1);
            }
        }


        let tensor_id = singleton.allocate_tensor(
            Shape::new(new_shape_indices.clone()),
            data,
            Operation::Sum(self.tensor_id, axis),
        );

        Tensor {
            tensor_id,
            shape: Shape::new(new_shape_indices),
            operation: Operation::Sum(self.tensor_id, axis),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn mean(&self, axes: Vec<usize>) -> Tensor {
        if axes.len() == 1 {
            let item = self.item();
            let sum = item.sum_axis(Axis(axes[0]));
            let mean = sum / item.shape()[axes[0]] as f32;

            let mut singleton = get_equation();
            let data : Vec<f32> = mean.into_iter().collect();
            let mut new_shape_indices = self.shape.indices.clone();
            new_shape_indices[axes[0]] = 1;
            info!("new_shape_indices: {:?}", new_shape_indices);
            info!("data length: {:?}", data.len());
            let mut new_shape_indices_vec = new_shape_indices.to_vec();
            // Cut down new_shape_indices_vec to the correct size
            // which will be the same as the original shape
            new_shape_indices_vec.truncate(self.shape.number_of_indices);
            info!("new_shape_indices_vec: {:?}", new_shape_indices_vec);
            let tensor_id = singleton.allocate_tensor_from_operation(
                Shape::new(new_shape_indices_vec.clone()),
                data,
                Operation::Mean(self.tensor_id),
            );

            return Tensor {
                tensor_id,
                shape: Shape::new(new_shape_indices_vec),
                operation: Operation::Mean(self.tensor_id),
                name: ['a'; NAME_LENGTH],
            };
        }
        if axes.len() == 2 {
            let item = self.item();
            let sum = item.sum_axis(Axis(axes[0])).sum_axis(Axis(axes[1] - 1));
            let mean = sum / (item.shape()[axes[0]] * item.shape()[axes[1]]) as f32;

            let mut singleton = get_equation();

            let data = mean.into_iter().collect();

            let mut new_shape_indices = self.shape.indices.clone();
            new_shape_indices[axes[0]] = 1;
            new_shape_indices[axes[1]] = 1;

            let tensor_id = singleton.allocate_tensor_from_operation(
                Shape::new(vec![
                    new_shape_indices[0],
                    new_shape_indices[1],
                    new_shape_indices[2],
                ]),
                data,
                Operation::Mean(self.tensor_id),
            );

            return Tensor {
                tensor_id,
                shape: Shape::new(vec![
                    new_shape_indices[0],
                    new_shape_indices[1],
                    new_shape_indices[2],
                ]),
                operation: Operation::Mean(self.tensor_id),
                name: ['a'; NAME_LENGTH],
            };
        } else {
            panic!("Mean only supports 1 or 2 axes")
        }
    }

    pub fn std(&self, axis: Vec<usize>) -> Tensor {
        let mean = self.mean(axis.clone());
        let mean_broadcast = mean.broadcast(self.shape.clone());
        let diff = *self - mean_broadcast;
        let diff_squared = diff.pow(2.0);
        let variance = diff_squared.mean(axis);
        let std = variance.pow(0.5);
        return std;
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
        let mut singleton = get_equation();
        let data = singleton
            .get_item(self.tensor_id)
            .clone()
            .par_iter()
            .map(|x| x.ln())
            .collect();
        let tensor_id = singleton.allocate_tensor_from_operation(
            self.shape.clone(),
            data,
            Operation::Log(self.tensor_id),
        );

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Log(self.tensor_id),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn set_index(&mut self, indexable: Indexable, data: Vec<f32>) {
        let mut singleton = get_equation();
        singleton.set_subset_of_tensor_data(self.tensor_id, indexable, data);
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        let mut singleton = get_equation();
        singleton.set_requires_grad(self.tensor_id, requires_grad);
    }

    pub fn broadcast(&self, shape: Shape) -> Tensor {
        let data: Vec<f32> = self
            .item()
            .broadcast(shape.as_ndarray_shape())            
            .unwrap()
            .iter()
            .map(|x| *x)
            .collect();

        let mut singleton = get_equation();
        // We need to dubpilcate the data to match the new shape

        let tensor_id = singleton.allocate_tensor_from_operation(
            shape.clone(),
            data,
            Operation::Broadcast(self.tensor_id, shape),
        );

        Tensor {
            tensor_id,
            shape,
            operation: Operation::Broadcast(self.tensor_id, shape),
            name: ['a'; NAME_LENGTH],
        }
    }

    fn swap_axes(data: Vec<f32>, shape: &Vec<usize>, axis1: usize, axis2: usize) -> Vec<f32> {

        // Swap the axes in the shape to get the new shape
        let mut new_shape = shape.to_vec();
        new_shape.swap(axis1, axis2);
    
        // Compute strides for the input and output shapes
        let input_strides = Tensor::compute_strides(shape);
        let output_strides = Tensor::compute_strides(&new_shape);
    
        // Prepare the output data vector
        let mut output_data = vec![0.0; data.len()];
    
        // Iterate over each index in the input data
        for idx in 0..data.len() {
            // Convert linear index to multi-dimensional index for input
            let mut multi_idx = vec![0; shape.len()];
            let mut remaining = idx;
            for (i, &stride) in input_strides.iter().enumerate() {
                multi_idx[i] = remaining / stride;
                remaining %= stride;
            }
    
            // Swap the axes in the multi-dimensional index
            multi_idx.swap(axis1, axis2);
    
            // Convert the multi-dimensional index back to a linear index for output
            let mut out_idx = 0;
            for (i, &val) in multi_idx.iter().enumerate() {
                out_idx += val * output_strides[i];
            }
    
            // Assign the value to the output data vector
            output_data[out_idx] = data[idx];
        }

        output_data
    }
    
    // Helper function to compute strides for a given shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let n = shape.len();
        let mut strides = vec![0; n];
        strides[n - 1] = 1;
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn transpose_inside(matrix: Vec<f32>, rows: usize, cols: usize) -> Vec<f32> {
        let mut transposed = vec![0.0; rows * cols];
        
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = matrix[r * cols + c];
            }
        }
        
        transposed
    }

    // Helper function for the most common use case
    pub fn transpose(&self) -> Tensor {
        return self.tranpose_with_provided_axis(0, 1);
    }

    /// this will swap the indices at the procived indices
    pub fn tranpose_with_provided_axis(&self, first_index: usize, second_index: usize) -> Tensor {
        let mut singleton = get_equation();
        let data = singleton.get_item(self.tensor_id).clone();

        let mut indices = vec![];
        let current_shape = self.shape;
        for i in 0..self.shape.number_of_indices {

            indices.push(current_shape.indices[i]);
        }
        let data = Tensor::swap_axes(data, &indices, first_index, second_index);
        // create a new shape with the indices swapped
        let mut new_shape_indices = self.shape.indices.clone();
        new_shape_indices.swap(first_index, second_index);
        let mut new_shape_indices = new_shape_indices.to_vec();
        new_shape_indices.truncate(self.shape.number_of_indices);
        let new_shape = Shape::new(new_shape_indices);

        let tensor_id = singleton.allocate_tensor_from_operation(
            new_shape.clone(),
            data,
            Operation::Transpose(self.tensor_id, first_index, second_index),
        );

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Transpose(self.tensor_id, first_index, second_index),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn max(&self, axis: usize) -> Tensor {
        let mut shape = self.shape.clone().indices;
        let number_of_indices = self.shape.number_of_indices;

        shape[axis] = 1;
        // Find the max value along the second axis (axis 1)
        let max_values = self.item().map_axis(Axis(axis), |row| {
            *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        });
        // Reshape the result to [32, 1]
        let mut new_shape = vec![];
        for i in 0..number_of_indices {
            if i != axis {
                new_shape.push(shape[i]);
            } else {
                new_shape.push(1);
            }
        }
        println!("new_shape: {:?}", new_shape);
        let result = max_values.into_shape(new_shape.clone()).unwrap();
        println!("result: {:?}", result.shape());
        let mut singleton = get_equation();
        // HACK: this should be generic to any sized shape, but I am not sure how to do that
        let new_shape = Shape::new(new_shape);
        println!("new_shape: {:?}", new_shape);
        info!("new_sssshape: {:?}", new_shape);
        let tensor_id = singleton.allocate_tensor_from_operation(
            new_shape,
            result.iter().map(|x| *x).collect(),
            Operation::Nop,
        );

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Nop,
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn reshape(&self, new_shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        // TOOD: this does not need to be a data copy,
        // it would be nice if we could simply copy the tensor_id, and the new shape
        let data = singleton.get_item(self.tensor_id).clone();
        let tensor_id = singleton.allocate_tensor_from_operation(
            new_shape.clone(),
            data,
            Operation::Reshape(self.tensor_id, new_shape),
        );

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Reshape(self.tensor_id, new_shape),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn squeeze(&self, axis: usize) -> Tensor {
        // This will just be a wrapper around reshape function
        self.reshape(self.shape.remove(axis))
    }

    pub fn concat(&self, other: Tensor) -> Tensor {
        if other.shape.number_of_indices != 1 {
            panic!("The tensor to concat must be a 1D tensor");
        }

        let mut singleton = get_equation();
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
        let tensor_id = singleton.allocate_tensor_from_operation(
            new_shape.clone(),
            new_data,
            Operation::Concat(self.tensor_id, other.tensor_id),
        );

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Concat(self.tensor_id, other.tensor_id),
            name: ['a'; NAME_LENGTH],
        }
    }

    pub fn softmax(&self, axis: usize) -> Tensor {
        let max = self.max(axis);
        let counts = (*self - max).exp();
        let sum = counts.sum(axis);
        let sum_inverted = sum.pow(-1.0);
        let softmax = counts * sum_inverted;
        softmax
    }

    pub fn cross_entropy_loss(&self, trues: Tensor) -> Tensor {
        let softmax = self.softmax(0);
        let log_softmax = softmax.log();
        let loss = trues * log_softmax;
        let sum = loss.sum(0);
        let mean = sum.mean(vec![0]);
        mean
    }
}
