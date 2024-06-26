use super::{indexable::Indexable, operation::Operation, shape::Shape};
use crate::central::get_equation;
use ndarray::parallel::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde_json::Value;
use std::path::Path;
use std::{fs, vec};

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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub tensor_id: TensorID,
    pub shape: Shape,
    pub operation: Operation,
    pub name: [char; 10],
}

impl Tensor {
    pub fn randn(shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_randn_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; 10],
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
            name: ['a'; 10], // Note: Adjust as necessary
        }
    }

    pub fn get_id(&self) -> usize {
        self.tensor_id.id
    }

    pub fn set_name(&mut self, name: [char; 10]) {
        self.name = name;
    }

    pub fn zeroes(shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_zero_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; 10],
        }
    }

    pub fn tril(x: usize, y: usize) -> Tensor {
        let mut array = vec![0.0; x * y];
        for i in 0..x {
            for j in 0..y {
                if i >= j {
                    array[i * y + j] = 1.0;
                }
            }
        }
        let mut singleton = get_equation();
        let tensor_id =
            singleton.allocate_tensor_from_operation(Shape::new(vec![x, y]), array, Operation::Nop);

        Tensor {
            tensor_id,
            shape: Shape::new(vec![x, y]),
            operation: Operation::Nop,
            name: ['a'; 10],
        }
    }

    pub fn ones(shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_ones_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; 10],
        }
    }

    pub fn element(shape: Shape, data: f32) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id = singleton.allocate_element_tensor(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; 10],
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Tensor {
        let mut singleton = get_equation();
        let tensor_id =
            singleton.allocate_tensor_from_operation(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a'; 10],
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
            name: ['a'; 10],
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
            name: ['a'; 10],
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
        let data = self
            .item()
            .sum_axis(Axis(axis))
            .clone()
            .into_iter()
            .map(|x| x)
            .collect();

        let mut singleton = get_equation();

        let mut new_shape_indicies = self.shape.indices.clone();
        new_shape_indicies[axis] = 1;

        let tensor_id = singleton.allocate_tensor(
            Shape::new(vec![new_shape_indicies[0], new_shape_indicies[1]]),
            data,
            Operation::Sum(self.tensor_id, axis),
        );

        Tensor {
            tensor_id,
            shape: Shape::new(vec![self.shape.indices[0], new_shape_indicies[1]]),
            operation: Operation::Sum(self.tensor_id, axis),
            name: ['a'; 10],
        }
    }

    pub fn mean(&self, axes: Vec<usize>) -> Tensor {
        if axes.len() == 1 {
            let item = self.item();
            let sum = item.sum_axis(Axis(axes[0]));
            let mean = sum / item.shape()[axes[0]] as f32;

            let mut singleton = get_equation();
            let data = mean.into_iter().collect();
            let mut new_shape_indices = self.shape.indices.clone();
            new_shape_indices[axes[0]] = 1;
            let tensor_id = singleton.allocate_tensor_from_operation(
                Shape::new(vec![new_shape_indices[0], new_shape_indices[1]]),
                data,
                Operation::Mean(self.tensor_id),
            );

            return Tensor {
                tensor_id,
                shape: Shape::new(vec![new_shape_indices[0], new_shape_indices[1]]),
                operation: Operation::Mean(self.tensor_id),
                name: ['a'; 10],
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
                name: ['a'; 10],
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
            name: ['a'; 10],
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
            name: ['a'; 10],
        }
    }

    pub fn max(&self, axis: usize) -> Tensor {
        let mut shape = self.shape.clone().indices;
        shape[axis] = 1;
        // Find the max value along the second axis (axis 1)
        let max_values = self.item().map_axis(Axis(axis), |row| {
            *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        });
        // Reshape the result to [32, 1]

        let result = max_values.into_shape((shape[0], shape[1])).unwrap();

        let mut singleton = get_equation();
        // HACK: this should be generic to any sized shape, but I am not sure how to do that
        let new_shape = Shape::new(vec![shape[0], shape[1]]);
        let tensor_id = singleton.allocate_tensor_from_operation(
            new_shape,
            result.iter().map(|x| *x).collect(),
            Operation::Nop,
        );

        Tensor {
            tensor_id,
            shape: new_shape,
            operation: Operation::Nop,
            name: ['a'; 10],
        }
    }

    pub fn reshape(&self, new_shape: Shape) -> Tensor {
        let mut singleton = get_equation();
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
            name: ['a'; 10],
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
            name: ['a'; 10],
        }
    }

    pub fn softmax(&self) -> Tensor {
        let max = self.max(1);
        let counts = (*self - max).exp();
        let sum = counts.sum(1);
        let sum_inverted = sum.pow(-1.0);
        let softmax = counts * sum_inverted;

        softmax
    }

    pub fn cross_entropy_loss(&self, trues: Tensor) -> Tensor {
        let softmax = self.softmax();
        let log_softmax = softmax.log();
        let loss = trues * log_softmax;
        let sum = loss.sum(1);
        let mean = sum.mean(vec![0]);
        mean
    }
}
