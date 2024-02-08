use std::collections::{HashMap, HashSet};
use ndarray::prelude::*;

use crate::Tensor;

use super::{internal_tensor::{self, InternalTensor}, operation::{self, Operation}, shape::Shape, tensor::TensorID};

pub struct Equation {
    pub data_store: Vec<f32>,
    internal_tensor_store: HashMap<TensorID, InternalTensor>,
    value_count: usize,
}

impl Equation {
    pub fn new() -> Equation {
        Equation {
            data_store: Vec::new(),
            value_count: 0,
            internal_tensor_store: HashMap::new()
        }
    }

    pub fn allocate_tensor(&mut self, shape:Shape, data: Vec<f32>, operation: Operation) -> TensorID {
        if shape.total_size() != data.len() {
            println!("Shape Length: {}", shape.total_size());
            println!("Data Length: {}", data.len());
            panic!("Shape and data length mismatch");
        }
        let tensor_id = TensorID::new(self.value_count);
        
        // Techinally this could be made into a single operation, but I am lazy
        let data_start_index = self.data_store.len();
        self.data_store.append(&mut data.clone());
        let grad_start_index = self.data_store.len();
        self.data_store.append(&mut vec![0.0; shape.total_size()]);

        let internal_tensor = InternalTensor::new(tensor_id, shape, operation, data_start_index, grad_start_index);
        self.internal_tensor_store.insert(tensor_id, internal_tensor);
        self.value_count += 1;
        return tensor_id;
    }

    pub fn allocate_zero_tensor(&mut self, shape:Shape, operation: Operation) -> TensorID {
        let data = vec![0.0; shape.total_size()];
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_ones_tensor(&mut self, shape:Shape, operation: Operation) -> TensorID {
        let data = vec![1.0; shape.total_size()];
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_tensor_from_operation(&mut self, shape:Shape, data: Vec<f32>, operation: Operation) -> TensorID {
        self.allocate_tensor(shape, data, operation)
    }

    pub fn get_item(&self, tensor_id: TensorID) -> Vec<f32> {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        let data = self.data_store[internal_tensor.data_start_index..internal_tensor.data_start_index + internal_tensor.shape.total_size()].to_vec();
        data
    }

    fn topological_sort_util(
        &self,
        node: TensorID,
        visited: &mut HashSet<TensorID>,
        stack: &mut Vec<TensorID>,
    ) {
        visited.insert(node);

        // Assuming 'dependencies' method returns all the nodes that the current node depends on
        if let Some(dependencies) = self.internal_tensor_store.get(&node).map(|n| n.dependencies()) {
            for dep in dependencies {
                if !visited.contains(&dep) {
                    self.topological_sort_util(dep, visited, stack);
                }
            }
        }

        stack.push(node);
    }

    /// When called by a user with a provided value, preforms one step of backprogation
    /// BUT does not zero grad, or update the data 
    /// 'starting_value' - which value in the graph to back propogate from
    pub fn backward(&mut self, starting_value: Tensor) {
        // Initialize visited set and stack for topological sort
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        // Perform the topological sort
        self.topological_sort_util(starting_value.tensor_id, &mut visited, &mut stack);

        // Initialize gradients
        let ones = ArrayD::ones(self.internal_tensor_store.get(&starting_value.tensor_id).unwrap().get_grad().shape());
        self.internal_tensor_store.get_mut(&starting_value.tensor_id).unwrap().set_grad(ones);

        // Process nodes in topologically sorted order
        while let Some(node) = stack.pop() {
            let _ = self.backward_for_value(node); // Assuming this calculates and returns the children of the node
        }
    }

    fn backward_for_value(&mut self, node: TensorID) -> ArrayD<f32> {
        let internal_tensor = self.internal_tensor_store.get(&node).unwrap();
        let grad = self.get_data(internal_tensor.grad_start_index, internal_tensor.shape);
        let operation = internal_tensor.operation;

        match operation {
            Operation::Nop => {
                grad
            },
            Operation::Add(a, b) => {
                let a_grad = self.backward_for_value(a);
                let b_grad = self.backward_for_value(b);
                let a_grad = a_grad + grad;
                let b_grad = b_grad + grad;
                self.internal_tensor_store.get_mut(&a).unwrap().set_grad(a_grad);
                self.internal_tensor_store.get_mut(&b).unwrap().set_grad(b_grad);
                grad
            }
        }
    }

    fn get_data(&self, starting_point: usize, shape: Shape) -> ArrayD<f32> {
        let mut data = Vec::new();
        for i in 0..shape.total_size() {
            data.push(self.data_store[starting_point + i]);
        }
        ArrayD::from_shape_vec(shape.to_ndarray_shape(), data).unwrap()
    }

    fn set_data(&self, starting_point: usize, data: ArrayD<f32>) {
        // TODO: add an assert that insure that the shape of the data is the same as the shape of the tensor
        let data = data.into_raw_vec();
        for i in 0..data.len() {
            self.data_store[starting_point + i] = data[i];
        }
    }
}