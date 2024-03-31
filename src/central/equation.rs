use std::{collections::{HashMap, HashSet}, ops::{Index, Mul}};
use ndarray::prelude::*;
use rand::Rng;

use super::{internal_tensor::InternalTensor, operation::Operation, shape::Shape, tensor::TensorID, indexable::Indexable};

pub struct Equation {
    pub data_store: Vec<f32>,
    internal_tensor_store: HashMap<TensorID, InternalTensor>,
    value_count: usize,
    advanced_logging: bool
}

impl Equation {
    pub fn new() -> Equation {
        Equation {
            data_store: Vec::new(),
            value_count: 0,
            internal_tensor_store: HashMap::new(),
            advanced_logging: false
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

    pub fn allocate_randn_tensor(&mut self, shape:Shape, operation: Operation) -> TensorID {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..shape.total_size()).map(|_| rng.gen()).collect();
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_element_tensor(&mut self, shape:Shape, data: f32, operation: Operation) -> TensorID {
        let data = vec![data; shape.total_size()];
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_tensor_from_operation(&mut self, shape:Shape, data: Vec<f32>, operation: Operation) -> TensorID {
        self.allocate_tensor(shape, data, operation)
    }

    pub fn matmul(&mut self, a: TensorID, b: TensorID) -> ArrayD<f32> {
        // preforms the matmul, returning the just the result datam does note allocate a new tensor
        let a_data = self.get_tensor_data(a);
        let b_data = self.get_tensor_data(b);
        // we need to into_dimensionality() to convert the 1D array to a 2D array
        let a_data = a_data.into_dimensionality::<Ix2>().unwrap();
        let b_data = b_data.into_dimensionality::<Ix2>().unwrap();
        let result = a_data.dot(&b_data);
        result.into_dyn()
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
    pub fn backward(&mut self, starting_value: TensorID) {
        // Initialize visited set and stack for topological sort
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        // Perform the topological sort
        self.topological_sort_util(starting_value, &mut visited, &mut stack);

        // Initialize gradients
        let ones = ArrayD::ones(self.internal_tensor_store.get(&starting_value).unwrap().shape.as_ndarray_shape());
        self.set_tensor_grad(starting_value, ones.clone());

        // Process nodes in topologically sorted order
        while let Some(node) = stack.pop() {
            let _ = self.backward_for_value(node); // Assuming this calculates and returns the children of the node
        }
    } 

    fn backward_for_value(&mut self, node: TensorID)  {
        let internal_tensor = self.internal_tensor_store.get(&node).unwrap();
        let data = self.get_tensor_data(node);
        let grad = self.get_tensor_grad(node);
        if self.advanced_logging == true {
            println!("Node: {:?}", node);
            println!("Data: {:?}", data);
            println!("Grad: {:?}", grad);
        }

        let operation = internal_tensor.operation.clone();
        match operation {
            Operation::Nop => {
                // Do nothing
            },
            Operation::Add(a, b) => {
                let left_hand_grad = self.get_tensor_grad(a);
                let right_hand_grad = self.get_tensor_grad(b);
                self.set_tensor_grad(a, left_hand_grad + grad.clone());
                self.set_tensor_grad(b, right_hand_grad + grad);
            },
            Operation::Mul(a, b) => {
                if self.advanced_logging == true {
                    println!("A: {:?}", a);
                    println!("B: {:?}", b);    
                }
                let left_hand_grad = self.get_tensor_grad(a);
                let right_hand_grad = self.get_tensor_grad(b);
                if self.advanced_logging == true {
                    println!("Left Hand Grad: {:?}", left_hand_grad);
                    println!("Right Hand Grad: {:?}", right_hand_grad);
                }

                let left_hand_data = self.get_tensor_data(a);
                let right_hand_data = self.get_tensor_data(b);
                if self.advanced_logging == true {
                    println!("Left Hand Data: {:?}", left_hand_data);
                    println!("Right Hand Data: {:?}", right_hand_data);
                }

                let new_left_hand_grad = right_hand_data * grad.clone();
                let new_right_hand_grad = left_hand_data * grad;
                if self.advanced_logging == true {
                    println!("New Left Hand Grad: {:?}", new_left_hand_grad);
                    println!("New Right Hand Grad: {:?}", new_right_hand_grad);
                }
                // yahoo and yahoo_2 are not good name, what might be better ones
                // is left_hand_grad and right_hand_grad
                // but I am too lazy to change it
                let right_hand_grad = right_hand_grad + new_right_hand_grad;
                let left_hand_grad = left_hand_grad + new_left_hand_grad;
                if self.advanced_logging == true {
                    println!("Right Hand Grad: {:?}", right_hand_grad);
                    println!("Left Hand Grad: {:?}", left_hand_grad);
                }
            
                self.set_tensor_grad(a, left_hand_grad);
                self.set_tensor_grad(b, right_hand_grad);            
            },
            Operation::Exp(a) => {
                let power_grad = self.get_tensor_grad(a);
                if self.advanced_logging {
                    println!("Power Grad: {:?}", power_grad);
                }

                let final_grad = power_grad + grad * data;
                if self.advanced_logging == true {
                    println!("Final Grad: {:?}", final_grad);
                }
                self.set_tensor_grad(a, final_grad);
            },
            Operation::Pow(base, power) => {

                let base_data = self.get_tensor_data(base);
                let power_data = self.get_tensor_data(power);
                let base_grad = self.get_tensor_grad(base);

                let power = power_data[0] - 1.0;
                //println!("Power: {:?}", power);
                
                let grad_update = power_data * base_data.mapv(|x| x.powf(power)) * grad.clone();
                //println!("Grad Update: {:?}", grad_update);
                self.set_tensor_grad(base, base_grad + grad_update);
            },
            Operation::MatMul(a, b) => {
                let out_grad = grad.clone().into_dimensionality::<Ix2>().unwrap();
                let right_hand_data = self.get_tensor_data(b);
                let right_hand_data_tranpose  = right_hand_data.t();

                let other = right_hand_data_tranpose.into_dimensionality::<Ix2>().unwrap();
                let left_hand_grad = self.get_tensor_grad(a);
                let hold = left_hand_grad + out_grad.clone().dot(&other).into_dyn();
                self.set_tensor_grad(a, hold);

                let left_hand_data = self.get_tensor_data(a);
                let right_hand_grad = self.get_tensor_grad(b);
                let other = left_hand_data.t().into_dimensionality::<Ix2>().unwrap();
                let temp = right_hand_grad + other.dot(&out_grad).into_dyn();
                self.set_tensor_grad(b, temp);

            },
            Operation::Sum(a) => {
                let left_hand_grad = self.get_tensor_grad(a);
                let grad_update = &left_hand_grad + &grad.broadcast(left_hand_grad.shape()).unwrap();
                if self.advanced_logging == true {
                    println!("Left Hand Grad: {:?}", left_hand_grad);
                    println!("Grad Update: {:?}", grad_update);
                }
                self.set_tensor_grad(a, grad_update);
            },
            Operation::Broadcast(a, _) => {
                if data.shape() == grad.shape() {
                    let mut new_grad = grad.clone();
                    for _ in 0..data.ndim() {
                        new_grad = new_grad.sum_axis(Axis(0));
                    }
                    self.set_tensor_grad(a, new_grad + self.get_tensor_grad(a));
                } else {
                    self.set_tensor_grad(a, grad + self.get_tensor_grad(a));
                }
            },
            Operation::Log(a) => {
                let base_data = self.get_tensor_data(a);
                let local_grad = base_data.map(|x|1.0 / (x + 1e-6));
                let existing_grad = self.get_tensor_grad(a);
                self.set_tensor_grad(a, existing_grad + (grad * local_grad));
            },
            Operation::View(source_tensor, origin_index) => {
                // All this should do is take the grad and put it in the right place
                let source_grad = self.get_tensor_grad(source_tensor);
                // View grad is not the same shape as source grad
                // so we want to allocate a zero tensor of the same shape as source grad
                // and then copy the view grad into the right place
                let source_shape = self.internal_tensor_store.get(&source_tensor).unwrap().shape;
                let mut new_view_grad = ArrayD::zeros(source_shape.as_ndarray_shape());
                match origin_index {
                    Indexable::Single(i) => {
                        //TODO: remove the 0, at some point as it is a "hack"
                        new_view_grad[[0, i]] = grad[0];
                    },
                    Indexable::Double(i, j) => {
                        new_view_grad[[i, j]] = grad[0];
                    },
                    _ => {
                        panic!("Not implemented");
                    }
                }


                if self.advanced_logging {
                    println!("New View Grad: {:?}", new_view_grad);
                    println!("Source Grad: {:?}", source_grad);
                }
                self.set_tensor_grad(source_tensor, source_grad + new_view_grad);
            },
            Operation::Mean(a) => {
                let curent_grad = self.get_tensor_grad(a);
                let grad_update = curent_grad + grad.clone() / (data.len() as f32);
                self.set_tensor_grad(a, grad_update);
            },
        }
    }

    pub fn get_tensor_data(&self, tensor_id: TensorID) -> ArrayD<f32> {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        let data = self.data_store[internal_tensor.data_start_index..internal_tensor.data_start_index + internal_tensor.shape.total_size()].to_vec();
        let data = ArrayD::from_shape_vec(internal_tensor.shape.as_ndarray_shape(), data).unwrap();
        data
    }

    pub fn set_tensor_data(&mut self, tensor_id: TensorID, data: ArrayD<f32>) {
        let data: Vec<f32> = data.into_raw_vec();
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        for i in 0..internal_tensor.shape.total_size() {
            self.data_store[internal_tensor.data_start_index + i] = data[i];
        }
    }

    pub fn get_tensor_grad(&self, tensor_id: TensorID) -> ArrayD<f32> {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        let grad = self.data_store[internal_tensor.grad_start_index..internal_tensor.grad_start_index + internal_tensor.shape.total_size()].to_vec();
        let grad = ArrayD::from_shape_vec(internal_tensor.shape.as_ndarray_shape(), grad).unwrap();
        grad
    }

    pub fn set_subset_of_tensor_data(&mut self, tensor_id: TensorID, indexable: Indexable, data: Vec<f32>) {
        // For now this is only one single points of data
        // Calculate the index of the data
        match indexable {
            Indexable::Single(_) => {
                self.set_single_index_tensor_data(tensor_id, indexable, data);
            },
            Indexable::Double(_, _) => {
                self.set_double_index_tensor_data(tensor_id, indexable, data);
            },
            _ => {
                panic!("Not implemented");
            }
        }
    } 

    fn set_single_index_tensor_data(&mut self, tensor_id: TensorID, indexable: Indexable, data: Vec<f32>) {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        // Can't use get_index, bug in it, calculate the offset manually
        // no need to match down from indexable, as we are only supporting single index
        let index = match indexable {
            Indexable::Single(i) => i,
            _ => panic!("Wrong Indexable")
        };
        let offset = internal_tensor.data_start_index + index;
        self.data_store[offset] = data[0];
    }

    fn set_double_index_tensor_data(&mut self, tensor_id: TensorID, indexable: Indexable, data: Vec<f32>) {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        // Can't use get_index, bug in it, calculate the offset manually
        // no need to match down from indexable, as we are only supporting single index

        match indexable {
            Indexable::Double(i, j) => {
                // this is for now, as I dont' have a good way to handle the case of more then 
                // a two indices being indexed in this manner
                assert!(internal_tensor.shape.number_of_indices == 2);
                let offset = internal_tensor.data_start_index + i * internal_tensor.shape.indices[0] + j;
                self.data_store[offset] = data[0];
            },
            _ => panic!("Wrong Indexable")
        };
        
        //let offset = internal_tensor.data_start_index + index;
        //self.data_store[offset] = data[0];
    }

    fn set_tensor_grad(&mut self, tensor_id: TensorID, grad: ArrayD<f32>) {

        assert!(grad.shape() == self.internal_tensor_store.get(&tensor_id).unwrap().shape.as_ndarray_shape());
        let grad: Vec<f32> = grad.into_raw_vec();
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        for i in 0..internal_tensor.shape.total_size() {
            self.data_store[internal_tensor.grad_start_index + i] = grad[i];
        }
    }

    pub fn zero_all_grads(&mut self) {
        for (_, internal_tensor) in self.internal_tensor_store.iter() {
            for i in 0..internal_tensor.shape.total_size() {
                self.data_store[internal_tensor.grad_start_index + i] = 0.0;
            }
        }
        
    }

    pub fn set_requires_grad(&mut self, tensor_id: TensorID, requires_grad: bool) {
        let internal_tensor = self.internal_tensor_store.get_mut(&tensor_id).unwrap();
        internal_tensor.requires_grad = requires_grad;
    }

    pub fn update_parameters(&mut self, learning_rate: f32) {
        let mut ids_to_update = HashSet::new();
        for (_, internal_tensor) in self.internal_tensor_store.iter() {
            if internal_tensor.requires_grad {
                ids_to_update.insert(internal_tensor.tensor_id);
            }
        }

        for id in ids_to_update {
            let grad = self.get_tensor_grad(id);
            let data = self.get_tensor_data(id);
            let new_data = data + grad * learning_rate;
            self.set_tensor_data(id, new_data);
        }

    }
}