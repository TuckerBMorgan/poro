use super::{add_op, matmul_op, mul_op, view};
use super::{
    indexable::Indexable, internal_tensor::InternalTensor, operation::Operation, shape::Shape,
    tensor::TensorID,
};
use core::panic;
use ndarray::linalg::general_mat_mul;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::prelude::*;
use rand_distr::{Distribution, Normal};
use std::time::Instant;
use std::{
    collections::{HashMap, HashSet},
    vec,
};

#[cfg(target_os = "windows")]
use cudarc::driver::{LaunchAsync, LaunchConfig};
#[cfg(target_os = "windows")]
use cudarc::nvrtc::{compile_ptx, Ptx};

use cudarc::cublas::*;

use log::info;

pub struct BackpropagationPacket<'a> {
    pub grad: ArrayD<f32>,
    pub data: ArrayD<f32>,
    pub equation: &'a mut Equation,
    pub operation: Operation,
    pub advanced_logging: bool,
}

/// The main struct that holds the data and operations for the equation
/// This is the main struct that the user will interact with
/// It is responsible for holding the data and operations for the equation
pub struct Equation {
    pub data_store: Vec<f32>,
    pub internal_tensor_store: HashMap<TensorID, InternalTensor>,
    value_count: usize,
    pub advanced_logging: bool,
    pub timings: HashMap<String, u128>,
    auto_grad: bool,
    #[cfg(target_os = "windows")]
    pub matmul_ptx: Option<Ptx>,
    pub operation_map: HashMap<Operation, TensorID>,
}

impl Equation {

    #[cfg(target_os = "windows")]
    pub fn new() -> Equation {
        Equation {
            data_store: Vec::new(),
            value_count: 0,
            internal_tensor_store: HashMap::new(),
            advanced_logging: false,
            timings: HashMap::new(),
            auto_grad: true,
            matmul_ptx: None,
            operation_map: HashMap::new(),
        }
    }
    #[cfg(target_os = "macos")]
    pub fn new() -> Equation {
        Equation {
            data_store: Vec::new(),
            value_count: 0,
            internal_tensor_store: HashMap::new(),
            advanced_logging: false,
            timings: HashMap::new(),
            auto_grad: true,
            operation_map: HashMap::new(),
        }
    }

    pub fn allocate_idenitity_tensor(&mut self, shape: Shape) -> TensorID {
        let mut data = vec![0.0; shape.total_size()];
        let mut index = 0;
        for i in 0..shape.indices[0] {
            for j in 0..shape.indices[1] {
                if i == j {
                    data[index] = 1.0;
                }
                index += 1;
            }
        }
        self.allocate_tensor(shape, data, Operation::Nop)
    }

    pub fn allocate_tril_tensor(&mut self, shape: Shape) -> TensorID {
        let mut data = vec![0.0; shape.total_size()];
        let mut index = 0;
        for i in 0..shape.indices[0] {
            for j in 0..shape.indices[1] {
                if i >= j {
                    data[index] = 1.0;
                }
                index += 1;
            }
        }
        self.allocate_tensor(shape, data, Operation::Nop)
    }

    pub fn allocate_tensor(
        &mut self,
        shape: Shape,
        data: Vec<f32>,
        operation: Operation,
    ) -> TensorID {
        if shape.total_size() != data.len() {
            info!("Shape: {:?}", shape);
            info!("shape number of indices: {}", shape.number_of_indices);
            info!("Shape Length: {}", shape.total_size());
            info!("Data Length: {}", data.len());
            info!("Operation: {:?}", operation);
            info!("Shape and data length mismatch");
        }

        match operation {
            Operation::Nop => {}
            _ => {
                if self.operation_map.contains_key(&operation) {
                    let tensor_id = self.operation_map.get(&operation).unwrap().clone();

                    // Set the data
                    let internal_tensor = self.internal_tensor_store.get_mut(&tensor_id).unwrap();
                    for i in 0..internal_tensor.shape.total_size() {
                        self.data_store[internal_tensor.data_start_index + i] = data[i];
                    }

                    return self.operation_map.get(&operation).unwrap().clone();
                }
            }
        }
        let tensor_id = TensorID::new(self.value_count);

        // Techinally this could be made into a single operation, but I am lazy
        let data_start_index = self.data_store.len();
        self.data_store.append(&mut data.clone());
        let grad_start_index = self.data_store.len();
        self.data_store.append(&mut vec![0.0; shape.total_size()]);

        let internal_tensor = InternalTensor::new(
            tensor_id,
            shape,
            operation,
            data_start_index,
            grad_start_index,
        );

        // Need to update the rules where any tensor that is created by
        // an operation where one of more of the inputs requires grad
        // then the output tensor should also require grad
        //       internal_tensor.requires_grad = self.auto_grad;

        self.internal_tensor_store
            .insert(tensor_id, internal_tensor);

        self.value_count += 1;
        self.operation_map.insert(operation, tensor_id);
        return tensor_id;
    }

    pub fn allocate_zero_tensor(&mut self, shape: Shape, operation: Operation) -> TensorID {
        let data = vec![0.0; shape.total_size()];
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_ones_tensor(&mut self, shape: Shape, operation: Operation) -> TensorID {
        let data = vec![1.0; shape.total_size()];
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_randn_tensor(&mut self, shape: Shape, operation: Operation) -> TensorID {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f32> = (0..shape.total_size())
            .map(|_| normal.sample(&mut rng))
            .collect();
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_element_tensor(
        &mut self,
        shape: Shape,
        data: f32,
        operation: Operation,
    ) -> TensorID {
        let data = vec![data; shape.total_size()];
        self.allocate_tensor(shape, data, operation)
    }

    pub fn allocate_tensor_from_operation(
        &mut self,
        shape: Shape,
        data: Vec<f32>,
        operation: Operation,
    ) -> TensorID {
        self.allocate_tensor(shape, data, operation)
    }

    fn add_time(&mut self, operation: &str, now: Instant) {
        if !self.timings.contains_key(&operation.to_string()) {
            self.timings.insert(operation.to_string(), 0);
        }
        let elapsed = now.elapsed().as_micros();
        let current_time = self.timings.get(&operation.to_string()).unwrap();
        self.timings
            .insert(operation.to_string(), current_time + elapsed);
    }

    #[cfg(target_os = "windows")]
    pub fn cuda_matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
        use cudarc::{cublas, driver::result::device};



        let now = Instant::now();
        let dev: std::sync::Arc<cudarc::driver::CudaDevice> =
            cudarc::driver::CudaDevice::new(0).unwrap();
        let d= cublas::CudaBlas::new(dev).unwrap();

        let a_shape = a.shape();
        let b_shape = b.shape();

        if self.matmul_ptx.is_none() {
            const PTX_SRC: &str = "
            extern \"C\" __global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
                int ROW = blockIdx.y * blockDim.y + threadIdx.y;
                int COL = blockIdx.x * blockDim.x + threadIdx.x;
    
                float tmpSum = 0.0;
    
                if (ROW < M && COL < K) {
                    // each thread computes one element of the block sub-matrix
                    for (int i = 0; i < N; i++) {
                        tmpSum = tmpSum + (A[ROW * N + i] + 0.0000001) * (B[i * K + COL] + 0.0000001);
                    }
                    C[ROW * K + COL] = tmpSum;
                }
            }";
            let ptx = compile_ptx(PTX_SRC).unwrap();
            self.matmul_ptx = Some(ptx);
        }

        dev.load_ptx(
            self.matmul_ptx.as_ref().unwrap().clone(),
            "matmul",
            &["matmul"],
        )
        .unwrap();



        let f = dev.get_func("matmul", "matmul").unwrap();
        let tile_size = 16;

        let m = a_shape[0];
        let n = a_shape[1];
        let k = b_shape[1];

        let grid_rows = (m + tile_size - 1) / tile_size;
        let grid_cols = (k + tile_size - 1) / tile_size;

        let grid_dims = (grid_cols as u32, grid_rows as u32, 1);
        let cfg = LaunchConfig {
            block_dim: (tile_size as u32, tile_size as u32, 1),
            grid_dim: grid_dims,
            shared_mem_bytes: 0,
        };

        self.add_time("Setup", now);
        let now = Instant::now();

        // TODO: Find out why I need to add a little to these to avoid a problem
        let epi = 0.00000001;
        let new_a: ArrayD<f32> = a + ArrayD::ones(a_shape) * epi;
        let new_b: ArrayD<f32> = b + ArrayD::ones(b_shape) * epi;
        let a_on_device = dev.htod_sync_copy(&new_a.into_raw_vec()).unwrap();
        let b_on_device = dev.htod_sync_copy(&new_b.into_raw_vec()).unwrap();

        let c_host = vec![0.0f32; a_shape[0] * b_shape[1]];

        let mut out_on_device = dev.htod_sync_copy(&c_host).unwrap();

        unsafe {
            let result = f.launch(
                cfg,
                (&a_on_device, &b_on_device, &mut out_on_device, m, n, k),
            );
            match result {
                Ok(_) => {}
                Err(e) => {
                    println!("Error: {:?}", e);
                }
            }
        }
        self.add_time("Launching", now);
        let now = Instant::now();
        let c_host = dev.dtoh_sync_copy(&out_on_device).unwrap();
        self.add_time("Copy back", now);
        return ArrayD::from_shape_vec(vec![a_shape[0], b_shape[1]], c_host).unwrap();
    }

    fn matmul_4d(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
        // Ensure both arrays are 4-dimensional
        if a.ndim() != 4 || b.ndim() != 4 {
            panic!("Both arrays must be 4-dimensional.");
        }
    
        let shape_a = a.shape();
        let shape_b = b.shape();
    
        // Check inner dimensions for compatibility
        if shape_a[3] != shape_b[2] {
            panic!(
                "Dimension mismatch: a.shape[3] ({}) != b.shape[2] ({})",
                shape_a[3], shape_b[2]
            );
        }
    
        // Check that the outer dimensions match
        if shape_a[0] != shape_b[0] || shape_a[1] != shape_b[1] {
            panic!("Outer dimensions of arrays do not match.");
        }
    
        // Determine the shape of the result array
        let result_shape = [shape_a[0], shape_a[1], shape_a[2], shape_b[3]];
        let mut result = ArrayD::<f32>::zeros(IxDyn(&result_shape));
    
        // Iterate over the first two dimensions
        for i in 0..shape_a[0] {
            for j in 0..shape_a[1] {
                // Extract 2D slices from both arrays
                let a_slice = a.slice(s![i, j, .., ..]);
                let b_slice = b.slice(s![i, j, .., ..]);
    
                // Convert slices to 2D arrays
                let a_mat = a_slice.into_dimensionality::<Ix2>().unwrap();
                let b_mat = b_slice.into_dimensionality::<Ix2>().unwrap();
    
                // Perform matrix multiplication
                let res = a_mat.dot(&b_mat);
    
                // Assign the result back to the corresponding slice in the result array
                result
                    .slice_mut(s![i, j, .., ..])
                    .assign(&res);
            }
        }
    
        result
    }

    pub fn standard_matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {

        if a.ndim() == 1 && b.ndim() == 2 {
            // Convert both down to 2D arrays
            let a_2d = a.clone().into_dimensionality::<Ix1>().unwrap();
            let b_2d = b.clone().into_dimensionality::<Ix2>().unwrap();
            return a_2d.dot(&b_2d).into_dyn();

        }

        if a.ndim() == 2 && b.ndim() == 1 {
            // Convert both down to 2D arrays
            let a_2d = a.clone().into_dimensionality::<Ix2>().unwrap();
            let b_2d = b.clone().into_dimensionality::<Ix1>().unwrap();

            return a_2d.dot(&b_2d).into_dyn();
        }


        if a.ndim() == 2 && b.ndim() == 2 {
            // Convert both down to 2D arrays
            let a_2d = a.clone().into_dimensionality::<Ix2>().unwrap();

            let b_2d = b.clone().into_dimensionality::<Ix2>().unwrap();
            return a_2d.dot(&b_2d).into_dyn();
        }

        if a.ndim() == 3 && b.ndim() == 2 {
            let a_3d = a.clone().into_dimensionality::<Ix3>().unwrap();
            let b_2d = b.clone().into_dimensionality::<Ix2>().unwrap();
            let mut result = Array3::zeros((a_3d.shape()[0], a_3d.shape()[1], b_2d.shape()[1]));
            for i in 0..a_3d.shape()[0] {
                let a_slice = a_3d.slice(s![i, .., ..]);
                let temp = a_slice.dot(&b_2d);
                result.slice_mut(s![i, .., ..]).assign(&temp);
            }
            return result.into_dyn();
        }

        if a.ndim() == 3 && b.ndim() == 3 {
            let a_3d = a.clone().into_dimensionality::<Ix3>().unwrap();
            let b_3d = b.clone().into_dimensionality::<Ix3>().unwrap();
            let mut result = Array3::zeros((a_3d.shape()[0], a_3d.shape()[1], b_3d.shape()[2]));
            for i in 0..a_3d.shape()[0] {
                let a_slice = a_3d.slice(s![i, .., ..]);
                let b_slice = b_3d.slice(s![i, .., ..]);
                let temp = a_slice.dot(&b_slice);
                result.slice_mut(s![i, .., ..]).assign(&temp);
            }
            return result.into_dyn();
        }

        if a.ndim() == 3 && b.ndim() == 1 {
            let a_3d = a.clone().into_dimensionality::<Ix3>().unwrap();
            let b_1d = b.clone().into_dimensionality::<Ix1>().unwrap();
            let mut result = Array3::zeros((a_3d.shape()[0], a_3d.shape()[1], 1));
            for i in 0..a_3d.shape()[0] {
                let a_slice = a_3d.slice(s![i, .., ..]);
                let temp = a_slice.dot(&b_1d);
                result.slice_mut(s![i, .., ..]).assign(&temp);
            }
            return result.into_dyn();
        }

        if a.ndim() == 4 && b.ndim() == 4 {
            return Equation::matmul_4d(a, b);
        }

        panic!("a shape: {:?}, b shape: {:?}", a.shape(), b.shape());
    }

    #[cfg(target_os = "windows")]
    pub fn matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
        // I have only done the 2D case for now
        // so even if we have a cuda device, we should default to the standard matmul
        // for any case of 2dx2d
        if cudarc::driver::CudaDevice::new(0).is_ok() && a.ndim() == 2 && b.ndim() == 2 {
            // return self.standard_matmul(a, b);
            return self.cuda_matmul(a, b);
        } else {
            return self.standard_matmul(a, b);
        }
    }

    #[cfg(target_os = "macos")]
    pub fn matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
        // I have only done the 2D case for now
        // so even if we have a cuda device, we should default to the standard matmul
        // for any case of 2dx2d

        return self.standard_matmul(a, b);
    }

    pub fn element_wise_add(&self, a: TensorID, b: TensorID) -> Vec<f32> {
        let a_internal = self.internal_tensor_store.get(&a).unwrap();
        let b_internal = self.internal_tensor_store.get(&b).unwrap();
        // I want to do in parrallel
        let indices = 0..a_internal.shape.total_size();

        let data = indices
            .into_par_iter()
            .map(|i| {
                let a_data = self.data_store[a_internal.data_start_index + i];
                let b_data = self.data_store[b_internal.data_start_index + i];
                let result = a_data + b_data;
                return result;
            })
            .collect::<Vec<f32>>();

        data
    }

    pub fn get_item(&self, tensor_id: TensorID) -> Vec<f32> {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        let data = self.data_store[internal_tensor.data_start_index
            ..internal_tensor.data_start_index + internal_tensor.shape.total_size()]
            .to_vec();
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
        if let Some(dependencies) = self
            .internal_tensor_store
            .get(&node)
            .map(|n| n.dependencies())
        {
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
        let now = Instant::now();
        // Initialize visited set and stack for topological sort
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        // Perform the topological sort
        self.topological_sort_util(starting_value, &mut visited, &mut stack);
        let elapsed = now.elapsed().as_micros();
        self.timings.insert("Topological Sort".to_string(), elapsed);

        // Initialize gradients
        let ones = ArrayD::ones(
            self.internal_tensor_store
                .get(&starting_value)
                .unwrap()
                .shape
                .as_ndarray_shape(),
        );
        self.set_tensor_grad(starting_value, ones.clone());

        // Process nodes in topologically sorted order
        while let Some(node) = stack.pop() {
            let _ = self.backward_for_value(node); // Assuming this calculates and returns the children of the node
        }
        if !self.timings.contains_key("Backward") {
            self.timings.insert("Backward".to_string(), 0);
        }
    }

    fn backward_for_value(&mut self, node: TensorID) {
        let now = Instant::now();
        let internal_tensor = self.internal_tensor_store.get(&node).unwrap();
        let data = self.get_tensor_data(node);
        let grad = self.get_tensor_grad(node);
        if self.advanced_logging == true {
            println!("Node: {:?}", node);
            println!("Data: {:?}", data);
            println!("Grad: {:?}", grad);
        }

        let operation = internal_tensor.operation.clone();
        let advance_logging = self.advanced_logging;
        let backprop_packet = BackpropagationPacket {
            grad: grad.clone(),
            data: data.clone(),
            equation: self,
            operation: operation.clone(),
            advanced_logging: advance_logging,
        };
        match operation {
            Operation::Nop => {
                // Do nothing
            }
            Operation::Add(_a, _b) => {
                add_op::backward(backprop_packet);
            }
            Operation::Mul(_a, _b) => {
                mul_op::backward(backprop_packet);
            }
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
            }
            Operation::Pow(base, power) => {
                let base_data = self.get_tensor_data(base);
                let power_data = self.get_tensor_data(power);
                let base_grad = self.get_tensor_grad(base);

                let power = power_data[0] - 1.0;

                let grad_update = power_data * base_data.mapv(|x| x.powf(power)) * grad.clone();
                self.set_tensor_grad(base, base_grad + grad_update);
            }
            Operation::MatMul(_a, _b) => {
                matmul_op::backward(backprop_packet);
            }
            Operation::Sum(a, _axis) => {
                let left_hand_grad = self.get_tensor_grad(a);
                let grad_update =
                    &left_hand_grad + &grad.broadcast(left_hand_grad.shape()).unwrap();
                if self.advanced_logging == true {
                    println!("Left Hand Grad: {:?}", left_hand_grad);
                    println!("Grad Update: {:?}", grad_update);
                }
                self.set_tensor_grad(a, grad_update);
            }
            Operation::Broadcast(a, _to_shape) => {
                let left_hand_grad = self.get_tensor_grad(a);

                let mut result = grad.clone();
                let input_shape = grad.shape();
                let origina_shape = left_hand_grad.shape();

                assert!(
                    input_shape.len() >= origina_shape.len(),
                    "Input Shape: {:?}, Original Shape: {:?}",
                    input_shape,
                    origina_shape
                );

                for i in 0..input_shape.len() {
                    let input_dim = input_shape[input_shape.len() - 1 - i];
                    let orig_dim = if i < origina_shape.len() {
                        origina_shape[origina_shape.len() - 1 - i]
                    } else {
                        1
                    };
                    if orig_dim == 1 && input_dim != 1 {
                        result = result.sum_axis(Axis(input_shape.len() - 1 - i));
                    }
                }

                result = result.into_shape(origina_shape).unwrap();
                let grad_update = left_hand_grad + result;

                self.set_tensor_grad(a, grad_update);
            }
            Operation::Log(a) => {
                let base_data = self.get_tensor_data(a);
                let local_grad = base_data.map(|x| 1.0 / (x));
                let existing_grad = self.get_tensor_grad(a);
                let grad_update = existing_grad + grad.clone() * local_grad;
                self.set_tensor_grad(a, grad_update);
            }
            Operation::View(_, _) => {
                view::backward(backprop_packet);
            }
            Operation::Mean(a) => {
                let curent_grad = self.get_tensor_grad(a);
                let local_grad = grad.clone() / (curent_grad.clone().len() as f32);
                let grad_update = curent_grad + local_grad;
                self.set_tensor_grad(a, grad_update);
            }
            Operation::Concat(a, b) => {
                // break apart the incoming grad into the two parts
                let left_hand_grad = self.get_tensor_grad(a);
                let right_hand_grad = self.get_tensor_grad(b);
                let mut left_hand_grad = left_hand_grad.clone();
                let mut right_hand_grad = right_hand_grad.clone();
                let grad = grad.clone();
                let left_shape = left_hand_grad.shape();
                let right_shape = right_hand_grad.shape();
                let grad_shape = grad.shape();
                let mut left_grad: ArrayD<f32> = ArrayD::zeros(left_shape);
                let mut right_grad = ArrayD::zeros(right_shape);
                let grad = ArrayD::zeros(grad_shape);
                let mut left_index = 0;
                let mut right_index = 0;
                let mut grad_index = 0;
                for i in 0..grad_shape[0] {
                    if i < left_shape[0] {
                        left_grad[left_index] = grad[grad_index];
                        left_index += 1;
                    } else {
                        right_grad[right_index] = grad[grad_index];
                        right_index += 1;
                    }
                    grad_index += 1;
                }
                left_hand_grad = left_hand_grad + left_grad;
                right_hand_grad = right_hand_grad + right_grad;
                self.set_tensor_grad(a, left_hand_grad);
                self.set_tensor_grad(b, right_hand_grad);
            }
            Operation::Reshape(a, _) => {
                let source_grad = self.get_tensor_grad(a);
                let grad_update = source_grad.clone();
                self.set_tensor_grad(a, grad_update);
            }
            Operation::Tanh(a) => {
                let mut source_data = self.get_tensor_data(a);
                let source_grad = self.get_tensor_grad(a);
                source_data.par_map_inplace(|x| *x = 1.0 - x.tanh().powf(2.0));
                let grad_update = source_grad + grad.clone() * source_data;
                self.set_tensor_grad(a, grad_update);
            }
            Operation::Transpose(a, first_index, second_index) => {
                let mut grad_clone = grad.clone();
                grad_clone.swap_axes(first_index, second_index);
                let source_grad = self.get_tensor_grad(a);
                let grad_update = source_grad + grad_clone;
                self.set_tensor_grad(a, grad_update);
            }
            Operation::Sin(a) => {
                let source_data = self.get_tensor_data(a);
                let source_grad = self.get_tensor_grad(a);
                let grad_update = source_grad + grad.clone() * source_data.map(|x| x.cos());
                self.set_tensor_grad(a, grad_update);
            }
            Operation::Cos(a) => {
                let source_data = self.get_tensor_data(a);
                let source_grad = self.get_tensor_grad(a);
                let grad_update = source_grad + grad.clone() * source_data.map(|x| -x.sin());
                self.set_tensor_grad(a, grad_update);
            },
            Operation::MaskedFill(a, b, c) => {
                panic!("Masked fill Not Implemented");
            }
        }

        match operation {
            Operation::MatMul(_a, _b) => {}
            _ => {
                if !self.timings.contains_key(&operation.to_string()) {
                    self.timings.insert(operation.to_string(), 0);
                }
                let elapsed = now.elapsed().as_micros();
                let current_time = self.timings.get(&operation.to_string()).unwrap();
                self.timings
                    .insert(operation.to_string(), current_time + elapsed);
            }
        }
    }

    pub fn get_tensor_data(&self, tensor_id: TensorID) -> ArrayD<f32> {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        let data = self.data_store[internal_tensor.data_start_index
            ..internal_tensor.data_start_index + internal_tensor.shape.total_size()]
            .to_vec();
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

    pub fn set_tensor_data_from_vec(&mut self, tensor_id: TensorID, data: &Vec<f32>) {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        for i in 0..internal_tensor.shape.total_size() {
            self.data_store[internal_tensor.data_start_index + i] = data[i];
        }
    }

    pub fn get_tensor_grad(&self, tensor_id: TensorID) -> ArrayD<f32> {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        let grad = self.data_store[internal_tensor.grad_start_index
            ..internal_tensor.grad_start_index + internal_tensor.shape.total_size()]
            .to_vec();
        let grad = ArrayD::from_shape_vec(internal_tensor.shape.as_ndarray_shape(), grad).unwrap();
        grad
    }

    pub fn set_subset_of_tensor_data(
        &mut self,
        tensor_id: TensorID,
        indexable: Indexable,
        data: Vec<f32>,
    ) {
        // For now this is only one single points of data
        // Calculate the index of the data
        match indexable {
            Indexable::Single(_) => {
                self.set_single_index_tensor_data(tensor_id, indexable, data);
            }
            Indexable::Double(_, _) => {
                self.set_double_index_tensor_data(tensor_id, indexable, data);
            }
            Indexable::Triple(_, _, _) => {
                self.set_triple_index_tensor_data(tensor_id, indexable, data);
            }
            Indexable::FromTensor(tensor_id) => {
                self.set_tensor_index_tensor_data(tensor_id, indexable, data);
            }
        }
    }

    fn set_single_index_tensor_data(
        &mut self,
        tensor_id: TensorID,
        indexable: Indexable,
        data: Vec<f32>,
    ) {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        // Can't use get_index, bug in it, calculate the offset manually
        // no need to match down from indexable, as we are only supporting single index
        let index = match indexable {
            Indexable::Single(i) => i,
            _ => panic!("Wrong Indexable"),
        };
        let offset = internal_tensor.data_start_index + index;
        self.data_store[offset] = data[0];
    }

    fn set_double_index_tensor_data(
        &mut self,
        tensor_id: TensorID,
        indexable: Indexable,
        data: Vec<f32>,
    ) {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        // Can't use get_index, bug in it, calculate the offset manually
        // no need to match down from indexable, as we are only supporting single index

        match indexable {
            Indexable::Double(i, j) => {
                // this is for now, as I dont' have a good way to handle the case of more then
                // a two indices being indexed in this manner
                assert!(internal_tensor.shape.number_of_indices == 2);
                let offset =
                    internal_tensor.data_start_index + i * internal_tensor.shape.indices[1] + j;
                self.data_store[offset] = data[0];
            }
            _ => panic!("Wrong Indexable"),
        };

        //let offset = internal_tensor.data_start_index + index;
        //self.data_store[offset] = data[0];
    }

    fn set_triple_index_tensor_data(
        &mut self,
        tensor_id: TensorID,
        indexable: Indexable,
        data: Vec<f32>,
    ) {
        let internal_tensor = self.internal_tensor_store.get(&tensor_id).unwrap();
        // Can't use get_index, bug in it, calculate the offset manually
        // no need to match down from indexable, as we are only supporting single index

        match indexable {
            Indexable::Triple(i, j, k) => {
                // this is for now, as I dont' have a good way to handle the case of more then
                // a two indices being indexed in this manner
                assert!(internal_tensor.shape.number_of_indices == 3);
                let offset = internal_tensor.data_start_index
                    + i * internal_tensor.shape.indices[1] * internal_tensor.shape.indices[2]
                    + j * internal_tensor.shape.indices[2]
                    + k;
                self.data_store[offset] = data[0];
            }
            _ => panic!("Wrong Indexable"),
        };

        //let offset = internal_tensor.data_start_index + index;
        //self.data_store[offset] = data[0];
    }

    fn set_tensor_index_tensor_data(&mut self, tensor_id: TensorID, indexable: Indexable, data: Vec<f32>) {
        let index = match indexable {
            Indexable::FromTensor(i) => i,
            _ => panic!("Wrong Indexable"),
        };
        let indices = self.get_tensor_data(index);

        for (count, index) in indices.iter().enumerate() {
            self.set_single_index_tensor_data(tensor_id, Indexable::Single(*index as usize), vec![data[count]]);
        }
    }

    pub fn set_tensor_grad(&mut self, tensor_id: TensorID, grad: ArrayD<f32>) {
        assert!(
            grad.shape()
                == self
                    .internal_tensor_store
                    .get(&tensor_id)
                    .unwrap()
                    .shape
                    .as_ndarray_shape()
        );
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

    pub fn disable_grad(&mut self) {
        self.auto_grad = false;
    }

    pub fn enable_grad(&mut self) {
        self.auto_grad = true;
    }
}
