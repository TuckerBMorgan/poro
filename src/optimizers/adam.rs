use std::collections::HashMap;
use std::hash::Hash;

use ndarray::ArrayD;

use crate::nn::Module;
use crate::optimizers::optimizer::Optimizer;
use crate::{get_equation, TensorID};

pub struct AdamBuilder {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for AdamBuilder {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-08,
        }
    }
}

pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m: HashMap<TensorID, Vec<f32>>,
    pub v: HashMap<TensorID, Vec<f32>>,
    pub t: usize,
}



impl Adam {
    pub fn new(adam_builder: AdamBuilder) -> Self {
        Self {
            learning_rate: adam_builder.learning_rate,
            beta1: adam_builder.beta1,
            beta2: adam_builder.beta2,
            epsilon: adam_builder.epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn record_parameters(&mut self, model: &dyn Module) {
        self.m.clear();
        self.v.clear();
        for param in model.get_parameters() {
            let test = param.item().len();
            let m1s: Vec<f32> = vec![0.0; test];
            let v1s: Vec<f32> = vec![0.0; test];
            self.m.insert(param.tensor_id, m1s); // Assuming Tensor has an id() method
            self.v.insert(param.tensor_id, v1s); // Assuming Tensor has an id() method
        }
    }

    fn step(&mut self, model: &mut dyn Module) {
        self.t += 1;
        for tensor in model.get_parameters() {

            let grad = tensor.grad(); // Assuming Tensor has a grad() method
            let m = self.m.get_mut(&tensor.tensor_id).unwrap();
            let v = self.v.get_mut(&tensor.tensor_id).unwrap();

            // Update biased first moment estimate
            for (m_i, g_i) in m.iter_mut().zip(grad.iter()) {
                *m_i = self.beta1 * *m_i + (1.0 - self.beta1) * g_i;
            }

            // Update biased second raw moment estimate
            for (v_i, g_i) in v.iter_mut().zip(grad.iter()) {
                *v_i = self.beta2 * *v_i + (1.0 - self.beta2) * g_i.powi(2);
            }

            let m_hat: Vec<f32> = m.iter().map(|x| x / (1.0 - self.beta1.powi(self.t as i32))).collect();
            let v_hat: Vec<f32> = v.iter().map(|x| x / (1.0 - self.beta2.powi(self.t as i32))).collect();

            let mut data = tensor.item().into_raw_vec();

            for i in 0..data.len() {
                data[i] -= self.learning_rate * m_hat[i] / (v_hat[i].sqrt() + self.epsilon);
            }
            let data_as_array = ArrayD::from_shape_vec(tensor.item().shape(), data).unwrap();
            get_equation().set_tensor_data(tensor.tensor_id, data_as_array); // Assuming Tensor has a tensor_id() method
        }

    }
}