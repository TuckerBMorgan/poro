use std::collections::HashMap;
use std::hash::Hash;

use ndarray::ArrayD;

use crate::nn::Module;
use crate::optimizers::optimizer::Optimizer;
use crate::{get_equation, TensorID};

pub struct AdamWBuilder {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for AdamWBuilder {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-08,
        }
    }
}

pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m: HashMap<TensorID, Vec<f32>>,
    pub v: HashMap<TensorID, Vec<f32>>,
    pub t: usize,
}

impl AdamW {
    pub fn new(adam_builder: AdamWBuilder) -> Self {
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

impl Optimizer for AdamW {
    fn record_parameters(&mut self, model: &dyn Module) {
        self.m.clear();
        self.v.clear();
        for param in model.get_parameters() {
            let test = param.item().len();
            let m1s: Vec<f32> = vec![0.0; test];
            let v1s: Vec<f32> = vec![0.0; test];
            self.m.insert(param.tensor_id, m1s);
            self.v.insert(param.tensor_id, v1s);
        }
    }

    fn step(&mut self, model: &mut dyn Module) {
        self.t += 1;
        for tensor in model.get_parameters() {
            let grad = tensor.grad();
            let m = self.m.get_mut(&tensor.tensor_id).unwrap();
            let v = self.v.get_mut(&tensor.tensor_id).unwrap();

            for (i, g) in grad.iter().enumerate() {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g.powi(2);
            }

            let m_hat = m.iter().map(|x| x / (1.0 - self.beta1.powi(self.t as i32))).collect::<Vec<f32>>();
            let v_hat = v.iter().map(|x| x / (1.0 - self.beta2.powi(self.t as i32))).collect::<Vec<f32>>();
            let mut new_weights = tensor.item().into_raw_vec();
            for (i, w) in new_weights.iter_mut().enumerate() {
                *w -= self.learning_rate * m_hat[i] / (v_hat[i].sqrt() + self.epsilon);
            }

            let new_weights = ArrayD::from_shape_vec(tensor.item().shape(), new_weights).unwrap();
            get_equation().set_tensor_data(tensor.tensor_id, new_weights);            
        }
    }
}