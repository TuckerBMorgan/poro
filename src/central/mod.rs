use lazy_static::lazy_static;


use std::sync::Mutex;
mod equation;
mod tensor;
mod operation;
mod shape;
mod internal_tensor;
mod indexable;
mod module;
mod model;

pub use tensor::{Tensor, TensorID};
pub use equation::Equation;
pub use shape::Shape;
pub use indexable::Indexable;
pub use module::{Module, Linear};
pub use model::{Model, LinearModel};

lazy_static! {
    pub static ref SINGLETON_INSTANCE: Mutex<Equation> = Mutex::new(Equation::new());
}


pub fn zero_all_grads() {
    let mut equation = SINGLETON_INSTANCE.lock().unwrap();
    equation.zero_all_grads();
}

pub fn update_parameters(learning_rate: f32) {
    let mut equation = SINGLETON_INSTANCE.lock().unwrap();
    equation.update_parameters(learning_rate);
}