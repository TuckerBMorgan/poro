use lazy_static::lazy_static;

use std::sync::Mutex;
mod add_op;
mod equation;
mod indexable;
mod internal_tensor;
mod operation;
mod shape;
mod tensor;
mod mul_op;

pub use equation::{Equation, BackpropagationPacket};
pub use indexable::Indexable;
pub use shape::Shape;
pub use tensor::{Tensor, TensorID};
pub use add_op::backward;

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
