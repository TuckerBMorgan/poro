use lazy_static::lazy_static;

use std::sync::{Mutex, MutexGuard};
mod add_op;
mod equation;
mod indexable;
mod internal_tensor;
mod operation;
mod shape;
mod tensor;
mod mul_op;
mod grad_control;

pub use equation::{Equation, BackpropagationPacket};
pub use indexable::Indexable;
pub use shape::Shape;
pub use tensor::{Tensor, TensorID};
pub use add_op::backward;
pub use grad_control::NoGrad;

lazy_static! {
    static ref SINGLETON_INSTANCE: Mutex<Equation> = Mutex::new(Equation::new());
}

pub fn get_equation() -> MutexGuard<'static, Equation> {
    SINGLETON_INSTANCE.lock().unwrap()
}

pub fn zero_all_grads() {
    let mut equation = get_equation();
    equation.zero_all_grads();
}

pub fn update_parameters(learning_rate: f32) {
    let mut equation = get_equation();
    equation.update_parameters(learning_rate);
}
