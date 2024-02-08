use lazy_static::lazy_static;


use std::sync::Mutex;
mod equation;
mod tensor;
mod operation;
mod shape;
mod internal_tensor;

pub use tensor::Tensor;
pub use equation::Equation;
pub use shape::Shape;


lazy_static! {
    pub static ref SINGLETON_INSTANCE: Mutex<Equation> = Mutex::new(Equation::new());
}
