use lazy_static::lazy_static;


use std::sync::Mutex;
mod equation;
mod tensor;
mod operation;
mod shape;
mod internal_tensor;
mod indexable;

pub use tensor::Tensor;
pub use equation::Equation;
pub use shape::Shape;
pub use indexable::Indexable;

lazy_static! {
    pub static ref SINGLETON_INSTANCE: Mutex<Equation> = Mutex::new(Equation::new());
}
