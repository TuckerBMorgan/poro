
// This could maybe be its own lib, along with model.rs
use crate::central::Tensor;


pub trait Module {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
    fn set_requires_grad(&mut self, requires_grad: bool) {
        for mut parameter in self.get_parameters() {
            parameter.set_requires_grad(requires_grad);
        }
    }
}
