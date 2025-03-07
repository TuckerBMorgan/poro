use crate::nn::Module;

pub trait Optimizer {
    fn record_parameters(&mut self, model: &dyn Module);
    fn step(&mut self, model: &mut dyn Module);
}