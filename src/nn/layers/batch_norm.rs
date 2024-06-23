use super::Module;
use crate::{central::Tensor, Shape};

/// This struct represents a Batch Normalization layer for 1D tensors.
/// It is used to normalize the activations of the previous layer at each batch.
pub struct BatchNorm1d {
    /// The gain tensor of the BatchNorm1d layer.
    gain: Tensor,
    /// The bias tensor of the BatchNorm1d layer.
    bias: Tensor,
}

impl BatchNorm1d {
    #[allow(unused)]
    pub fn new(number_of_weights: usize) -> BatchNorm1d {
        // Initialize the gain and bias tensors
        let gain = Tensor::ones(Shape::new(vec![number_of_weights]));
        let bias = Tensor::zeroes(Shape::new(vec![number_of_weights]));
        BatchNorm1d { gain, bias }
    }
}

impl Module for BatchNorm1d {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        // Perform the forward pass: x * gain + bias
        if x.shape.number_of_indices == 2 {
            let bnmeani = x.mean(vec![0]);
            let bnvari = x.std(vec![0]);
            let offset = *x - bnmeani;
            let numer = offset * self.gain;
            let hpreact = numer / bnvari + self.bias;
            return hpreact;
        } else if x.shape.number_of_indices == 3 {
            let bnmeani = x.mean(vec![0, 1]);
            let bnvari = x.std(vec![0, 1]);
            let offset = *x - bnmeani;
            let numer = offset * self.gain;
            let hpreact = numer / bnvari + self.bias;
            return hpreact;
        } else {
            panic!("BatchNorm1d only supports 2D and 3D tensors");
        }
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.gain.clone(), self.bias.clone()]
    }
}

impl From<BatchNorm1d> for Box<dyn Module> {
    fn from(layer: BatchNorm1d) -> Box<dyn Module> {
        Box::new(layer)
    }
}
