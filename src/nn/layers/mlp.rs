use crate::central::Tensor;
use crate::nn::layers::linear::{LinearLayer, LinearLayerConfig};
use crate::nn::layers::module::Module;

struct NewGLU {

}

impl Module for NewGLU {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let x_pow = x.pow(3.0);
        let why = 1.0 + ((2.0 / 3.14f32).sqrt() * (*x + 0.044715 * x_pow)).tanh();
        return 0.5 * *x * why;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}


struct MLPConfig {
    embedding_dim: usize,
 }

pub struct MLP {
    c_fc: LinearLayer,
    c_proj: LinearLayer,
    gelu: NewGLU,
}

impl MLP {
    pub fn new(config: MLPConfig) -> Self {
        let c_fc = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: config.embedding_dim,
            number_of_weights: 4 * config.embedding_dim,
        });

        let c_proj = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: 4 * config.embedding_dim,
            number_of_weights: config.embedding_dim,
        });

        MLP {
            c_fc,
            c_proj,
            gelu: NewGLU {}
        }
    }

    pub fn from_weights_and_bias(c_fc_weights: Tensor, c_fc_bias: Tensor, c_proj_weights: Tensor, c_proj_bias: Tensor) -> Self {
        let c_fc = LinearLayer::from_weights_and_bias(c_fc_weights, c_fc_bias);
        let c_proj = LinearLayer::from_weights_and_bias(c_proj_weights, c_proj_bias);
        MLP {
            c_fc,
            c_proj,
            gelu: NewGLU {}
        }
    }
}

impl Module for MLP {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let x = self.c_fc.forward(x);
        let x = self.gelu.forward(&x);
        let x = self.c_proj.forward(&x);
        x
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.extend(self.c_fc.get_parameters());
        parameters.extend(self.c_proj.get_parameters());
        parameters
    }
}