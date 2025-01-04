use crate::central::Tensor;
use crate::nn::layers::linear::{LinearLayer, LinearLayerConfig};
use crate::nn::layers::module::Module;
use std::f32::consts::PI;

pub struct NewGLU {

}

impl Module for NewGLU {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let x_pow = x.pow(3.0);
        let why = 1.0 + (((2.0 / PI).sqrt() * (*x + 0.044715 * x_pow)).tanh_mapped());
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
    pub c_fc: LinearLayer,
    pub c_proj: LinearLayer,
    gelu: NewGLU,
}

impl MLP {
    fn new(config: MLPConfig) -> Self {
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

mod tests {
    use crate::nn::layers::linear;
    use crate::nn::layers::mlp::MLP;
    use crate::nn::layers::module::Module;
    use crate::central::Tensor;
    use crate::Shape;

    #[test]
    pub fn from_python_weights_and_bias()  {
        
        use std::fs::File;
        let linear_1_weights_path = "data/tests/mlp/linear_1_weights.txt";
        let linear_1_bias_path = "data/tests/mlp/linear_1_bias.txt";
        let linear_2_weights_path = "data/tests/mlp/linear_2_weights.txt";
        let linear_2_bias_path = "data/tests/mlp/linear_2_bias.txt";
        let test_input_path = "data/tests/mlp/test_input.txt";
        let expected_output_path = "data/tests/mlp/output.txt";
        let fake_target = "data/tests/mlp/fake_target.txt";
        let expected_loss = "data/tests/mlp/expected_loss.txt";
        let linear_1_weight_grad_path = "data/tests/mlp/linear_1_weight_grad.txt";
        let linear_1_bias_grad_path = "data/tests/mlp/linear_1_bias_grad.txt";
        let linear_2_weight_grad_path = "data/tests/mlp/linear_2_weight_grad.txt";
        let linear_2_bias_grad_path = "data/tests/mlp/linear_2_bias_grad.txt";

        let mut linear_1_weight_file = File::open(linear_1_weights_path).unwrap();
        let mut linear_1_bias_file = File::open(linear_1_bias_path).unwrap();
        let mut linear_2_weight_file = File::open(linear_2_weights_path).unwrap();
        let mut linear_2_bias_file = File::open(linear_2_bias_path).unwrap();
        let mut test_input_file = File::open(test_input_path).unwrap();
        let mut expected_output_file = File::open(expected_output_path).unwrap();
        let mut fake_target_file = File::open(fake_target).unwrap();
        let mut expected_loss_file = File::open(expected_loss).unwrap();
        let mut linear_1_weight_grad_file = File::open(linear_1_weight_grad_path).unwrap();
        let mut linear_1_bias_grad_file = File::open(linear_1_bias_grad_path).unwrap();
        let mut linear_2_weight_grad_file = File::open(linear_2_weight_grad_path).unwrap();
        let mut linear_2_bias_grad_file = File::open(linear_2_bias_grad_path).unwrap();


        let linear_1_weights = Tensor::from_bytestream(&mut linear_1_weight_file, false).unwrap();
        let linear_1_bias = Tensor::from_bytestream(&mut linear_1_bias_file, false).unwrap();
        let linear_2_weights = Tensor::from_bytestream(&mut linear_2_weight_file, false).unwrap();
        let linear_2_bias = Tensor::from_bytestream(&mut linear_2_bias_file, false).unwrap();
        let test_input = Tensor::from_bytestream(&mut test_input_file, false).unwrap();
        let expected_output = Tensor::from_bytestream(&mut expected_output_file, false).unwrap();
        let fake_target = Tensor::from_bytestream(&mut fake_target_file, false).unwrap();
        let expected_loss = Tensor::from_bytestream(&mut expected_loss_file, false).unwrap();

        let expected_linear_1_weight_grad = Tensor::from_bytestream(&mut linear_1_weight_grad_file, false).unwrap();
        let expected_linear_1_bias_grad = Tensor::from_bytestream(&mut linear_1_bias_grad_file, false).unwrap();
        let expected_linear_2_weight_grad = Tensor::from_bytestream(&mut linear_2_weight_grad_file, false).unwrap();
        let expected_linear_2_bias_grad = Tensor::from_bytestream(&mut linear_2_bias_grad_file, false).unwrap();       

        let mut mlp = MLP::from_weights_and_bias(linear_1_weights, linear_1_bias, linear_2_weights, linear_2_bias);
        let output = mlp.forward(&test_input);

        let output_as_flat_array = output.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();
        let expected_output_as_flat_array = expected_output.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();
        for (i, (a, b)) in output_as_flat_array.iter().zip(expected_output_as_flat_array.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "Mismatch at index {}: {} != {}", i, a, b);
        }

        let mse_loss = (output - fake_target).pow(2.0).mean(vec![1]);

        for i in 0..mse_loss.shape.size() {
            assert!((mse_loss.item()[[0, i]] - expected_loss.item()[[0, i]]).abs() < 1e-4);
        }

        // weight grad check
        mse_loss.backward();

        let linear_1_weight_bias = mlp.c_fc.bias.grad();
        let data = expected_linear_1_bias_grad.item();
        for i in 0..expected_linear_1_bias_grad.shape.size() {
            let left = linear_1_weight_bias[i];
            let right = data[[i]];
            assert!((left - right).abs() < 1e-4);
        }
        let linear_1_weight_grad = mlp.c_fc.weights.grad();
        let shape = linear_1_weight_grad.shape();
        let data = expected_linear_1_weight_grad.item();
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                let left = linear_1_weight_grad[[x, y]];
                let right = data[[x, y]];
                assert!((left - right).abs() < 1e-4);
            }
        }
        let linear_2_weight_bias = mlp.c_proj.bias.grad();
        let data = expected_linear_2_bias_grad.item();
        for i in 0..expected_linear_2_bias_grad.shape.size() {
            let left = linear_2_weight_bias[i];
            let right = data[[i]];
            assert!((left - right).abs() < 1e-4);
        }
        let linear_2_weight_grad = mlp.c_proj.weights.grad();
        let shape = linear_2_weight_grad.shape();
        let data = expected_linear_2_weight_grad.item();
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                let left = linear_2_weight_grad[[x, y]];
                let right = data[[x, y]];
                assert!((left - right).abs() < 1e-4);
            }
        }




    }

}