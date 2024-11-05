use crate::central::Tensor;
use crate::nn::layers::module::Module;

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    pub fn new(embedding_dim: usize) -> Self {
        let weight = Tensor::randn(vec![embedding_dim].into());
        let bias = Tensor::randn(vec![embedding_dim].into());
        LayerNorm {
            weight,
            bias,
        }
    }

    pub fn from_weights_and_bias(weight: Tensor, bias: Tensor) -> Self {
        LayerNorm {
            weight,
            bias,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        

        let mean = x.mean(vec![1]);
        let input_minus_mean = *x - mean;
        let var = (input_minus_mean * input_minus_mean).mean(vec![1]);

        let std_inv = (var + 1e-05).pow(0.5);
        let normalized_input = input_minus_mean / std_inv;
        let output = normalized_input * self.weight + self.bias;
        output
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
mod tests {
    use ndarray::iter::Axes;
    use ndarray::Axis;

    use crate::central::Tensor;
    use crate::nn::layers::layer_norm::LayerNorm;
    use crate::nn::layers::module::Module;

    #[test]
    fn test_layer_norm() {

        use std::fs::File;
        let layer_norm_weights_path = "data/tests/layer_norm/layer_norm_weights.txt";
        let layer_norm_bias_path = "data/tests/layer_norm/layer_norm_bias.txt";
        let test_input_path = "data/tests/layer_norm/test_input.txt";
        let expected_output_path = "data/tests/layer_norm/expected_output.txt";
        let fake_target = "data/tests/layer_norm/fake_target.txt";
        let expected_loss = "data/tests/layer_norm/expected_loss.txt";

        let mut layer_norm_weights_file = File::open(layer_norm_weights_path).unwrap();
        let mut layer_norm_bias_file = File::open(layer_norm_bias_path).unwrap();
        let mut test_input_file = File::open(test_input_path).unwrap();
        let mut expected_output_file = File::open(expected_output_path).unwrap();
        let mut fake_target_file = File::open(fake_target).unwrap();
        let mut expected_loss_file = File::open(expected_loss).unwrap();
        let mut layer_norm_weight_grad_file = File::open("data/tests/layer_norm/layer_norm_weights_grad.txt").unwrap();
        let mut layer_norm_bias_grad_file = File::open("data/tests/layer_norm/layer_norm_bias_grad.txt").unwrap();

        // Load weights and bias from files
        let layer_norm_weight= Tensor::from_bytestream(&mut layer_norm_weights_file, false).unwrap();
        let layer_norm_bias = Tensor::from_bytestream(&mut layer_norm_bias_file, false).unwrap();
        let test_input = Tensor::from_bytestream(&mut test_input_file, false).unwrap();
        let expected_output = Tensor::from_bytestream(&mut expected_output_file, false).unwrap();
        let fake_target = Tensor::from_bytestream(&mut fake_target_file, false).unwrap();
        let expected_loss = Tensor::from_bytestream(&mut expected_loss_file, false).unwrap();
        let expected_layer_norm_weight_grad = Tensor::from_bytestream(&mut layer_norm_weight_grad_file, false).unwrap();
        let layer_norm_bias_grad = Tensor::from_bytestream(&mut layer_norm_bias_grad_file, false).unwrap();


        
        // Create LayerNorm instance
        let mut layer_norm = LayerNorm::from_weights_and_bias(layer_norm_weight, layer_norm_bias);

        // Perform forward pass
        let output = layer_norm.forward(&test_input);
        let ouput_as_flat_array = output.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();
        let expected_as_flat_array = expected_output.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();

        // Check if the output is approximately equal to the expected output
        for (o, e) in ouput_as_flat_array.iter().zip(expected_as_flat_array.iter()) {
            assert!((o - e).abs() < 1e-4, "Output {} is not approximately equal to expected {}", o, e);
        }
        let testing = output - fake_target;
        println!("fake_target: {:?}", (testing.pow(2.0)).reshape(vec![2 * 768].into()).mean(vec![0]).item());
        println!("expected_loss: {:?}", expected_loss.item());
        let diff = output - fake_target;




        let mse_loss = diff.pow(2.0).reshape(vec![2 * 768].into()).mean(vec![0]); 

        let expected_mse_loss = expected_loss.item();
        for (loss, expected) in mse_loss.item().iter().zip(expected_mse_loss.iter()) {
            assert!((loss - expected).abs() < 1e-2, "Loss {} is not approximately equal to expected {}", loss, expected);
        }

        mse_loss.backward();

        let expected_layer_norm_weight_grad_output_flatten = expected_layer_norm_weight_grad.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();
        let layer_norm_weight_grad_output_flatten = layer_norm.weight.grad().iter().map(|x| x.clone()).collect::<Vec<f32>>();

        for (g, eg) in layer_norm_weight_grad_output_flatten.iter().zip(expected_layer_norm_weight_grad_output_flatten.iter()) {
            assert!((g - eg).abs() < 1e-2, "Gradient {} is not approximately equal to expected gradient {}", g, eg);
        }

        



        



    }
}