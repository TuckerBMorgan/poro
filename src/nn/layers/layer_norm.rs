use core::panic;

use crate::central::Tensor;
use crate::nn::layers::module::Module;

pub struct LayerNorm {
    pub weight: Tensor,
    bias: Tensor,
    length_of_normalized_shape: usize,
    normalized_input: Option<Tensor>,
    var: Option<Tensor>,
    mean: Option<Tensor>,
    input_minus_mean: Option<Tensor>,
}

impl LayerNorm {
    pub fn new(embedding_dim: usize) -> Self {
        let weight = Tensor::randn(vec![embedding_dim].into());
        let bias = Tensor::randn(vec![embedding_dim].into());
        LayerNorm {
            weight,
            bias,
            length_of_normalized_shape: weight.shape.number_of_indices,
            normalized_input: None,
            var: None,
            mean: None,
            input_minus_mean: None
        }
    }

    pub fn from_weights_and_bias(weight: Tensor, bias: Tensor) -> Self {
        LayerNorm {
            weight,
            bias,
            length_of_normalized_shape: weight.shape.number_of_indices,
            normalized_input: None,
            var: None,
            mean: None,
            input_minus_mean: None
        }
    }
}

impl Module for LayerNorm {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let mean_indices = vec![2];

        let mean = x.mean(mean_indices.clone());

        self.mean = Some(mean.clone());

        let input_minus_mean = *x - mean;

        self.input_minus_mean = Some(input_minus_mean.clone());

        let var: Tensor = x.var(mean_indices);;//(input_minus_mean - input_minus_mean).pow(2.0).mean(mean_indices.clone());
        // 
        self.var = Some(var.clone());

        let std_inv = (var + 1e-5).pow(0.5);

        let normalized_input = input_minus_mean / std_inv;

        self.normalized_input = Some(normalized_input.clone());

        let output = normalized_input * self.weight + self.bias;


        output
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
mod tests {
    use core::panic;

    use ndarray::prelude::*;
    #[test]
    fn mean_tests() {
        let array = ArrayD::from_shape_simple_fn(vec![5, 5, 5], ||{return 5.0});        
        let test = Tensor::from_vec(array.into_iter().collect(), vec![5, 5, 5].into());
        let test = test.mean(vec![1, 2]);
        println!("{:?}", test.shape);
    }
    use ndarray::iter::Axes;
    use ndarray::Axis;

    use crate::central::Tensor;
    use crate::nn::layers::layer_norm::LayerNorm;
    use crate::nn::layers::module::Module;
    use simplelog::*;
    use log::info;
    use std::path::Path;    
    use std::fs::File;

    #[test]
    fn simple_test() {

        WriteLogger::init(
            LevelFilter::Info, // Set the log level
            Config::default(), // Use the default configuration
            File::create(Path::new("./LayerNormTest.log")).unwrap(), // Create or open the log file
        ).unwrap();

        let test_input_a = Tensor::from_vec(vec![1.0, 2., 3., 4., 5.0, 6.0], vec![2, 2, 2].into());
        let test_input_b = Tensor::from_vec(vec![7., 8., 9., 10.0, 11.0, 12.], vec![2, 2, 2].into());
        let test_input = test_input_a + test_input_b;
        let test_mean = test_input.mean(vec![2]);
        let fake_output = Tensor::from_vec(vec![5.0, 10., 15.], vec![3, 1].into());
        let error = test_mean - fake_output;
        let error_mean = error.mean(vec![0]);
        println!("{:?}", error_mean.item());
        error_mean.backward();
        panic!("test_input {:?}", test_input.get_id());

    }
    
    #[test]
    fn test_layer_norm() {

        WriteLogger::init(
            LevelFilter::Info, // Set the log level
            Config::default(), // Use the default configuration
            File::create(Path::new("./LayerNormTest.log")).unwrap(), // Create or open the log file
        ).unwrap();

        

        let layer_norm_weights_path = "data/tests/layer_norm/layer_norm_weights.txt";
        let layer_norm_bias_path = "data/tests/layer_norm/layer_norm_bias.txt";
        let test_input_path = "data/tests/layer_norm/test_input.txt";
        let test_input_a_path = "data/tests/layer_norm/test_input_a.txt";
        let test_input_b_path = "data/tests/layer_norm/test_input_b.txt";
        let expected_output_path = "data/tests/layer_norm/expected_output.txt";
        let fake_target = "data/tests/layer_norm/fake_target.txt";
        let expected_loss = "data/tests/layer_norm/expected_loss.txt";

        let mut layer_norm_weights_file = File::open(layer_norm_weights_path).unwrap();
        let mut layer_norm_bias_file = File::open(layer_norm_bias_path).unwrap();
        let mut test_input_file = File::open(test_input_path).unwrap();
        let mut test_input_a_file = File::open(test_input_a_path).unwrap();
        let mut test_input_b_file = File::open(test_input_b_path).unwrap();
        

        let mut expected_output_file = File::open(expected_output_path).unwrap();
        let mut fake_target_file = File::open(fake_target).unwrap();
        let mut expected_loss_file = File::open(expected_loss).unwrap();

        let mut layer_norm_weight_grad_file = File::open("data/tests/layer_norm/layer_norm_weights_grad.txt").unwrap();
        let mut layer_norm_bias_grad_file = File::open("data/tests/layer_norm/layer_norm_bias_grad.txt").unwrap();

        // Load weights and bias from files
        let layer_norm_weight= Tensor::from_bytestream(&mut layer_norm_weights_file, false).unwrap();
        let layer_norm_bias = Tensor::from_bytestream(&mut layer_norm_bias_file, false).unwrap();
        let test_input = Tensor::from_bytestream(&mut test_input_file, false).unwrap();
        let test_input_a = Tensor::from_bytestream(&mut test_input_a_file, false).unwrap();
        let test_input_b = Tensor::from_bytestream(&mut test_input_b_file, false).unwrap();
        let expected_output = Tensor::from_bytestream(&mut expected_output_file, false).unwrap();
        let fake_target = Tensor::from_bytestream(&mut fake_target_file, false).unwrap();
        let expected_loss = Tensor::from_bytestream(&mut expected_loss_file, false).unwrap();
        let expected_layer_norm_weight_grad = Tensor::from_bytestream(&mut layer_norm_weight_grad_file, false).unwrap();
        let layer_norm_bias_grad = Tensor::from_bytestream(&mut layer_norm_bias_grad_file, false).unwrap();


        
        // Create LayerNorm instance
        let mut layer_norm = LayerNorm::from_weights_and_bias(layer_norm_weight, layer_norm_bias);
        let real_test_input = test_input_a + test_input_b;
        // Perform forward pass
        let output = layer_norm.forward(&real_test_input);


        let ouput_as_flat_array = output.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();
        let expected_as_flat_array = expected_output.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();

        // Check if the output is approximately equal to the expected output
        for (o, e) in ouput_as_flat_array.iter().zip(expected_as_flat_array.iter()) {
           // assert!((o - e).abs() < 1e-4, "Output {} is not approximately equal to expected {}", o, e);
        }

        let diff = output - fake_target;


        let mse_loss = diff.pow(2.0).reshape(vec![4 * 64 * 768].into()).mean(vec![0]); 

        let expected_mse_loss = expected_loss.item();
        for (loss, expected) in mse_loss.item().iter().zip(expected_mse_loss.iter()) {
         //   assert!((loss - expected).abs() < 1e-2, "Loss {} is not approximately equal to expected {}", loss, expected);
        }

        mse_loss.backward();
        //println!("{:?}", output.item());
        println!("{:?}",real_test_input.grad());
        //println!("{:?}",layer_norm.mean.unwrap().grad());
        //println!("{:?}", mse_loss.item());
        panic!("---");

        let expected_layer_norm_weight_grad_output_flatten = expected_layer_norm_weight_grad.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();
        let layer_norm_weight_grad_output_flatten = layer_norm.weight.grad().iter().map(|x| x.clone()).collect::<Vec<f32>>();

        for (g, eg) in layer_norm_weight_grad_output_flatten.iter().zip(expected_layer_norm_weight_grad_output_flatten.iter()) {
            assert!((g - eg).abs() < 1e-2, "Gradient {} is not approximately equal to expected gradient {}", g, eg);
        }

        let expected_layer_norm_bias_grad_output_flatten = layer_norm_bias_grad.item().iter().map(|x| x.clone()).collect::<Vec<f32>>();
        let layer_norm_bias_grad_output_flatten = layer_norm.bias.grad().iter().map(|x| x.clone()).collect::<Vec<f32>>();

        for (g, eg) in layer_norm_bias_grad_output_flatten.iter().zip(expected_layer_norm_bias_grad_output_flatten.iter()) {
            assert!((g - eg).abs() < 1e-2, "Gradient {} is not approximately equal to expected gradient {}", g, eg);
        }



    }
}