use crate::{central::Tensor, Shape, nn::layers::module::Module};

#[derive(Debug, Default, Copy, Clone)]
pub struct LinearLayerConfig {
    pub number_of_inputs: usize,
    pub number_of_weights: usize,
}

impl LinearLayerConfig {
    pub fn new(number_of_inputs: usize, number_of_weights: usize) -> LinearLayerConfig {
        LinearLayerConfig { number_of_inputs, number_of_weights }
    }
}
pub struct LinearLayer {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl LinearLayer {

    pub fn new(config: LinearLayerConfig) -> LinearLayer {
        let weights = Tensor::randn(Shape::new(vec![config.number_of_inputs, config.number_of_weights]));
        let bias = Tensor::ones(Shape::new(vec![config.number_of_weights]));
        LinearLayer { weights, bias }
    }

    pub fn from_weights_and_bias(weights: Tensor, bias: Tensor) -> LinearLayer {
        LinearLayer { weights, bias }
    }
}

impl Module for LinearLayer {
    
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let weight_transpose = self.weights.transpose();
        (*x << weight_transpose) + self.bias
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

impl From<LinearLayer> for Box<dyn Module> {
    fn from(layer: LinearLayer) -> Box<dyn Module> {
        Box::new(layer)
    }
}


mod tests {
    use crate::nn::layers::linear::{LinearLayer, LinearLayerConfig};
    use crate::nn::layers::module::Module;
    use crate::central::Tensor;
    use crate::Shape;

    #[test]
    pub fn test_linear_layer() {
        let config = LinearLayerConfig::new(3, 2);
        let mut linear_layer = LinearLayer::new(config);
    }

    #[test]
    pub fn from_python_weights_and_bias()  {
        
        use std::fs::File;
        let weights_path = "data/tests/linear/linear_weights.txt";
        let bias_path = "data/tests/linear/linear_bias.txt";
        let input_file = "data/tests/linear/linear_input.txt";
        let output_file = "data/tests/linear/linear_output.txt";
        let fake_target = "data/tests/linear/linear_fake_target.txt";
        let weight_grad_path = "data/tests/linear/linear_weight_grad.txt";
        let bias_grad_path = "data/tests/linear/linear_bias_grad.txt";
        let loss_path = "data/tests/linear/linear_loss.txt";
        
        let mut weights_file = File::open(weights_path).unwrap();
        let mut bias_file = File::open(bias_path).unwrap();
        let mut input_file = File::open(input_file).unwrap();
        let mut output_file = File::open(output_file).unwrap();
        let mut fake_target_file = File::open(fake_target).unwrap();
        let mut loss_file = File::open(loss_path).unwrap();
        let mut weight_grad_file = File::open(weight_grad_path).unwrap();
        let mut bias_grad_file = File::open(bias_grad_path).unwrap();

        let weights = Tensor::from_bytestream(&mut weights_file, false).unwrap();
        let bias = Tensor::from_bytestream(&mut bias_file, false).unwrap();
        let input = Tensor::from_bytestream(&mut input_file, false).unwrap();
        let expected_output = Tensor::from_bytestream(&mut output_file, false).unwrap();
        let fake_target = Tensor::from_bytestream(&mut fake_target_file, false).unwrap();
        let expected_loss = Tensor::from_bytestream(&mut loss_file, false).unwrap();
        let expected_weight_grad = Tensor::from_bytestream(&mut weight_grad_file, false).unwrap();
        let expected_bias_grad = Tensor::from_bytestream(&mut bias_grad_file, false).unwrap();
        

        let mut linear_layer = LinearLayer::from_weights_and_bias(weights, bias);
        let output = linear_layer.forward(&input);

        for i in 0..output.shape.size() {
            assert!((output.item()[i] - expected_output.item()[i]).abs() < 1e-6);
        }

        let mse_loss = (output - fake_target).pow(2.0).mean(vec![0]);

        for i in 0..mse_loss.shape.size() {
            assert!((mse_loss.item()[i] - expected_loss.item()[i]).abs() < 1e-6);
        }

        // weight grad check
        mse_loss.backward();

        let weight_bias = linear_layer.bias.grad();
        
        for i in 0..expected_bias_grad.shape.size() {
            let left = weight_bias[i];
            let right = expected_bias_grad.item()[[i]];
            assert!((left - right).abs() < 1e-6);
        }


        let weight_grad = linear_layer.weights.grad();
        for x in 0..5 {
            for y in 0..5 {
                let left = weight_grad[[x, y]];
                let right = expected_weight_grad.item()[[x, y]];
                assert!((left - right).abs() < 1e-6);
            }
        }


    }

}