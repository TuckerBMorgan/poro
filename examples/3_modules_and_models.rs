use log::info;
use poro::central::*;
use poro::nn::layers::LinearLayer;
use poro::nn::model::Sequential;
use poro::nn::Model;
use poro::nn::LinearLayerConfig;
use poro::nn::Module;
use poro::Array2;

use std::fs::File;
use std::path::Path;

use simplelog::*;

fn main() {

    WriteLogger::init(
        LevelFilter::Info, // Set the log level
        Config::default(), // Use the default configuration
        File::create(Path::new("modules_and_models.log")).unwrap(), // Create or open the log file
    ).unwrap();

    let mut tensor_from = Tensor::from_vec(vec![1.0, 2.0, 3.0,  4.0], vec![2, 2].into());
    let mut bias = Tensor::from_vec(vec![0.0, 0.0], vec![2].into());
    let mut linear_from = LinearLayer::from_weights_and_bias(tensor_from, bias);
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2].into());

    let array_2d = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let array_input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let array_output = array_input.dot(&array_2d.t());

    info!("array_output: {:?}", array_output);

    let output = linear_from.forward(&input);
    info!("lnear_weights {:?}", linear_from.weights.item());
    info!("Output: {:?}", output.item());
    info!("non linear layer test: {:?}", (input << tensor_from).item());
    return;

    let linear_layer_config = LinearLayerConfig::new(3, 1);
    // A layer is single module that can be used in a model
    let layer = LinearLayer::new(linear_layer_config);

    // A model is a sequence of layers
    // You can create a model by calling "into" on a vector of layers
    let mut linear_model: Sequential = vec![layer.into()].into();

    let inputs = vec![
        vec![2.0f32, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    let inputs_as_tensor = Tensor::from_vec(
        inputs.iter().flatten().map(|x| *x).collect(),
        vec![4, 3].into(),
    );

    let outputs = vec![1.0f32, -1.0, -1.0, 1.0];

    let outputs_as_tensor =
        Tensor::from_vec(outputs.iter().map(|x| *x).collect(), vec![4, 1].into());

    // This is your training loop
    for _ in 0..50 {
        zero_all_grads();
        let prediction = linear_model.forward(&inputs_as_tensor);
        let loss = (prediction - outputs_as_tensor).pow(2.0);
        info!("Loss: {:?}", loss.item());
        loss.backward();
        update_parameters(-0.01);
    }
}
