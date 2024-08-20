use poro::central::*;
use poro::nn::layers::LinearLayer;
use poro::nn::model::Sequential;
use poro::nn::Model;
use poro::nn::LinearLayerConfig;
fn main() {
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
        loss.backward();
        update_parameters(-0.01);
    }
}
