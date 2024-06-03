use crate::central::*;

pub trait Model {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
}

pub struct LinearModel {
    layers: Vec<Box<dyn Module>>,
}

impl LinearModel {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        LinearModel { layers }
    }
}

impl Model for LinearModel {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut output = x.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        for layer in &self.layers {
            parameters.extend(layer.get_parameters());
        }
        parameters
    }
}

#[test]
fn linear_model() {
    let mut linear_model = LinearModel::new(vec![Box::new(Linear::new(3, 1))]);

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

    for _ in 0..50 {
        zero_all_grads();
        let prediction = linear_model.forward(&inputs_as_tensor);
        let loss = (prediction - outputs_as_tensor).pow(2.0);
        loss.backward();
        update_parameters(-0.01);
    }
}
