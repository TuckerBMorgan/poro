use poro::central::{get_equation, Shape, Tensor};

pub struct Neuron {
    pub weights: Vec<Tensor>,
    pub bias: Tensor,
}

impl Neuron {
    pub fn new(number_of_inputs: usize) -> Neuron {
        let mut weights = Vec::with_capacity(number_of_inputs);
        for _ in 0..number_of_inputs {
            let mut weight_tensor = Tensor::randn(Shape::new(vec![1]));
            weight_tensor.set_requires_grad(true);
            weights.push(weight_tensor);
        }
        let bias = Tensor::randn(Shape::new(vec![1]));
        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: &Vec<Tensor>) -> Tensor {
        let weighted_sum = inputs
            .iter()
            .zip(&self.weights)
            .fold(self.bias.clone(), |acc, (input, weight)| {
                acc + (*input * *weight)
            });

        weighted_sum.tanh()
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(number_of_neurons: usize, number_of_inputs: usize) -> Layer {
        let mut neurons = Vec::with_capacity(number_of_neurons);
        for _ in 0..number_of_inputs {
            neurons.push(Neuron::new(number_of_neurons));
        }
        Layer { neurons }
    }

    pub fn forward(&self, inputs: &Vec<Tensor>) -> Vec<Tensor> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(number_of_inputs: usize, mut neurons_per_layers: Vec<usize>) -> Network {
        let mut sizes = vec![number_of_inputs];
        sizes.append(&mut neurons_per_layers);
        let mut layers = vec![];
        for a in sizes.chunks(2) {
            layers.push(Layer::new(a[0], a[1]));
        }
        Network { layers }
    }

    pub fn forward(&self, inputs: &Vec<Tensor>) -> Vec<Tensor> {
        let mut outputs = inputs.clone();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }
}

fn main() {
    let mlp = Network::new(3, vec![4, 4, 1]);
    let inputs = vec![
        vec![2.0f32, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    let outputs = vec![1.0f32, -1.0, -1.0, 1.0];
    for _ in 0..50 {
        {
            let mut singeton = get_equation();
            singeton.zero_all_grads();
        }

        for (input, output) in inputs.iter().zip(&outputs) {
            let input = input
                .iter()
                .map(|x| Tensor::element(Shape::new(vec![1]), *x))
                .collect();
            let output = Tensor::element(Shape::new(vec![1]), *output);
            let prediction = mlp.forward(&input)[0];
            let loss = (prediction - output).pow(2.0);
            loss.backward();
        }

        {
            let mut singeton = get_equation();
            singeton.update_parameters(-0.01);
        }
    }

    let total_loss = inputs
        .iter()
        .zip(&outputs)
        .fold(0.0, |acc, (input, output)| {
            let input = input
                .iter()
                .map(|x| Tensor::element(Shape::new(vec![1]), *x))
                .collect();
            let output = Tensor::element(Shape::new(vec![1]), *output);
            let prediction = mlp.forward(&input)[0];
            acc + (prediction - output).pow(2.0).item()[[0]]
        });
    println!("{:?}", total_loss);
}
