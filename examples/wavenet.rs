use std::collections::HashMap;

use poro::central::{update_parameters, zero_all_grads, Indexable, Shape, Tensor};
use poro::nn::layers::{BatchNorm1d, LinearLayer, Module, Tanh};
use poro::nn::model::{Model, Sequential};
use std::fs::read_to_string;
use poro::nn::LinearLayerConfig;

struct EmbeddingLayer {
    weight: Tensor,
}

impl EmbeddingLayer {
    pub fn new(number_of_embeddings: usize, embedding_dims: usize) -> Self {
        EmbeddingLayer {
            weight: Tensor::randn(Shape::new(vec![number_of_embeddings, embedding_dims])),
        }
    }

    pub fn from_tensor(weights: Tensor) -> Self {
        EmbeddingLayer { weight: weights }
    }
}

impl Module for EmbeddingLayer {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let output_tensor = self.weight.view(Indexable::FromTensor(input.tensor_id));
        return output_tensor;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

impl From<EmbeddingLayer> for Box<dyn Module> {
    fn from(layer: EmbeddingLayer) -> Box<dyn Module> {
        Box::new(layer)
    }
}

struct FlattenConsecutive {
    block_size: usize,
}

impl FlattenConsecutive {
    pub fn new(block_size: usize) -> Self {
        FlattenConsecutive { block_size }
    }
}

impl Module for FlattenConsecutive {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let b = input.shape.indices[0];
        let t = input.shape.indices[1];
        let c = input.shape.indices[2];
        let new_middle = t / self.block_size;
        let new_end = self.block_size * c;

        let new_input = input.reshape(Shape::new(vec![b, new_middle, new_end]));

        if new_input.shape.indices[1] == 1 {
            return new_input.squeeze(1);
        }

        return new_input;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

impl From<FlattenConsecutive> for Box<dyn Module> {
    fn from(layer: FlattenConsecutive) -> Box<dyn Module> {
        Box::new(layer)
    }
}

fn build_wavenet_dataset_from_subset(
    words: &[String],
    stoi: &HashMap<char, usize>,
) -> (Vec<[usize; 8]>, Vec<usize>) {
    let mut xs = vec![];
    let mut ys = vec![];
    for word in words {
        let fixed = String::from("........") + word + ".";
        let chars: Vec<char> = fixed.chars().collect();
        for i in 0..chars.len() - 8 {
            let pair = (
                chars[i],
                chars[i + 1],
                chars[i + 2],
                chars[i + 3],
                chars[i + 4],
                chars[i + 5],
                chars[i + 6],
                chars[i + 7],
                chars[i + 8],
            );
            xs.push([
                stoi[&pair.0],
                stoi[&pair.1],
                stoi[&pair.2],
                stoi[&pair.3],
                stoi[&pair.4],
                stoi[&pair.5],
                stoi[&pair.6],
                stoi[&pair.7],
            ]);
            ys.push(stoi[&pair.8]);
        }
    }
    (xs, ys)
}

fn read_lines(filename: &str) -> Vec<String> {
    let mut result = Vec::new();

    for line in read_to_string(filename).unwrap().lines() {
        result.push(line.to_string())
    }

    result
}

fn main() {
    let n_embd = 24;
    let n_hidden = 128;
    let block_size = 8;

    let names = read_lines("./data/bigram/names.txt");

    let mut stoi = HashMap::new();
    let mut itos = HashMap::new();
    let mut i = 0;
    for c in ".abcdefghijklmnopqrstuvwxyz".chars() {
        stoi.insert(c, i);
        itos.insert(i, c);
        i += 1;
    }
    let n1 = (names.len() as f32 * 0.8f32) as usize;
    let (xtr, ytr) = build_wavenet_dataset_from_subset(&names[..n1], &stoi);
    let linear_layer_config_1 = LinearLayerConfig::new(n_embd * 2, n_hidden);
    let linear_layer_config_2 = LinearLayerConfig::new(n_hidden * 2, n_hidden);
    let linear_layer_config_3 = LinearLayerConfig::new(n_hidden * 2, n_hidden);
    let linear_layer_config_4 = LinearLayerConfig::new(n_hidden, 27);
    let mut model: Sequential = vec![
        EmbeddingLayer::new(27, n_embd).into(),
        FlattenConsecutive::new(2).into(),
        LinearLayer::new(linear_layer_config_1).into(),
        BatchNorm1d::new(n_hidden).into(),
        Tanh::new().into(),
        FlattenConsecutive::new(2).into(),
        LinearLayer::new(linear_layer_config_2).into(),
        BatchNorm1d::new(n_hidden).into(),
        Tanh::new().into(),
        FlattenConsecutive::new(2).into(),
        LinearLayer::new(linear_layer_config_3).into(),
        BatchNorm1d::new(n_hidden).into(),
        Tanh::new().into(),
        LinearLayer::new(linear_layer_config_4).into(),
    ]
    .into();

    model.set_requires_grad(true);

    let max_steps = 10;
    let batch_size = 32;

    for _i in 0..max_steps {
        zero_all_grads();
        let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![batch_size, 8]));
        for b in 0..batch_size {
            for b_index in 0..block_size {
                test_index_tensor
                    .set_index([b, b_index].into(), vec![xtr[b][b_index] as f32].into());
            }
        }
        let test = model.forward(&test_index_tensor);
        let mut test_ytrue_onehot = Tensor::element(Shape::new(vec![batch_size, 27]), 0.0);
        for b in 0..batch_size {
            test_ytrue_onehot.set_index([b, ytr[b]].into(), vec![1.0].into());
        }
        let loss = test.cross_entropy_loss(test_ytrue_onehot);
        println!("Loss: {}", loss.item());
        loss.backward();
        update_parameters(0.01);
    }
}
