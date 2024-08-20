use serde_json::value::Index;

use crate::central::*;
use crate::nn::*;
use crate::nn::layers::*;

pub struct PositionalEncoding {
    pub encoding: Tensor,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut encoding = Tensor::zeroes(vec![max_len, d_model].into());
        let position = Tensor::arange(0, max_len, 1).reshape(vec![max_len, 1].into());
        let mut position_real = Tensor::element(vec![10, 5].into(), 1.0);
        let position_data_as_vec = position.item().into_raw_vec();
        for x in 0..10 {
            for y in 0..5 {
                position_real.set_index(Indexable::Single(x * 5 + y), vec![position_data_as_vec[x]]);
            }

        }

        let po = Tensor::arange(0, d_model, 2) * (-(10000.0f32).ln() / d_model as f32);
        let div_term = Tensor::exp(&po).reshape(vec![d_model / 2].into());

        let mut div_term_vec = vec![0.0;50];

        for y in 0..10 {
            for x in 0..5 {
                div_term_vec[y * 5 + x] = div_term.item().into_raw_vec()[x];
            }
        }
        
        let div_term_real = Tensor::from_vec(div_term_vec, vec![10, 5].into());

        let test = div_term_real * position_real;

        let sin = test.sin();
        let cos = test.cos();

        for i in 0..max_len {
            for j in 0..d_model {
                if j % 2 == 0 {
                    encoding.set_index(Indexable::Double(i, j), vec![sin.view(Indexable::Double(i, j / 2)).item()[0]]);
                } else {
                    encoding.set_index(Indexable::Double(i, j), vec![cos.view(Indexable::Double(i, j / 2)).item()[0]]);
                }
            }
        }

        Self { encoding }
    }
}

impl Module for PositionalEncoding {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let indexes_as_tensor = Tensor::arange(0, x.shape.indices[1], 1);
        *x + self.encoding.view(Indexable::FromTensor(indexes_as_tensor.tensor_id))
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.encoding.clone()]
    }
}

// One day it would be great to get this working in Rust, but for now, here is a Python implementation of a Transformer model using PyTorch:
/*
*/

pub struct DecoderLayer {
    self_attention: AttentionHead,
    feed_forward: LinearLayer,
    
    norm1: BatchNorm1d,
    norm2: BatchNorm1d,

    dropout1: Dropout,
    dropout2: Dropout,
}

impl DecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let self_attention = AttentionHead::new(d_model, num_heads);
        let linear_layer_config = LinearLayerConfig { number_of_inputs: d_model, number_of_weights: d_ff };
        let feed_forward = LinearLayer::new(linear_layer_config);
        let norm1 = BatchNorm1d::new(d_model);
        let norm2 = BatchNorm1d::new(d_model);
        let dropout1 = Dropout::new(0.1);
        let dropout2 = Dropout::new(0.1);
        Self { self_attention, feed_forward, norm1, norm2, dropout1, dropout2 }
    }
}

impl Module for DecoderLayer {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let attn_output = self.self_attention.forward(x);
        let x = *x + self.dropout1.forward(&attn_output);
        let x = self.norm1.forward(&x);
        let ff_output = self.feed_forward.forward(&x);
        let x = x + self.dropout2.forward(&ff_output);
        let x = self.norm2.forward(&x);
        x
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.append(&mut self.self_attention.get_parameters());
        parameters.append(&mut self.feed_forward.get_parameters());
        parameters.append(&mut self.norm1.get_parameters());
        parameters.append(&mut self.norm2.get_parameters());
        parameters.append(&mut self.dropout1.get_parameters());
        parameters.append(&mut self.dropout2.get_parameters());
        parameters
    }

}
/* 
# Example usage
vocab_size = 10000  # Example vocabulary size
d_model = 512  # Dimensionality of the model
num_layers = 6  # Number of decoder layers
num_heads = 8  # Number of attention heads
d_ff = 2048  # Dimensionality of the feed-forward network
max_len = 5000  # Maximum sequence length

model = DecoderOnlyTransformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
input_sequence = torch.randint(0, vocab_size, (32, 100))  # Example input: batch of 32 sequences of length 100
output = model(input_sequence)

print(output.shape)  # Should output: (32, 100, vocab_size)
*/


pub struct DecoderOnlyTransformer {
    embedding: Embedding,
    positional_encoding: PositionalEncoding,
    layers: Vec<DecoderLayer>,
    fc_out: LinearLayer,
}

impl DecoderOnlyTransformer {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize, num_heads: usize, d_ff: usize, max_len: usize) -> Self {
        let embedding = Embedding::new(vocab_size, d_model);
        let positional_encoding = PositionalEncoding::new(d_model, max_len);
        let layers = (0..num_layers).map(|_| DecoderLayer::new(d_model, num_heads, d_ff)).collect();
        let linear_layer_config = LinearLayerConfig { number_of_inputs: d_model, number_of_weights: vocab_size };
        let fc_out = LinearLayer::new(linear_layer_config);
        Self { embedding, positional_encoding, layers, fc_out }
    }
}

impl Model for DecoderOnlyTransformer {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let x = self.embedding.forward(x);
        let mut x = x + self.positional_encoding.forward(&x);
        for layer in &mut self.layers {
            x = layer.forward(&x);
        }
        self.fc_out.forward(&x)
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.append(&mut self.embedding.get_parameters());
        for layer in &self.layers {
            parameters.append(&mut layer.get_parameters());
        }
        parameters.append(&mut self.fc_out.get_parameters());
        parameters
    }
}