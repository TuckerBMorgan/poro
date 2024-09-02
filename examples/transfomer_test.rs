
use std::env::consts::ARCH;

use poro::nn::model::DecoderOnlyTransformer;
use poro::central::Tensor;
use poro::nn::{AttentionHead, Embedding, LinearLayer, Model, Module};
use poro::nn::model::PositionalEncoding;
use poro::nn::LinearLayerConfig;
use poro::Indexable;
use serde_json::Value;
use std::io::{Read, Result};
use std::convert::TryInto;
struct NewGLU {

}

impl Module for NewGLU {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let x_pow = x.pow(3.0);
        let why = 1.0 + ((2.0 / 3.14f32).sqrt() * (*x + 0.044715 * x_pow)).tanh();
        return 0.5 * *x * why;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

struct CasualSelfAttentionConfig {
    number_of_heads: usize,
    embedding_dim: usize,
}

struct CasualSelfAttention {
    query_attention: LinearLayer,
    key_attention: LinearLayer,
    value_attention: LinearLayer,
    c_proj: LinearLayer,
}

impl CasualSelfAttention {
    pub fn new(config: CasualSelfAttentionConfig) -> Self {
        let query_attention = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: config.embedding_dim,
            number_of_weights: config.embedding_dim,
        });

        let key_attention = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: config.embedding_dim,
            number_of_weights: config.embedding_dim,
        });

        let value_attention = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: config.embedding_dim,
            number_of_weights: config.embedding_dim,
        });

        let c_proj = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: config.embedding_dim,
            number_of_weights: config.embedding_dim,
        });

        CasualSelfAttention {
            query_attention,
            key_attention,
            value_attention,
            c_proj,
        }
    }
}

impl Module for CasualSelfAttention {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let query = self.query_attention.forward(x);
        let key = self.key_attention.forward(x);
        let value = self.value_attention.forward(x);

        let attn_weights = query << key.tranpose_with_provided_axis(0, 1);
        
        let attn_weights = attn_weights.softmax(attn_weights.shape.number_of_indices - 1);
        let attn_output = attn_weights << value;
        let x = self.c_proj.forward(&attn_output);
        x
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.extend(self.query_attention.get_parameters());
        parameters.extend(self.key_attention.get_parameters());
        parameters.extend(self.value_attention.get_parameters());
        parameters.extend(self.c_proj.get_parameters());
        parameters
    }
}


 struct MLPConfig {
    embedding_dim: usize,
 }

struct MLP {
    c_fc: LinearLayer,
    c_proj: LinearLayer,
    gelu: NewGLU,
}

impl MLP {
    pub fn new(config: MLPConfig) -> Self {
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

    pub fn from_weight_file(filepath: &str) -> MLP {
        let as_json = std::fs::read_to_string(filepath).unwrap();
        
        let json: Value = serde_json::from_str(&as_json).unwrap();
        //just print keys
        for key in json.as_object().unwrap().keys() {
            println!("{}", key);
        }
        let as_array = json["c_fc.weight"].as_array().unwrap();
        let test = as_array.iter().map(|x| x.as_array().unwrap()).flatten().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let c_fc_weights = Tensor::from_vec(test, vec![768, 4 * 768].into());
        let as_array = json["c_fc.bias"].as_array().unwrap();
        let test = as_array.iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let c_fc_bias = Tensor::from_vec(test, vec![4 * 768].into());
        let c_fc = LinearLayer::from_weights_and_bias(c_fc_weights, c_fc_bias);

        let as_array = json["c_proj.weight"].as_array().unwrap();
        let test = as_array.iter().map(|x| x.as_array().unwrap()).flatten().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let c_proj_weights = Tensor::from_vec(test, vec![4 * 768, 768].into());
        let as_array = json["c_proj.bias"].as_array().unwrap();
        let test = as_array.iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        let c_proj_bias = Tensor::from_vec(test, vec![768].into());
        let c_proj = LinearLayer::from_weights_and_bias(c_proj_weights, c_proj_bias);

        let gelu = NewGLU {};

        MLP {
            c_fc,
            c_proj,
            gelu
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


struct BlockConfig {
    embedding_dim: usize,
    casual_self_attention_config: CasualSelfAttentionConfig,
}

struct Block {
    ln_1: LinearLayer,
    ln_2: LinearLayer,
    attn: CasualSelfAttention,
    mlp: MLP,
    embedding_dim: usize,
}

impl Block {
    pub fn new(config: BlockConfig) -> Self {
        let ln_1 = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: config.embedding_dim,
            number_of_weights: config.embedding_dim,
        });

        let ln_2 = LinearLayer::new(LinearLayerConfig {
            number_of_inputs: config.embedding_dim,
            number_of_weights: config.embedding_dim,
        });

        let attn = CasualSelfAttention::new(config.casual_self_attention_config);

        let mlp = MLP::new(MLPConfig {
            embedding_dim: config.embedding_dim,
        });

        Block {
            ln_1,
            ln_2,
            attn,
            mlp,
            embedding_dim: config.embedding_dim,
        }
    }
}

impl Module for Block {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let x = self.ln_1.forward(x);
        let x = self.attn.forward(&x);
        let x = x + x;
        let x = self.ln_2.forward(&x);
        let x = self.mlp.forward(&x);
        let x = x + x;
        x
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.extend(self.ln_1.get_parameters());
        parameters.extend(self.ln_2.get_parameters());
        parameters.extend(self.attn.get_parameters());
        parameters.extend(self.mlp.get_parameters());
        parameters
    }
}

struct GPTConfig {
    block_size: usize,
    vocab_size: usize,
    number_of_layers: usize,
    number_of_heads: usize,
    embedding_dim: usize  
}

impl GPTConfig {
    pub fn from_bytestream<R: Read>(reader: &mut R) -> GPTConfig {
        let mut buffer = [0u8; 4];
        // magic number
        
        reader.read_exact(&mut buffer).unwrap();
        println!("{:?}", buffer);
        // version
        reader.read_exact(&mut buffer).unwrap();
        println!("{:?}", buffer);

        reader.read_exact(&mut buffer).unwrap();
        let block_size = u32::from_le_bytes(buffer).try_into().unwrap();
        reader.read_exact(&mut buffer).unwrap();
        let vocab_size = u32::from_le_bytes(buffer).try_into().unwrap();
        reader.read_exact(&mut buffer).unwrap();
        let number_of_layers = u32::from_le_bytes(buffer).try_into().unwrap();
        reader.read_exact(&mut buffer).unwrap();
        let number_of_heads = u32::from_le_bytes(buffer).try_into().unwrap();
        reader.read_exact(&mut buffer).unwrap();
        let embedding_dim = u32::from_le_bytes(buffer).try_into().unwrap();
        println!("Voc")
        GPTConfig {
            block_size,
            vocab_size,
            number_of_layers,
            number_of_heads,
            embedding_dim,
        }
    }
}

struct GPT {
    wte: Embedding,
    wpe: PositionalEncoding,
    blocks: Vec<Block>,
}

impl GPT {
    pub fn build_from_checkpoint_file(filepath: &str) -> GPT {
        let read = &mut std::fs::File::open(filepath).unwrap();   
        // feed 256 bytes to GPTConfig::from_bytestream
        let mut buffer = [0; 256 * 4];

        read.read_exact(&mut buffer).unwrap();

        let gpt_config = GPTConfig::from_bytestream(&mut buffer.as_ref());
        
        let mut buffer = [0u8; 4];
        read.read_exact(&mut buffer).unwrap();
        let number_of_dimensions = i32::from_le_bytes(buffer);
        println!("{:?}", number_of_dimensions);
        for i in 0..number_of_dimensions {
            read.read_exact(&mut buffer).unwrap();
            let dimension = i32::from_le_bytes(buffer);
            println!("{:?}", dimension);
        }

        panic!("Not implemented");
    }
}

fn main() {
    // open the file "gpt2.bin" and feed it to GPTConfig::from_bytestream
    let mut gpt = GPT::build_from_checkpoint_file("gpt2.bin");
    return;
    let gpt_config = GPTConfig::from_bytestream(&mut std::fs::File::open("gpt2.bin").unwrap());

    return;
    let _ = MLP::from_weight_file("mlp_weights.json");
    return;
    let mut position_encoder = PositionalEncoding::new(10, 10);    
    let embeddings_weight = Tensor::load_from_weight_file("./embedding.json");    
    
    let key_weights = Tensor::load_from_weight_file("./key.json");
    let value_weights = Tensor::load_from_weight_file("./value.json");
    let query_weights = Tensor::load_from_weight_file("./query.json");

    let key_bias = Tensor::load_from_weight_file("./key_bias.json");
    let value_bias = Tensor::load_from_weight_file("./value_bias.json");
    let query_bias = Tensor::load_from_weight_file("./query_bias.json");

    let key_linear = LinearLayer::from_weights_and_bias(key_weights, key_bias);
    let value_linear = LinearLayer::from_weights_and_bias(value_weights, value_bias);
    let query_linear = LinearLayer::from_weights_and_bias(query_weights, query_bias);

    let mut embeddings = Embedding::from_tensor(embeddings_weight);
    let test_input = Tensor::arange(0, 10, 1).reshape(vec![1, 10].into());
    

    let mut attention_head = AttentionHead::from_pretrained(query_linear, key_linear, value_linear);
    let output = attention_head.forward(&test_input);
    println!("{:?}", output.item());

    let mut decoder = DecoderOnlyTransformer::new( 10, 10, 10, 10, 10, 10);
    let output = decoder.forward(&test_input);
    println!("{:?}", output.item());
}