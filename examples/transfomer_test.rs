
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

    pub fn from_weights_and_bias(query_weights: Tensor, query_bias: Tensor, key_weights: Tensor, key_bias: Tensor, value_weights: Tensor, value_bias: Tensor, c_proj_weights: Tensor, c_proj_bias: Tensor) -> Self {
        let query_attention = LinearLayer::from_weights_and_bias(query_weights, query_bias);
        let key_attention = LinearLayer::from_weights_and_bias(key_weights, key_bias);
        let value_attention = LinearLayer::from_weights_and_bias(value_weights, value_bias);
        let c_proj = LinearLayer::from_weights_and_bias(c_proj_weights, c_proj_bias);

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

    pub fn from_weights_and_bias(c_fc_weights: Tensor, c_fc_bias: Tensor, c_proj_weights: Tensor, c_proj_bias: Tensor) -> Self {
        let c_fc = LinearLayer::from_weights_and_bias(c_fc_weights, c_fc_bias);
        let c_proj = LinearLayer::from_weights_and_bias(c_proj_weights, c_proj_bias);
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

    pub fn from_weights_and_bias(ln_1_weights: Tensor, ln_1_bias: Tensor, ln_2_weights: Tensor, ln_2_bias: Tensor, query_weights: Tensor, query_bias: Tensor, key_weights: Tensor, key_bias: Tensor, value_weights: Tensor, value_bias: Tensor, c_proj_weights: Tensor, c_proj_bias: Tensor, mlp_c_fc_weights: Tensor, mlp_c_fc_bias: Tensor, mlp_c_proj_weights: Tensor, mlp_c_proj_bias: Tensor) -> Self {
        let ln_1 = LinearLayer::from_weights_and_bias(ln_1_weights, ln_1_bias);
        let ln_2 = LinearLayer::from_weights_and_bias(ln_2_weights, ln_2_bias);
        let attn = CasualSelfAttention::from_weights_and_bias(query_weights, query_bias, key_weights, key_bias, value_weights, value_bias, c_proj_weights, c_proj_bias);
        let mlp = MLP::from_weights_and_bias(mlp_c_fc_weights, mlp_c_fc_bias, mlp_c_proj_weights, mlp_c_proj_bias);

        Block {
            ln_1,
            ln_2,
            attn,
            mlp,
            embedding_dim: 768,
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
    wpe: Embedding,
    blocks: Vec<Block>,
    final_layer_norm: LinearLayer,
}

impl GPT {
    pub fn build_from_checkpoint_file(filepath: &str) -> GPT {
        let read = &mut std::fs::File::open(filepath).unwrap();   
        // feed 256 bytes to GPTConfig::from_bytestream
        let mut buffer = [0; 256 * 4];

        read.read_exact(&mut buffer).unwrap();

        let gpt_config = GPTConfig::from_bytestream(&mut buffer.as_ref());

        println!("Reading in wte weights");
        let wte_weights = Tensor::from_bytestream(read).unwrap();

        println!("Reading in wpe weights");
        let wpe_weights = Tensor::from_bytestream(read).unwrap();

        let mut blocks = vec![];
        for _ in 0..gpt_config.number_of_layers {
            blocks.push(vec![]);
            
        }

        println!("Reading in block weights");
        for i in 0..gpt_config.number_of_layers {
            let block_ln_1_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(block_ln_1_weights);
        }
        println!("Reading in block biases");
        for i in 0..gpt_config.number_of_layers { 
            let block_ln_1_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(block_ln_1_bias);
        }


        println!("Reading in q weights");
        for i in 0..gpt_config.number_of_layers {
            let q_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(q_weights);
        }

        println!("Reading in k weights");
        for i in 0..gpt_config.number_of_layers {
            let k_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(k_weights);
        }

        println!("Reading in v weights");
        for i in 0..gpt_config.number_of_layers {
            let v_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(v_weights);
        }

        println!("Reading in q biases");
        for i in 0..gpt_config.number_of_layers {
            let q_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(q_bias);
        }

        println!("Reading in k biases");
        for i in 0..gpt_config.number_of_layers {
            let k_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(k_bias);
        }

        println!("Reading in v biases");
        for i in 0..gpt_config.number_of_layers {
            let v_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(v_bias);
        }

        println!("Reading in c_proj weights");
        for i in 0..gpt_config.number_of_layers {
            let c_proj_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(c_proj_weights);
        }

        println!("Reading in c_proj biases");
        for i in 0..gpt_config.number_of_layers {
            let c_proj_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(c_proj_bias);
        }

        println!("Reading in block weights");
        for i in 0..gpt_config.number_of_layers {
            let block_ln_2_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(block_ln_2_weights);
        }
        println!("Reading in block biases");
        for i in 0..gpt_config.number_of_layers {
            let block_ln_2_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(block_ln_2_bias);
        }

        println!("MLP weights");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_fc_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(mlp_c_fc_weights);
        }

        println!("MLP biases");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_fc_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(mlp_c_fc_bias);
        }

        println!("MLP weights");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_proj_weights = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(mlp_c_proj_weights);
        }

        println!("MLP biases");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_proj_bias = Tensor::from_bytestream(read).unwrap();
            blocks[i].push(mlp_c_proj_bias);
        }

        println!("Reading in final layer norm weights");
        let final_layer_norm_weights = Tensor::from_bytestream(read).unwrap();

        println!("Reading in final layer norm bias");
        let final_layer_norm_bias = Tensor::from_bytestream(read).unwrap();

        let wte = Embedding::from_tensor(wte_weights);
        let wpe = Embedding::from_tensor(wpe_weights);

        let mut blocks = blocks.iter().map(|x| Block::from_weights_and_bias(x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone(), x[4].clone(), x[5].clone(), x[6].clone(), x[7].clone(), x[8].clone(), x[9].clone(), x[10].clone(), x[11].clone(), x[12].clone(), x[13].clone(), x[14].clone(), x[15].clone())).collect::<Vec<Block>>();

        let final_layer_norm = LinearLayer::from_weights_and_bias(final_layer_norm_weights, final_layer_norm_bias);

        GPT {
            wte,
            wpe,
            blocks,
            final_layer_norm,
        }
    }

}

impl Model for GPT {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let toks = self.wte.forward(x);
        let pos = self.wpe.forward(x);
        let mut x = toks + pos;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x);
        }
        let x = self.final_layer_norm.forward(&x);
        x
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.extend(self.wte.get_parameters());
        parameters.extend(self.wpe.get_parameters());
        for block in self.blocks.iter() {
            parameters.extend(block.get_parameters());
        }
        parameters.extend(self.final_layer_norm.get_parameters());
        parameters
    }
}

fn main() {
    // open the file "gpt2.bin" and feed it to GPTConfig::from_bytestream
    let mut gpt = GPT::build_from_checkpoint_file("gpt2.bin");
    let input = Tensor::randn(vec![1, 1, 1].into());
    let output = gpt.forward(&input);
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