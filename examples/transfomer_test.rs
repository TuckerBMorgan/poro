
use poro::nn::model::DecoderOnlyTransformer;
use poro::central::Tensor;
use poro::nn::{AttentionHead, Embedding, LinearLayer, Model, Module, MLP, LayerNorm, CasualSelfAttention, CasualSelfAttentionConfig};
use poro::nn::model::PositionalEncoding;
use poro::nn::layers::NewGLU;
use std::io::{Read, Seek};
use std::convert::TryInto;
use simplelog::*;
use log::info;

use std::io::{BufWriter, Write};
use std::fs::OpenOptions;

use tokenizers::tokenizer::{Result, Tokenizer};
fn write_f32_vector_to_file(path: &str, data: &[f32]) -> std::io::Result<()> {
    // Create or open the file
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    // Create a buffered writer to improve performance
    let mut writer = BufWriter::new(file);
    
    // Write each f32 value to the file
    for &value in data {
        // Write the float value as a string
        writeln!(writer, "{}", value)?;
    }
    
    // Ensure all data is flushed to the file
    writer.flush()?;
    
    Ok(())
}

fn write_string_vector_to_file(path: &str, data: &str) -> std::io::Result<()> {
    // Create or open the file
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    // Create a buffered writer to improve performance
    let mut writer = BufWriter::new(file);

        // Write the float value as a string
        writeln!(writer, "{}", data)?;
    
    // Ensure all data is flushed to the file
    writer.flush()?;
    
    Ok(())
}

struct Block {
    ln_1: LayerNorm,
    attn: CasualSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {

    pub fn from_components(ln1: LayerNorm, ln2: LayerNorm, attn: CasualSelfAttention, mlp: MLP) -> Self {
        Block {
            ln_1: ln1,
            ln_2: ln2,
            attn,
            mlp,
        }
    }
}

impl Module for Block {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let y = self.ln_1.forward(x);
                // Create or open the file

 //       let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Linear_1");
 //       let _ = write_f32_vector_to_file("./rust_checkfile.txt", &y.item().into_raw_vec());
        info!("LayerNorm 1 done");
        
        let y = self.attn.forward(&y);
    //     let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Attn");
    //    let _ = write_f32_vector_to_file("./rust_checkfile.txt", &y.item().into_raw_vec());

        info!("Attention done");
        let x = *x + y;
 
        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Residual");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &y.item().into_raw_vec());

        info!("Add done");
        let y = self.ln_2.forward(&x);
 
        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Linear_2");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &y.item().into_raw_vec());

        info!("LayerNorm 2 done");
        let y = self.mlp.forward(&y);


        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$MLP");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &y.item().into_raw_vec());

        info!("MLP done");
        let x = x + y;

        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Residual_2");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &x.item().into_raw_vec());

        info!("Add done");
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
        //println!("{:?}", buffer);
        // version
        reader.read_exact(&mut buffer).unwrap();
        //println!("{:?}", buffer);

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
    final_layer_norm: LayerNorm,
}

impl GPT {
    pub fn build_from_checkpoint_file(filepath: &str) -> GPT {
        let read = &mut std::fs::File::open(filepath).unwrap();   
        // feed 256 bytes to GPTConfig::from_bytestream
        let mut buffer = [0; 256 * 4];

        read.read_exact(&mut buffer).unwrap();

        let gpt_config = GPTConfig::from_bytestream(&mut buffer.as_ref());

        info!("Reading in wte weights");
        let wte_weights = Tensor::from_bytestream(read, false).unwrap();

        info!("Reading in wpe weights");
        let wpe_weights = Tensor::from_bytestream(read, false).unwrap();

        let mut blocks = vec![];
        for _ in 0..gpt_config.number_of_layers {
            blocks.push(vec![]);
            
        }

        info!("Reading in block weights");
        for i in 0..gpt_config.number_of_layers {
            let block_ln_1_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(block_ln_1_weights);
        }
        info!("Reading in block biases");
        for i in 0..gpt_config.number_of_layers { 
            let block_ln_1_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(block_ln_1_bias);
        }


        info!("Reading in q weights");
        for i in 0..gpt_config.number_of_layers {
            let q_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(q_weights);
        }

        info!("Reading in k weights");
        for i in 0..gpt_config.number_of_layers {
            let k_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(k_weights);
        }

        info!("Reading in v weights");
        for i in 0..gpt_config.number_of_layers {
            let v_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(v_weights);
        }

        info!("Reading in q biases");
        for i in 0..gpt_config.number_of_layers {
            let q_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(q_bias);
        }

        info!("Reading in k biases");
        for i in 0..gpt_config.number_of_layers {
            let k_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(k_bias);
        }
        
        info!("Reading in v biases");
        for i in 0..gpt_config.number_of_layers {
            let v_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(v_bias);
        }

        info!("Reading in c_proj weights");
        for i in 0..gpt_config.number_of_layers {
            let c_proj_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(c_proj_weights);
        }

        info!("Reading in c_proj biases");
        for i in 0..gpt_config.number_of_layers {
            let c_proj_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(c_proj_bias);
        }

        info!("Reading in block weights");
        for i in 0..gpt_config.number_of_layers {
            let block_ln_2_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(block_ln_2_weights);
        }
        info!("Reading in block biases");
        for i in 0..gpt_config.number_of_layers {
            let block_ln_2_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(block_ln_2_bias);
        }

        info!("MLP weights");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_fc_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(mlp_c_fc_weights);
        }

        info!("MLP biases");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_fc_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(mlp_c_fc_bias);
        }

        info!("MLP weights");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_proj_weights = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(mlp_c_proj_weights);
        }

        info!("MLP biases");
        for i in 0..gpt_config.number_of_layers {
            let mlp_c_proj_bias = Tensor::from_bytestream(read, false).unwrap();
            blocks[i].push(mlp_c_proj_bias);
        }

        info!("Reading in final layer norm weights");
        let final_layer_norm_weights = Tensor::from_bytestream(read, false).unwrap();

        info!("Reading in final layer norm bias");
        let final_layer_norm_bias = Tensor::from_bytestream(read, false).unwrap();

        let wte = Embedding::from_tensor(wte_weights);
        let wpe = Embedding::from_tensor(wpe_weights);

        let mut blocks_vec = vec![];
        for i in 0..gpt_config.number_of_layers {
            let ln_1 = LayerNorm::from_weights_and_bias(blocks[i][0].clone(), blocks[i][1].clone());

            let attn = CasualSelfAttention::from_weights_and_bias(
                    blocks[i][2].clone(), blocks[i][5].clone(), 
                    blocks[i][3].clone(), blocks[i][6].clone(),
                    blocks[i][4].clone(), blocks[i][7].clone(),
                blocks[i][8].clone(), blocks[i][9].clone()
        );

            let ln_2 = LayerNorm::from_weights_and_bias(blocks[i][10].clone(), blocks[i][11].clone());

            let mlp = MLP::from_weights_and_bias(blocks[i][12].clone(), blocks[i][13].clone(),
                                                     blocks[i][14].clone(), blocks[i][15].clone());
            let block = Block::from_components(ln_1, ln_2, attn, mlp);
            blocks_vec.push(block);
        }

        let final_layer_norm = LayerNorm::from_weights_and_bias(final_layer_norm_weights, final_layer_norm_bias);

        GPT {
            wte,
            wpe,
            blocks: blocks_vec,
            final_layer_norm,
        }
    }

}

impl Model for GPT {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        info!("GPT forward");

        let toks = self.wte.forward(x);

        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Toks");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &toks.item().into_raw_vec());

        info!("WTE done");
        let pos_arange = Tensor::arange(0, x.shape.indices[1], 1).reshape(vec![1, x.shape.indices[1]].into());
        let pos = self.wpe.forward(&pos_arange);



        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Pos");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &pos.item().into_raw_vec());

        info!("WPE done");
        let mut x = toks + pos;

        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$TokPos");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &x.item().into_raw_vec());
        info!("Add done");
        let mut counts = 0;
        for block in self.blocks.iter_mut() {
            info!("Block {}", counts);
            x = block.forward(&x);
            counts += 1;
        }



        info!("Blocks done");
        let x = self.final_layer_norm.forward(&x);

        //let _ = write_string_vector_to_file("./rust_checkfile.txt", "$LN");
        //let _ = write_f32_vector_to_file("./rust_checkfile.txt", &x.item().into_raw_vec());

        info!("Final layer norm done");
        let mut test = LinearLayer::from_weights_and_bias(self.wte.tensor.clone(), Tensor::zeroes(Shape::new(vec![0].into())));

        let x = test.forward(&x);

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

fn u8_to_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2) // Split into chunks of 2 bytes
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]])) // Convert to u16
        .collect() // Collect into a Vec<u16>
}

use poro::{update_parameters, Shape};
use rand::Rng;
struct DataLoader {
    tokens: Vec<u16>,
    batch_size: usize,
    seq_length: usize,
    current_position: usize
}
impl DataLoader {
    pub fn new(batch_size: usize, seq_length: usize) -> Self {
        // load the file at "./data/tiny_shakespeare/tiny_shakespeare_train.bin"

        //read the header which is 256 * 4 bytes
        let mut buffer = [0; 256 * 4];
        let read = &mut std::fs::File::open("./data/tinyshakespeare/tiny_shakespeare_val.bin").unwrap();
        read.read_exact(&mut buffer).unwrap();

        // read the rest into a tensor
        let mut vec_buffer = Vec::new();
        let  as_vec = read.read_to_end(&mut vec_buffer).unwrap();



        let vec_buffer = u8_to_u16(&vec_buffer);
        // combine the u8 into u16s
        




        DataLoader {
            tokens: vec_buffer,
            batch_size,
            seq_length,
            current_position: 0
        }
    }

    pub fn advance(&mut self) {
        self.current_position += self.seq_length;
        if self.current_position + self.seq_length >= self.tokens.len() {
            self.current_position = 0;
        }
    }
    
    pub fn next_batch(&mut self) -> (Tensor, Tensor) {
        let mut input_tensor = Tensor::zeroes(Shape::new(vec![self.batch_size, self.seq_length].into()));
        let mut target_tensor = Tensor::zeroes(Shape::new(vec![self.batch_size, self.seq_length].into()));
    
        for i in 0..self.batch_size {
            // Calculate the starting position for the current batch item
            let batch_start = self.current_position + i * self.seq_length;
    
            // Ensure the slices respect the batch's position
            let input_slice = &self.tokens[batch_start..batch_start + self.seq_length];
            let target_slice = &self.tokens[batch_start + 1..batch_start + self.seq_length + 1];
    
            for j in 0..self.seq_length {
                input_tensor.set_index([i, j].into(), vec![input_slice[j] as f32].into());
                target_tensor.set_index([i, j].into(), vec![target_slice[j] as f32].into());
            }
        }
    
        // Update the current position by the total tokens consumed
        self.current_position += self.batch_size * self.seq_length;
    
        // Handle wrap-around if the end of tokens is reached
        if self.current_position + (self.batch_size * self.seq_length) >= self.tokens.len() {
            self.advance();
        }
    
        (input_tensor, target_tensor)
    }
    
}
use std::fs::File;
use std::path::Path;


fn get_learning_rate(iteration: usize) -> f32 {
    let num_iterations = 20;
    let learning_rate =1e-4f32;
    let learning_rate_decay = 1.0f32;
    let warumup_iterations = 10;
    let min_learning_rate = learning_rate * learning_rate_decay;

    if iteration < warumup_iterations {
        return learning_rate * (iteration as f32 / warumup_iterations as f32);
    } 

    if iteration < num_iterations {
        return min_learning_rate;
    }

    let decay_ratio = (iteration - warumup_iterations) as f32 / (num_iterations as f32 - warumup_iterations as f32);

    assert!(decay_ratio >= 0.0 && decay_ratio <= 1.0);

    let coeff = 0.5 * (1.0 + f32::cos(std::f32::consts::PI * decay_ratio));
    return min_learning_rate + (learning_rate - min_learning_rate) * coeff;
}


fn main() {
    WriteLogger::init(
        LevelFilter::Info, // Set the log level
        Config::default(), // Use the default configuration
        File::create(Path::new("Transfomer_test.log")).unwrap(), // Create or open the log file
    ).unwrap();

    let vocab_size = 50304;
    let batch_size = 1;
    let seq_length = 64;
    let mut data_loead = DataLoader::new(batch_size, seq_length);

    let (x, y) = data_loead.next_batch();

    let mut one_hot_encoded_trues = Tensor::zeroes(Shape::new(vec![batch_size, seq_length, 50257].into()));
    let y = y.item();

    for i in 0..batch_size {
        for j in 0..seq_length {
            let index = y[[i, j]] as usize;
            one_hot_encoded_trues.set_index([i, j, index].into(), vec![1.0].into());
        }
    }

    let one_hot_encoded_trues = one_hot_encoded_trues.reshape(vec![batch_size *  seq_length, 50257].into());

    let mut gpt = GPT::build_from_checkpoint_file("./gpt2.bin");
    let test_ouput = gpt.forward(&x);
    println!("{:?}", gpt.wte.tensor.shape);
    panic!("");

    let test_output = test_ouput.reshape(vec![batch_size * seq_length, 50257].into());

    let loss = test_output.cross_entropy_loss(one_hot_encoded_trues);
    loss.backward();
    println!("{:?}", gpt.wte.tensor.grad());
    let mut index = 0;
    for row in gpt.wte.tensor.grad().rows() {
        if index == 11 {
            println!("{:?}", row);
        }
        index += 1;
    }
    panic!("");

    return;

    //println!("{:?}", tokenizer.encode("Hello, how are you?", false).unwrap());
    return;

    let file_path = "./rust_checkfile.txt";

    // Open the file with write mode to truncate it (clear contents)
    let mut file = OpenOptions::new().write(true).truncate(true).open(file_path).unwrap();

    // Optionally, you can write something to the file, here it's just empty
    file.write_all(b"").unwrap();

    let test_input_file_path = "./test_input.bin";
    let test_input = Tensor::from_bytestream(&mut File::open(test_input_file_path).unwrap(), false).unwrap();
    
    println!("{:?}", test_input.shape);
    // open the file "gpt2.bin" and feed it to GPTConfig::from_bytestream

    let input = Tensor::randn(vec![1, 1, 1].into());
    let output = gpt.forward(&test_input);
    return;
    let gpt_config = GPTConfig::from_bytestream(&mut std::fs::File::open("gpt2.bin").unwrap());

    return;
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