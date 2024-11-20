use crate::nn::{Module, LinearLayer, LinearLayerConfig};
use std::io::{BufWriter, Write};
use std::fs::OpenOptions;
use crate::central::Tensor;
use log::info;

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
pub struct CasualSelfAttentionConfig {
    embedding_dim: usize,
}

pub struct CasualSelfAttention {
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
        info!("CasualSelfAttention forward");
        info!("Query start");
        let b = x.shape.indices[0];
        let t = x.shape.indices[1];
        let c = x.shape.indices[2];
        println!("{:?}", x.shape);


        let num_heads = 12;
       
        let query = self.query_attention.forward(&x);
        let key = self.key_attention.forward(&x);
        let value = self.value_attention.forward(&x);




        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Query");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &query.item().into_raw_vec());

        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Key");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &key.item().into_raw_vec());

        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Value");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &value.item().into_raw_vec());

        
        let query = query.reshape(vec![b, t, num_heads, c / num_heads].into()).tranpose_with_provided_axis(1, 2);
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$QueryT");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &query.item().into_raw_vec());
        let key = key.reshape(vec![b, t, num_heads, c / num_heads].into()).tranpose_with_provided_axis(1, 2);
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$KeyT");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &key.item().into_raw_vec());
        let value = value.reshape(vec![b, t, num_heads, c / num_heads].into()).tranpose_with_provided_axis(1, 2);
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$ValueT");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &value.item().into_raw_vec());

        let key_super_tranposed = key.tranpose_with_provided_axis(2, 3);
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$KeyST");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &key_super_tranposed.item().into_raw_vec());
        let query_key = query << key_super_tranposed;
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$QueryKey");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &query_key.item().into_raw_vec());
        let denom = 1.0 / (key.shape.indices[key.shape.number_of_indices - 1] as f32).sqrt();
        let attn_weights = query_key * denom;
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$AttnWeights");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &attn_weights.item().into_raw_vec());
        let mask = Tensor::tril(vec![t, t].into()).reshape(vec![1, 1, t, t].into());

        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Premask");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &mask.item().into_raw_vec());
        println!("Attn weights {:?}", attn_weights.shape);
        let mask_broadcasted = mask.broadcast(vec![b, num_heads, t, t].into());
        let filled = attn_weights.masked_fill(&mask_broadcasted, std::f32::NEG_INFINITY);
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Filled");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &filled.item().into_raw_vec());
        let attn_weights = filled.softmax(attn_weights.shape.number_of_indices - 1);
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$Softmax");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &attn_weights.item().into_raw_vec());
        let attn_output = attn_weights << value;
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$AttnOutput");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &attn_output.item().into_raw_vec());
        let attn_output = attn_output.tranpose_with_provided_axis(1, 2).reshape(vec![b, t, c].into());
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$AttnOutputReshape");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &attn_output.item().into_raw_vec());
        let x = self.c_proj.forward(&attn_output);
        let _ = write_string_vector_to_file("./rust_checkfile.txt", "$CProj");
        let _ = write_f32_vector_to_file("./rust_checkfile.txt", &x.item().into_raw_vec());
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

mod tests {
    use std::vec;
    use crate::nn::layers::linear::LinearLayerConfig;
    use crate::nn::layers::module::Module;
    use crate::central::Tensor;
    use crate::Shape;

    #[test]
    fn test_casual_self_attention() {
        let file_names = vec![
            "./data/tests/causal_self_attention/causal_self_attention_query_weights.txt",
            "./data/tests/causal_self_attention/causal_self_attention_query_bias.txt",
            "./data/tests/causal_self_attention/causal_self_attention_key_weights.txt", 
            "./data/tests/causal_self_attention/causal_self_attention_key_bias.txt",
            "./data/tests/causal_self_attention/causal_self_attention_value_weights.txt",
            "./data/tests/causal_self_attention/causal_self_attention_value_bias.txt",
            "./data/tests/causal_self_attention/causal_self_attention_c_proj_weights.txt", 
            "./data/tests/causal_self_attention/causal_self_attention_c_proj_bias.txt",
            "./data/tests/causal_self_attention/fake_output.txt",
            "./data/tests/causal_self_attention/loss.txt",
            "./data/tests/causal_self_attention/fake_input.txt",
            "./data/tests/causal_self_attention/expected_output.txt",
            "./data/tests/causal_self_attention/causal_self_attention_c_proj_weights_grad.txt",
            "./data/tests/causal_self_attention/casual_self_attention_c_proj_bias_grad.txt",
            "./data/tests/causal_self_attention/causal_self_attention_query_weights_grad.txt",
            "./data/tests/causal_self_attention/causal_self_attention_query_bias_grad.txt",
            "./data/tests/causal_self_attention/causal_self_attention_key_weights_grad.txt",
            "./data/tests/causal_self_attention/causal_self_attention_key_bias_grad.txt",
            "./data/tests/causal_self_attention/causal_self_attention_value_weights_grad.txt",
            "./data/tests/causal_self_attention/causal_self_attention_value_bias_grad.txt"
        ];

        let mut tenors = vec![];

        for file_name in file_names.iter() {
            println!("Reading file {}", file_name);
            let mut file = std::fs::File::open(file_name).unwrap();
            let t = crate::central::Tensor::from_bytestream(&mut file, false).unwrap();
            tenors.push(t);
        }

        let mut casual_self_attention = super::CasualSelfAttention::from_weights_and_bias(
            tenors[0].clone(),
            tenors[1].clone(),
            tenors[2].clone(),
            tenors[3].clone(),
            tenors[4].clone(),
            tenors[5].clone(),
            tenors[6].clone(),
            tenors[7].clone(),
        );

        let input = tenors[10].clone();
        let output = casual_self_attention.forward(&input);
        
        let output_flatten = output.item().into_raw_vec();
        let expected_output_flatten = tenors[11].item().into_raw_vec();

        for (i, (o, e)) in output_flatten.iter().zip(expected_output_flatten.iter()).enumerate() {
            assert!((o - e).abs() < 1e-6, "Mismatch at index {}", i);
        }

        let mse_loss = (output - tenors[8]).pow(2.0).reshape(vec![1024 * 768].into()).mean(vec![0]);

        let expected_loss_flatten = tenors[9].item().into_raw_vec();
        let loss_flatten = mse_loss.item().into_raw_vec();


        for (i, (l, e)) in loss_flatten.iter().zip(expected_loss_flatten.iter()).enumerate() {
            assert!((l - e).abs() < 1e-4, "Mismatch at index {}", i);
        }
        mse_loss.backward();

        let grad = casual_self_attention.c_proj.weights.grad();
        let expected_grad = tenors[12].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();
        const ERR : f32 = 1e-3;
        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }

        let grad = casual_self_attention.c_proj.bias.grad();
        let expected_grad = tenors[13].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();

        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }

        let grad = casual_self_attention.query_attention.weights.grad();
        let expected_grad = tenors[14].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();

        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }

        let grad = casual_self_attention.query_attention.bias.grad();
        let expected_grad = tenors[15].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();

        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }

        let grad = casual_self_attention.key_attention.weights.grad();
        let expected_grad = tenors[16].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();

        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }

        let grad = casual_self_attention.key_attention.bias.grad();
        let expected_grad = tenors[17].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();

        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }

        let grad = casual_self_attention.value_attention.weights.grad();
        let expected_grad = tenors[18].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();

        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }

        let grad = casual_self_attention.value_attention.bias.grad();
        let expected_grad = tenors[19].clone();
        let grad_flatten = grad.into_raw_vec();
        let expected_grad_flatten = expected_grad.item().into_raw_vec();

        for (i, (g, e)) in grad_flatten.iter().zip(expected_grad_flatten.iter()).enumerate() {
            assert!((g - e).abs() < ERR, "Mismatch at index {}  a {} b {}", i, g, e);
        }


    }
}