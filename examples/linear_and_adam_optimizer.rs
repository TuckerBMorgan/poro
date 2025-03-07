use poro::{get_equation, nn::{LinearLayer, LinearLayerConfig}, Adam, AdamBuilder, AdamW, AdamWBuilder, Optimizer};
use poro::nn::Module;
use poro::Tensor;

fn main() {
    let read = &mut std::fs::File::open("./data/tests/adam_optimizer/weight.txt").unwrap();   
    let weights_from_txt_file = Tensor::from_bytestream(
        read,
        false
    ).unwrap();


    let read  = &mut std::fs::File::open("./data/tests/adam_optimizer/bias.txt").unwrap();
    let bias_from_txt_file = Tensor::from_bytestream(
        read,
        false
    ).unwrap();

    let read  = &mut std::fs::File::open("./data/tests/adam_optimizer/input.txt").unwrap();
    let input_from_txt_file = Tensor::from_bytestream(
        read,
        false
    ).unwrap();

    let read  = &mut std::fs::File::open("./data/tests/adam_optimizer/fake_output.txt").unwrap();
    let fake_output_from_txt_file = Tensor::from_bytestream(
        read,
        false
    ).unwrap();

    let mut lienar = LinearLayer::from_weights_and_bias(weights_from_txt_file, bias_from_txt_file);

    let adam_builder_default = AdamWBuilder::default();
    let mut adam = AdamW::new(adam_builder_default);
    
    adam.record_parameters(&lienar);
    lienar.set_requires_grad(true);

   // println!("weights: {:?}", lienar.weights.item());
    for _ in 0..10 {
        get_equation().zero_all_grads();
        let output = lienar.forward(&input_from_txt_file);
        let loss = output - fake_output_from_txt_file;
        let loss = loss.sum(0);//.sum(1);
        loss.backward();

        adam.step(&mut lienar);
    }
    println!("weights: {:?}", lienar.weights.item());
}