use poro::central::{get_equation, Indexable, Shape, Tensor};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::fs::read_to_string;

fn read_lines(filename: &str) -> Vec<String> {
    let mut result = Vec::new();

    for line in read_to_string(filename).unwrap().lines() {
        result.push(line.to_string())
    }

    result
}

fn build_dataset_from_subset(
    words: &[String],
    stoi: &HashMap<char, usize>,
) -> (Vec<[usize; 3]>, Vec<usize>) {
    let mut xs = vec![];
    let mut ys = vec![];
    for word in words {
        let fixed = String::from("...") + word + ".";
        let chars: Vec<char> = fixed.chars().collect();
        for i in 0..chars.len() - 3 {
            let pair = (chars[i], chars[i + 1], chars[i + 2], chars[i + 3]);
            xs.push([stoi[&pair.0], stoi[&pair.1], stoi[&pair.2]]);
            ys.push(stoi[&pair.3]);
        }
    }
    (xs, ys)
}

fn main() {
    // let mut times = HashMap::new();

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
    let n2 = (names.len() as f32 * 0.9f32) as usize;
    let (xtr, ytr) = build_dataset_from_subset(&names[..n1], &stoi);
    let (_xdev, _ydev) = build_dataset_from_subset(&names[n1..n2], &stoi);
    let (_cte, _yte) = build_dataset_from_subset(&names[n2..], &stoi);

    let mut c = Tensor::load_from_weight_file("./data/bigram/tensor_C.json");

    c.set_requires_grad(true);
    let mut w1 = Tensor::load_from_weight_file("./data/bigram/tensor_W1.json");
    w1.set_requires_grad(true);
    let mut b1 = Tensor::load_from_weight_file("./data/bigram/tensor_b1.json");
    b1.set_requires_grad(true);
    let mut w2 = Tensor::load_from_weight_file("./data/bigram/tensor_W2.json");
    w2.set_requires_grad(true);
    let mut b2 = Tensor::load_from_weight_file("./data/bigram/tensor_b2.json");
    b2.set_requires_grad(true);

    const EPOCH_COUNT: usize = 10;
    let BATCH_SIZE: usize = 32; //xtr.len();
    let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![BATCH_SIZE, 3]));

    for epoch in 0..1 {
        println!("Epoch: {:?}", epoch);
        {
            let mut singleton = get_equation();
            singleton.zero_all_grads();
        }

        for b in 0..BATCH_SIZE {
            test_index_tensor.set_index([b, 0].into(), vec![xtr[b][0] as f32].into());
            test_index_tensor.set_index([b, 1].into(), vec![xtr[b][1] as f32].into());
            test_index_tensor.set_index([b, 2].into(), vec![xtr[b][2] as f32].into());
        }

        let test = c.view(Indexable::FromTensor(test_index_tensor.tensor_id));
        let reshape = test.reshape(Shape::new(vec![BATCH_SIZE, 30]));
        //println!("test.shape: {:?}", reshape.shape);
        //println!("w1.shape: {:?}", w1.shape);
        let test_mult = reshape << w1;
        let test_add = test_mult + b1;
        let test_tanh = test_add.tanh();
        let test_output_ = test_tanh << w2;
        let test_output = test_output_ + b2;

        let test_max = test_output.max(1);
        let test_counts = (test_output - test_max).exp();
        let test_counts_sum = test_counts.sum(1);

        let test_counts_sum_inverted = test_counts_sum.pow(-1.0);
        let test_probabilities = test_counts * test_counts_sum_inverted;

        let mut test_ytrue_onehot = Tensor::element(Shape::new(vec![BATCH_SIZE, 27]), 0.0);
        for b in 0..BATCH_SIZE {
            test_ytrue_onehot.set_index([b, ytr[b]].into(), vec![1.0].into());
        }

        let test_prob_log = test_probabilities.log();

        let test_presum = test_ytrue_onehot * test_prob_log;
        let test_sum = (-test_presum).sum(1);
        let test_mean = test_sum.mean(vec![0]);

        println!("Loss: {:?}", test_mean.item());
        test_mean.backward();
        println!("test_output.grad: {:?}", w1.grad());
        {
            let mut singleton = get_equation();
            singleton.update_parameters(-0.1);
        }
    }
}
