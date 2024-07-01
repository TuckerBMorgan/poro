use poro::central::{get_equation, Indexable, Shape, Tensor};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::time::Instant;

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

fn add_time(timings: &mut HashMap<String, u128>, operation: &str, now: Instant) {
    if !timings.contains_key(&operation.to_string()) {
        timings.insert(operation.to_string(), 0);
    }
    let elapsed = now.elapsed().as_micros();
    let current_time = timings.get(&operation.to_string()).unwrap();
    timings.insert(operation.to_string(), current_time + elapsed);
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
    //1. Copy the weights and try it in an isolated test
    //2. Try offsetting before I push them into the gpu
    //3. upgrade my cuda to 12.5
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

    const EPOCH_COUNT: usize = 25;
    let batch_size: usize = xtr.len();
    let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![batch_size, 3]));

    let mut timings = HashMap::new();

    for epoch in 0..EPOCH_COUNT {
        let now = Instant::now();
        let timeFrame = Instant::now();
        println!("Epoch: {:?}", epoch);
        {
            let mut singleton = get_equation();
            singleton.zero_all_grads();
        }
        add_time(&mut timings, "Zero", now);
        let now = Instant::now();
        for b in 0..batch_size {
            test_index_tensor.set_index([b, 0].into(), vec![xtr[b][0] as f32].into());
            test_index_tensor.set_index([b, 1].into(), vec![xtr[b][1] as f32].into());
            test_index_tensor.set_index([b, 2].into(), vec![xtr[b][2] as f32].into());
        }

        let test = c.view(Indexable::FromTensor(test_index_tensor.tensor_id));
        let reshape = test.reshape(Shape::new(vec![batch_size, 30]));
        add_time(&mut timings, "Data Fill", now);
        let now = Instant::now();
        let test_mult = reshape << w1;
        add_time(&mut timings, "Matmul", now);

        let now = Instant::now();
        let test_add = test_mult + b1;
        add_time(&mut timings, "Add", now);
        let now = Instant::now();
        let test_tanh = test_add.tanh_mapped();
        add_time(&mut timings, "Tanh", now);

        let now = Instant::now();
        let test_output_ = test_tanh << w2;
        add_time(&mut timings, "Matmul", now);
        let now = Instant::now();
        let test_output = test_output_ + b2;
        add_time(&mut timings, "Add", now);

        let now = Instant::now();
        let test_max = test_output.max(1);
        let test_counts = (test_output - test_max).exp();
        let test_counts_sum = test_counts.sum(1);
        let test_counts_sum_inverted = test_counts_sum.pow(-1.0);
        let test_probabilities = test_counts * test_counts_sum_inverted;
        add_time(&mut timings, "Softmax", now);

        let now = Instant::now();
        let mut test_ytrue_onehot = Tensor::element(Shape::new(vec![batch_size, 27]), 0.0);
        for b in 0..batch_size {
            test_ytrue_onehot.set_index([b, ytr[b]].into(), vec![1.0].into());
        }

        let test_prob_log = test_probabilities.log();
        let test_presum = test_ytrue_onehot * test_prob_log;
        let test_sum = (-test_presum).sum(1);
        let test_mean = test_sum.mean(vec![0]);
        add_time(&mut timings, "Loss", now);

        println!("Loss: {:?}", test_mean.item());
        let now = Instant::now();
        test_mean.backward();
        add_time(&mut timings, "Backward", now);
        let now = Instant::now();
        {
            let mut singleton = get_equation();
            singleton.update_parameters(-0.1);
        }
        add_time(&mut timings, "Update", now);
        println!("Time Frame: {:?}", timeFrame.elapsed().as_micros());
    }
}
