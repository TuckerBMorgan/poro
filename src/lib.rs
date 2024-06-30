#![feature(unboxed_closures)]
#![feature(fn_traits)]

pub mod central;
pub mod nn;
pub use central::*;
pub use ndarray::prelude::*;

#[cfg(test)]
mod tests {

    use crate::nn::*;
    use crate::*;
    use rand::Rng;
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn basic_pow_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a.pow(2.0);
        let result = b.item();
        assert!(result == arr2(&[[4.0]]).into_dyn());
    }

    #[test]
    fn backward_pow_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a.pow(2.0);
        b.backward();
        let result = b.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[4.0]]).into_dyn());
    }

    #[test]
    fn basic_exp_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a.exp();
        let result = b.item();
        assert!(result == arr2(&[[2.0f32.exp()]]).into_dyn());
    }

    #[test]
    fn backward_exp_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a.exp();
        b.backward();
        let result = b.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[2.0f32.exp()]]).into_dyn());
    }

    #[test]
    fn micrograd_copy_test() {
        let x1 = Tensor::element(Shape::new(vec![1]), 2.0);
        let x2 = Tensor::element(Shape::new(vec![1]), 0.0);

        let w1 = Tensor::element(Shape::new(vec![1]), -3.0);
        let w2 = Tensor::element(Shape::new(vec![1]), 1.0);

        let b = Tensor::element(Shape::new(vec![1]), 6.8813735870195432);

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let l = 2.0f32 * n;
        let e = l.exp();
        let o_1 = e - 1.0;
        let o_2 = e + 1.0;
        let o = o_1 / o_2;
        o.backward();

        let n_result = n.item();
        assert!(approx_equal(n_result[[0]], 0.8813734, 1e-6));
        let n_grad = n.grad();
        assert!(approx_equal(n_grad[[0]], 0.5, 1e-6));

        let b_result = b.item();
        assert!(approx_equal(b_result[[0]], 6.8813735870195432, 1e-6));
        let b_grad = b.grad();
        assert!(approx_equal(b_grad[[0]], 0.5, 1e-6));

        let x1_result = x1.item();
        assert!(approx_equal(x1_result[[0]], 2.0, 1e-6));
        let x1_grad = x1.grad();
        assert!(approx_equal(x1_grad[[0]], -1.5, 1e-6));

        let x1w1x2w2_result = x1w1x2w2.item();
        assert!(approx_equal(x1w1x2w2_result[[0]], -6.0, 1e-6));
        let x1w1x2w2_grad = x1w1x2w2.grad();
        assert!(approx_equal(x1w1x2w2_grad[[0]], 0.5, 1e-6));

        let x2w2_result = x2w2.item();
        assert!(approx_equal(x2w2_result[[0]], 0.0, 1e-6));
        let x2w2_grad = x2w2.grad();
        assert!(approx_equal(x2w2_grad[[0]], 0.5, 1e-6));

        let x1w1_result = x1w1.item();
        assert!(approx_equal(x1w1_result[[0]], -6.0, 1e-6));
        let x1w1_grad = x1w1.grad();
        assert!(approx_equal(x1w1_grad[[0]], 0.5, 1e-6));

        let w2_result = w2.item();
        assert!(approx_equal(w2_result[[0]], 1.0, 1e-6));
        let w2_grad = w2.grad();
        assert!(w2_grad[[0]] == 0.0);

        let x2_result = x2.item();
        assert!(approx_equal(x2_result[[0]], 0.0, 1e-6));
        let x2_grad = x2.grad();
        assert!(approx_equal(x2_grad[[0]], 0.5, 1e-6));

        let w1_result = w1.item();
        assert!(approx_equal(w1_result[[0]], -3.0, 1e-6));
        let w1_grad = w1.grad();
        assert!(approx_equal(w1_grad[[0]], 1.0, 1e-6));
    }

    use core::f32;

    use std::collections::HashMap;
    use std::fs::read_to_string;

    fn read_lines(filename: &str) -> Vec<String> {
        let mut result = Vec::new();

        for line in read_to_string(filename).unwrap().lines() {
            result.push(line.to_string())
        }

        result
    }

    #[test]
    fn something_up_with_cuda() {
        let a = Tensor::randn(Shape::new(vec![32, 27]));
        let b = Tensor::randn(Shape::new(vec![27, 200]));
        let c = a << b;

        let a_as_nd = a.item().into_dimensionality::<Ix2>().unwrap();
        let b_as_nd = b.item().into_dimensionality::<Ix2>().unwrap();
        let c_dot = a_as_nd.dot(&b_as_nd);

        println!("{:?}", c.item());
        println!("{:?}", c_dot);
    }

    #[test]
    fn array_indexing_testbed() {
        let mut a = ArrayD::from_elem(vec![27, 10], 0.0);
        let mut b = ArrayD::from_elem(vec![32, 3], 0.0);
        let mut c = ArrayD::from_elem(vec![32, 3, 10], 0.0);
        // Ok now I want to index A with b
        // roughly in the same way I would do in numpy
        // Fill A with some random values

        for i in 0..27 {
            for j in 0..10 {
                a[[i, j]] = i as f32 + j as f32;
            }
        }

        // Fill B with some random values, between 0 and 27
        let mut rng = rand::thread_rng();
        for i in 0..32 {
            for j in 0..3 {
                b[[i, j]] = rng.gen_range(0..27) as f32;
            }
        }

        //
        for i in 0..32 {
            for j in 0..3 {
                for k in 0..10 {
                    let test = [i, j, k];
                    c[test] = a[[b[[i, j]] as usize, k]];
                }
            }
        }

        println!("{:?}", c);
    }

    #[test]
    fn linear_module() {
        let mut linear = LinearLayer::new(3, 1);

        let inputs = vec![
            vec![2.0f32, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];

        let inputs_as_tensor = Tensor::from_vec(
            inputs.iter().flatten().map(|x| *x).collect(),
            vec![4, 3].into(),
        );

        let outputs = vec![1.0f32, -1.0, -1.0, 1.0];
        let outputs_as_tensor =
            Tensor::from_vec(outputs.iter().map(|x| *x).collect(), vec![4, 1].into());

        for _ in 0..50 {
            zero_all_grads();
            let prediction = linear.forward(&inputs_as_tensor);
            let loss = (prediction - outputs_as_tensor).pow(2.0);
            loss.backward();
            update_parameters(-0.01);
        }
    }

    fn build_batch_norm_dataset_from_subset(
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

    #[test]
    fn batch_norm_simple_test() {
        let n_hidden = 200;

        const BATCH_SIZE: usize = 32;
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
        let (xtr, ytr) = build_batch_norm_dataset_from_subset(&names[..n1], &stoi);
        let (_xdev, _ydev) = build_batch_norm_dataset_from_subset(&names[n1..n2], &stoi);
        let (_cte, _yte) = build_batch_norm_dataset_from_subset(&names[n2..], &stoi);

        let mut c = Tensor::load_from_weight_file("./data/batchnorm/C.json");
        c.set_requires_grad(true);
        let mut w1 = Tensor::load_from_weight_file("./data/batchnorm/W1.json");
        w1.set_requires_grad(true);
        let mut w2 = Tensor::load_from_weight_file("./data/batchnorm/W2.json");
        w2.set_requires_grad(true);
        let mut b2 = Tensor::load_from_weight_file("./data/batchnorm/b2.json");
        b2.set_requires_grad(true);

        let mut bngain = Tensor::load_from_weight_file("./data/batchnorm/bngain.json");
        bngain.set_requires_grad(true);
        let mut bnbiases = Tensor::load_from_weight_file("./data/batchnorm/bnbias.json");
        bnbiases.set_requires_grad(true);

        let mut bnmean_running = Tensor::zeroes(Shape::new(vec![1, n_hidden]));
        bnmean_running.set_requires_grad(true);
        let mut bnvar_running = Tensor::ones(Shape::new(vec![1, n_hidden]));
        bnvar_running.set_requires_grad(true);

        let max_steps = 2;

        for _i in 0..max_steps {
            zero_all_grads();
            let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![BATCH_SIZE, 3]));
            for b in 0..BATCH_SIZE {
                test_index_tensor.set_index([b, 0].into(), vec![xtr[b][0] as f32].into());
                test_index_tensor.set_index([b, 1].into(), vec![xtr[b][1] as f32].into());
                test_index_tensor.set_index([b, 2].into(), vec![xtr[b][2] as f32].into());
            }
            let test = c.view(Indexable::FromTensor(test_index_tensor.tensor_id));
            let reshape = test.reshape(Shape::new(vec![BATCH_SIZE, 30]));
            let hpreact = reshape << w1;

            let bnmeani = hpreact.mean(vec![0]);
            let bnvari = hpreact.std(vec![0]);
            let offset = hpreact - bnmeani;
            let numer = offset * bngain;
            let hpreact = numer / bnvari + bnbiases;

            let h = hpreact.tanh();
            let logits = (h << w2) + b2;

            let mut test_ytrue_onehot = Tensor::element(Shape::new(vec![BATCH_SIZE, 27]), 0.0);
            for b in 0..BATCH_SIZE {
                test_ytrue_onehot.set_index([b, ytr[b]].into(), vec![1.0].into());
            }

            let loss = logits.cross_entropy_loss(test_ytrue_onehot);
            println!("Loss: {}", loss.item());

            loss.backward();
            update_parameters(-0.01);
        }
        println!("w1 grad {:?}", w1.grad());
    }

    use crate::nn::model::{Model, Sequential};
    #[test]
    fn batch_norm_test() {
        let batch_size = 32;
        let block_size = 3;
        let vocab_size = 100;
        let n_embd = 10;
        let n_hidden = 100;
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

        let (xtr, _ytr) = build_dataset_from_subset(&names[..n1], &stoi);

        let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![batch_size, 3]));
        for b in 0..batch_size {
            test_index_tensor.set_index([b, 0].into(), vec![xtr[b][0] as f32].into());
            test_index_tensor.set_index([b, 1].into(), vec![xtr[b][1] as f32].into());
            test_index_tensor.set_index([b, 2].into(), vec![xtr[b][2] as f32].into());
        }

        let c = Tensor::randn(Shape::new(vec![vocab_size, n_embd]));

        let mut linear_model: Sequential = vec![
            LinearLayer::new(n_embd * block_size, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),
            LinearLayer::new(n_hidden, vocab_size).into(),
            BatchNorm1d::new(vocab_size).into(),
        ]
        .into();

        let test = c.view(Indexable::FromTensor(test_index_tensor.tensor_id));
        let reshape = test.reshape(Shape::new(vec![32, 30]));

        let output = linear_model.forward(&reshape);
        output.backward();
        update_parameters(-0.01);
    }
}
