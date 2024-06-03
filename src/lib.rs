#![feature(unboxed_closures)]
#![feature(fn_traits)]

mod central;
pub use central::*;
pub use ndarray::prelude::*;

#[cfg(test)]
mod tests {

    use rand::Rng;
    use crate::*;
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn add_test() {
        let a = Tensor::ones(Shape::new(vec![2, 2]));
        let b = Tensor::zeroes(Shape::new(vec![2, 2]));
        let c = a + b;
        let result = c.item();
        assert!(result == arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
    }

    #[test]
    fn add_test_2() {
        let a = Tensor::zeroes(Shape::new(vec![2, 2]));
        let b = Tensor::zeroes(Shape::new(vec![2, 2]));
        let c = a + b;
        let d = c.clone() + c;
        let result = d.item();
        assert!(result == arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn());
    }

    #[test]
    fn backward_add_test() {

        let a  = Tensor::ones(Shape::new(vec![1, 1]));
        let b = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let c = a + b;
        c.backward();
        let result = c.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = b.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
    }

    #[test]
    fn sub_test() {
        let a = Tensor::ones(Shape::new(vec![2, 2]));
        let b = Tensor::zeroes(Shape::new(vec![2, 2]));
        let c = a - b;
        let result = c.item();
        assert!(result == arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
    }

    #[test]
    fn sub_test_2() {
        let a = Tensor::zeroes(Shape::new(vec![2, 2]));
        let b = Tensor::zeroes(Shape::new(vec![2, 2]));
        let c = a - b;
        let d = c.clone() - c;
        let result = d.item();
        assert!(result == arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn());
    }

    #[test]
    fn backward_sub_test() {
        let a  = Tensor::ones(Shape::new(vec![1, 1]));
        let b = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let c = a - b;
        c.backward();
        let result = c.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = b.grad();
        assert!(result == arr2(&[[-1.0]]).into_dyn());
    }

    #[test]
    fn mul_test() {
        let a = Tensor::ones(Shape::new(vec![2, 2]));
        let b = Tensor::ones(Shape::new(vec![2, 2]));
        let c = a * b;
        let result = c.item();
        assert!(result == arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
    }

    #[test]
    fn mul_test_2() {
        let a = Tensor::ones(Shape::new(vec![2, 2]));
        let b = Tensor::element(Shape::new(vec![2, 2]), 2.0);
        let c = a * b;
        let result = c.item();
        assert!(result == arr2(&[[2.0, 2.0], [2.0, 2.0]]).into_dyn());
    }

    #[test]
    fn backward_mul_test() {

        let a  = Tensor::ones(Shape::new(vec![1, 1]));
        let b = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let c = a * b;
        c.backward();
        let result = c.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[2.0]]).into_dyn());
        let result = b.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
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
    fn basic_div_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a / 2.0;
        let result = b.item();
        assert!(result == arr2(&[[1.0]]).into_dyn());
    }

    #[test]
    fn backward_div_test() {
        let a = Tensor::element(Shape::new(vec![1, 1]), 2.0);
        let b = a / 2.0;
        b.backward();
        let result = b.grad();
        assert!(result == arr2(&[[1.0]]).into_dyn());
        let result = a.grad();
        assert!(result == arr2(&[[0.5]]).into_dyn());
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

    pub struct Neuron {
        pub weights: Vec<Tensor>,
        pub bias: Tensor,
    }

    impl Neuron {
        pub fn new(number_of_inputs: usize) -> Neuron {
            let mut weights = Vec::with_capacity(number_of_inputs);
            for _ in 0..number_of_inputs {
                let mut weight_tensor = Tensor::randn(Shape::new(vec![1]));
                weight_tensor.set_requires_grad(true);
                weights.push(weight_tensor );

            }
            let bias = Tensor::randn(Shape::new(vec![1]));
            Neuron { weights, bias }
        }

        pub fn forward(&self, inputs: &Vec<Tensor>) -> Tensor {
            let weighted_sum = inputs.iter().zip(&self.weights).fold(
                self.bias.clone(),
                |acc, (input, weight)| acc + (*input * *weight),
            );

            weighted_sum.tanh()
        }
    }

    pub struct Layer {
        pub neurons: Vec<Neuron>,
    }

    impl Layer {
        pub fn new(number_of_neurons: usize, number_of_inputs: usize) -> Layer {
            let mut neurons = Vec::with_capacity(number_of_neurons);
            for _ in 0..number_of_inputs {
                neurons.push(Neuron::new(number_of_neurons));
            }
            Layer { neurons }
        }

        pub fn forward(&self, inputs: &Vec<Tensor>) -> Vec<Tensor> {
            self.neurons.iter().map(|neuron| neuron.forward(inputs)).collect()
        }
    }

    pub struct Network {
        pub layers: Vec<Layer>,
    }

    impl Network {
        pub fn new(number_of_inputs: usize, mut neurons_per_layers: Vec<usize> ) -> Network {
            let mut sizes = vec![number_of_inputs];
            sizes.append(&mut neurons_per_layers);
            let mut layers = vec![];
            for a in sizes.chunks(2) {
                layers.push(Layer::new(a[0], a[1]));
            }
            Network { layers }
        }

        pub fn forward(&self, inputs: &Vec<Tensor>) -> Vec<Tensor> {
            let mut outputs = inputs.clone();
            for layer in &self.layers {
                outputs = layer.forward(&outputs);
            }
            outputs
        }
    }

    #[test]
    fn mlp_test() {
        let mlp = Network::new(3, vec![4, 4, 1]);
        let inputs = vec![
            vec![2.0f32, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];

        let outputs = vec![1.0f32, -1.0, -1.0, 1.0];
        for _ in 0..50 {
            {
                let mut singeton = SINGLETON_INSTANCE.lock().unwrap();
                singeton.zero_all_grads();
            }

            for (input, output) in inputs.iter().zip(&outputs) {
                let input = input.iter().map(|x| Tensor::element(Shape::new(vec![1]), *x)).collect();
                let output = Tensor::element(Shape::new(vec![1]), *output);
                let prediction = mlp.forward(&input)[0];
                let loss = (prediction - output).pow(2.0);
                loss.backward();
            }

            {
                let mut singeton = SINGLETON_INSTANCE.lock().unwrap();
                singeton.update_parameters(-0.01);
            }
        }

        let total_loss = inputs.iter().zip(&outputs).fold(0.0, |acc, (input, output)| {
            let input = input.iter().map(|x| Tensor::element(Shape::new(vec![1]), *x)).collect();
            let output = Tensor::element(Shape::new(vec![1]), *output);
            let prediction = mlp.forward(&input)[0];
            acc + (prediction - output).pow(2.0).item()[[0]]
        });
        println!("{:?}", total_loss);
    }

    use core::{f32, time};
    use std::f32::consts::E;
    use std::fs::read_to_string;
    use std::collections::{HashMap, HashSet};
    use std::process::Output;

    fn read_lines(filename: &str) -> Vec<String> {
        let mut result = Vec::new();

        for line in read_to_string(filename).unwrap().lines() {
            result.push(line.to_string())
        }

        result
    }

    fn generate_dataset() -> (HashMap<char, usize>, HashMap<usize, char>, Vec<usize>, Vec<usize>) {
        let names = read_lines("./data/bigram/names.txt");

        let chars: HashSet<char> = names.iter()
        .flat_map(|word| word.chars())
        .collect();    
        let mut chars_vec: Vec<char> = chars.into_iter().collect();


        chars_vec.sort_unstable();

        let mut stoi: HashMap<char, usize> = HashMap::new();
        for (i, &c) in chars_vec.iter().enumerate() {
            stoi.insert(c, i + 1);
        }
        stoi.insert('.', 0);


        let itos: HashMap<usize, char> = stoi.iter()
            .map(|(&c, &i)| (i, c))
            .collect();

            let mut xs: Vec<usize> = vec![];
            let mut ys = vec![];

            // I BUILT MY DATASET WRONG
            // Less tired, and with more time on your hand, this will be easy to solve
            for name in names.iter() {
                let fixed = String::from(".") + &name + ".";
                let chars: Vec<char> = fixed.chars().collect();
                for i in 0..chars.len() - 1 {
                    let pair = (chars[i], chars[i + 1]);
                    xs.push(stoi[&pair.0]);
                    ys.push(stoi[&pair.1]);
                }
            }

            let var_name: (HashMap<char, usize>, HashMap<usize, char>, Vec<usize>, Vec<usize>) = (stoi, itos, xs, ys);
            return var_name;
    }

    #[test]
    fn three_dimension_matmul_test() {
        let a = Tensor::randn(Shape::new(vec![3, 2, 2]));
        let b = Tensor::randn(Shape::new(vec![2, 2]));
        let c = a << b;
        let result = c.item();
        println!("{:?}", result);
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

    fn build_dataset_from_subset(words: &[String], stoi: &HashMap<char, usize>) -> (Vec<[usize;3]>, Vec<usize>) {
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
    fn mm_mlp_test() {
       // let mut times = HashMap::new();

        
        let BATCH_SIZE = 32;
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
        let (Xtr, Ytr) = build_dataset_from_subset(&names[..n1], &stoi);
        let (Xdev, Ydev) = build_dataset_from_subset(&names[n1..n2], &stoi);
        let (Xte, Yte) = build_dataset_from_subset(&names[n2..], &stoi);

        let mut C = Tensor::load_from_weight_file("./data/bigram/tensor_C.json");
      
        C.set_requires_grad(true);
        let mut W1 = Tensor::load_from_weight_file("./data/bigram/tensor_W1.json");
        W1.set_requires_grad(true);
        let mut b1 = Tensor::load_from_weight_file("./data/bigram/tensor_b1.json");
        b1.set_requires_grad(true);
        let mut W2 = Tensor::load_from_weight_file("./data/bigram/tensor_W2.json");
        W2.set_requires_grad(true);
        let mut b2 = Tensor::load_from_weight_file("./data/bigram/tensor_b2.json");
        b2.set_requires_grad(true);

        let EPOCH_COUNT = 50;


        for epoch in 0..10 {
            println!("Epoch: {:?}", epoch);
            {
                let mut singleton = SINGLETON_INSTANCE.lock().unwrap();
                singleton.zero_all_grads();
            }
    

            let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![BATCH_SIZE, 3]));
            for b in 0..BATCH_SIZE {
                test_index_tensor.set_index([b, 0].into(), vec![Xtr[b][0] as f32].into());
                test_index_tensor.set_index([b, 1].into(), vec![Xtr[b][1] as f32].into());
                test_index_tensor.set_index([b, 2].into(), vec![Xtr[b][2] as f32].into());
            }

            let test = C.view(Indexable::FromTensor(test_index_tensor.tensor_id));
            let reshape = test.reshape(Shape::new(vec![BATCH_SIZE, 30]));
            let test_mult = reshape << W1;
            let test_add = test_mult + b1;
            let test_tanh = test_add.tanh();
            let test_output = test_tanh << W2;
            let test_output = test_output + b2;

            let test_max = test_output.max(1);
            let test_counts = (test_output - test_max).exp();
            let test_counts_sum = test_counts.sum(1);

            let test_counts_cum_inverted = test_counts_sum.pow(-1.0);
            let test_probabilities = test_counts * test_counts_cum_inverted;

            let mut test_ytrue_onehot = Tensor::element(Shape::new(vec![BATCH_SIZE, 27]), 0.0);
            for b in 0..BATCH_SIZE {
                test_ytrue_onehot.set_index([b, Ytr[b]].into(), vec![1.0].into());
            }

            let test_prob_log = test_probabilities.log();

            let test_presum = test_ytrue_onehot * test_prob_log;
            let test_sum = (-test_presum).sum(1);
            let test_mean = test_sum.mean(0);

            println!("Loss: {:?}", test_mean.item());
            test_mean.backward();

            {
                let mut singleton = SINGLETON_INSTANCE.lock().unwrap();
                singleton.update_parameters(-0.1);
            }

        }


    }

    use crate::central::Linear;
    use crate::central::{Model, LinearModel};

    #[test]
    fn linear_module() {
        let mut linear = Linear::new(3, 1);

        let inputs = vec![
            vec![2.0f32, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];

        let inputs_as_tensor = Tensor::from_vec(inputs.iter().flatten().map(|x|*x).collect(), vec![4, 3].into());

        let outputs = vec![1.0f32, -1.0, -1.0, 1.0];
        let outputs_as_tensor = Tensor::from_vec(outputs.iter().map(|x| *x).collect(), vec![4, 1].into());

        for _ in 0..50 {
            zero_all_grads();
            let prediction = linear.forward(&inputs_as_tensor);
            let loss = (prediction - outputs_as_tensor).pow(2.0);
            loss.backward();
            update_parameters(-0.01);
        }
    }

    #[test]
    fn linear_model() {
        let mut linear_model = LinearModel::new(vec![Box::new(Linear::new(3, 1))]);

        let inputs = vec![
            vec![2.0f32, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];

        let inputs_as_tensor = Tensor::from_vec(inputs.iter().flatten().map(|x|*x).collect(), vec![4, 3].into());

        let outputs = vec![1.0f32, -1.0, -1.0, 1.0];

        let outputs_as_tensor = Tensor::from_vec(outputs.iter().map(|x| *x).collect(), vec![4, 1].into());

        for _ in 0..50 {
            zero_all_grads();
            let prediction = linear_model.forward(&inputs_as_tensor);
            let loss = (prediction - outputs_as_tensor).pow(2.0);
            loss.backward();
            update_parameters(-0.01);
        }
    }
}
