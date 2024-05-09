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

    use core::f32;
    use std::f32::consts::E;
    use std::fs::read_to_string;
    use std::collections::{HashMap, HashSet};

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
    fn bigram_test_single_pass() {
        let from_test_weights =vec![[ 
         1.5674e+00, -2.3729e-01, -2.7385e-02, -1.1008e+00,  2.8588e-01,
        -2.9643e-02, -1.5471e+00,  6.0489e-01,  7.9136e-02,  9.0462e-01,
        -4.7125e-01,  7.8682e-01, -3.2843e-01, -4.3297e-01,  1.3729e+00,
         2.9334e+00,  1.5618e+00, -1.6261e+00,  6.7716e-01, -8.4039e-01,
         9.8488e-01, -1.4837e-01, -1.4795e+00,  4.4830e-01, -7.0730e-02,
         2.4968e+00,  2.4448e+00],
       [-6.7006e-01, -1.2199e+00,  3.0314e-01, -1.0725e+00,  7.2762e-01,
         5.1114e-02,  1.3095e+00, -8.0220e-01, -8.5042e-01, -1.8068e+00,
         1.2523e+00, -1.2256e+00,  1.2165e+00, -9.6478e-01, -2.3211e-01,
        -3.4762e-01,  3.3244e-01, -1.3263e+00,  1.1224e+00,  5.9641e-01,
         4.5846e-01,  5.4011e-02, -1.7400e+00,  1.1560e-01,  8.0319e-01,
         5.4108e-01, -1.1646e+00],
       [ 1.4756e-01, -1.0006e+00,  3.8012e-01,  4.7328e-01, -9.1027e-01,
        -7.8305e-01,  1.3506e-01, -2.1161e-01, -1.0406e+00, -1.5367e+00,
         9.3743e-01, -8.8303e-01,  1.7457e+00,  2.1346e+00, -8.5614e-01,
         5.4082e-01,  6.1690e-01,  1.5160e+00, -1.0447e+00, -6.6414e-01,
        -7.2390e-01,  1.7507e+00,  1.7530e-01,  9.9280e-01, -6.2787e-01,
         7.7023e-02, -1.1641e+00],
       [ 1.2473e+00, -2.7061e-01, -1.3635e+00,  1.3066e+00,  3.2307e-01,
         1.0358e+00, -8.6249e-01, -1.2575e+00,  9.4180e-01, -1.3257e+00,
         1.4670e-01,  1.6913e-01, -1.5397e+00, -7.2759e-01,  1.1491e+00,
        -8.7462e-01, -2.9771e-01, -1.3707e+00,  1.1500e-01, -1.0188e+00,
        -8.3777e-01, -2.1057e+00, -2.6044e-01, -1.7149e+00, -3.3787e-01,
        -1.8263e+00, -8.3897e-01],
       [-1.5723e+00,  4.5795e-01, -5.6533e-01,  5.4281e-01,  1.7549e-01,
        -2.2901e+00, -7.0928e-01, -2.9283e-01, -2.1803e+00,  7.9311e-02,
         9.0187e-01,  1.2028e+00, -5.6144e-01, -1.3753e-01, -1.3799e-01,
        -2.0977e+00, -7.9238e-01,  6.0689e-01, -1.4777e+00, -5.1029e-01,
         5.6421e-01,  9.6838e-01, -3.1114e-01, -3.0603e-01, -1.7495e+00,
        -1.6335e+00,  3.8761e-01],
       [ 4.7236e-01,  1.4830e+00,  3.1748e-01,  1.0588e+00,  2.3982e+00,
         4.6827e-01, -6.5650e-01,  6.1662e-01, -6.2197e-01,  5.1007e-01,
         1.3563e+00,  2.3445e-01, -4.5585e-01, -1.3132e-03, -5.1161e-01,
         5.5570e-01,  4.7458e-01, -1.3867e+00,  1.6229e+00,  1.7197e-01,
         9.8846e-01,  5.0657e-01,  1.0198e+00, -1.9062e+00, -4.2753e-01,
        -2.1259e+00,  9.6041e-01],
       [ 1.2482e+00,  2.5341e-01,  2.8188e+00, -3.3983e-01,  7.0311e-01,
         4.0716e-01, -1.9018e-01, -6.9652e-01,  1.7039e+00,  7.4204e-01,
         9.7370e-01,  3.0028e-01, -2.8971e-01, -3.1566e-01, -8.7898e-01,
         1.0661e-01,  1.8598e+00,  5.5752e-02,  1.2815e+00, -6.3182e-01,
        -1.2464e+00,  6.8305e-01, -3.9455e-01,  1.4388e-02,  5.7216e-01,
         8.6726e-01,  6.3149e-01],
       [-1.2230e+00, -2.1286e-01,  5.0950e-01,  3.2713e-01,  1.9661e+00,
        -2.4091e-01, -7.9515e-01,  2.7198e-01, -1.1100e+00, -4.5285e-01,
        -4.9578e-01,  1.2648e+00,  1.4625e+00,  1.1199e+00,  9.9539e-01,
        -1.2353e+00,  7.3818e-01,  8.1415e-01, -7.3806e-01,  5.6714e-01,
        -1.4601e+00, -2.4780e-01,  8.8282e-01, -8.1004e-02, -9.5299e-01,
        -4.8838e-01, -7.3712e-01],
       [ 7.0609e-01, -1.9295e-01,  1.2348e+00,  3.3308e-01,  1.3283e+00,
        -1.0921e+00, -8.3952e-01,  1.9098e-01, -7.1750e-01, -3.8668e-01,
        -1.2542e+00,  1.2068e+00, -1.7102e+00, -4.7701e-01, -1.0527e+00,
        -1.4367e-01, -2.7737e-01,  1.1634e+00, -6.6910e-01,  6.4918e-01,
         5.8243e-01,  1.9264e+00, -3.7846e-01,  7.9577e-03,  5.1068e-01,
         7.5927e-01, -1.6086e+00],
       [-1.6065e-01,  1.3784e+00, -2.7804e-01,  2.0710e-01,  1.0033e+00,
        -5.9772e-01, -3.9771e-01, -1.2801e+00,  9.2445e-02,  1.0526e-01,
        -3.9072e-01, -4.0091e-01,  5.6533e-01, -1.5065e+00,  1.2898e+00,
        -1.5100e+00,  1.0930e+00,  1.0797e+00, -8.6681e-02,  1.3423e+00,
         1.5184e-01,  2.4687e-01,  3.1895e-01, -9.8614e-01, -2.1382e-01,
        -6.4308e-02, -8.5528e-01],
       [ 1.6113e-01,  4.4925e-01,  8.1827e-01, -8.1628e-01, -3.9243e-01,
        -7.4521e-01, -9.4649e-01, -1.5941e-01, -1.5047e+00,  8.4682e-01,
        -4.9158e-02,  9.3866e-02, -6.4533e-01,  1.2108e+00, -7.8198e-01,
         3.8449e-01, -8.5259e-01,  1.0464e+00, -1.8493e+00,  9.1092e-01,
        -9.9360e-01,  6.0195e-01, -1.0890e-01,  5.2587e-01, -9.4046e-01,
        -1.2773e-01, -2.5679e-01],
       [-1.5437e+00,  3.7950e-01, -1.7705e+00, -1.2085e+00,  9.4773e-01,
        -9.1355e-01,  7.1023e-01,  7.9512e-01,  5.7662e-01, -7.3778e-01,
        -1.5264e+00,  7.1173e-01,  1.4056e+00, -4.0636e-01, -7.4648e-01,
         4.9790e-01,  1.1298e-01, -4.1854e-01,  1.7905e-01,  2.3483e-01,
         7.3510e-01, -6.1577e-01,  7.0467e-01,  1.1630e-01,  2.8365e-01,
        -2.5043e+00, -5.1931e-01],
       [-5.9134e-01, -1.1059e-01,  8.3416e-01, -1.0505e+00,  3.6345e-01,
         1.8195e-01, -4.8045e-01,  5.3309e-01,  6.7869e-01, -3.5974e-01,
        -1.3270e+00, -8.2526e-01,  6.3614e-01,  1.9110e-01,  7.5476e-01,
         4.0538e-01,  2.2565e+00,  1.3655e+00, -5.6192e-01, -3.0423e-01,
         2.9894e-01,  1.8784e+00,  5.5958e-01,  1.3388e+00,  4.1606e-01,
         6.8491e-01, -1.4790e-01],
       [ 1.9359e-01,  1.0532e+00,  6.3393e-01,  2.5786e-01,  9.6408e-01,
        -2.4855e-01,  2.4756e-02, -3.0404e-02,  1.5622e+00, -4.4852e-01,
        -1.2345e+00,  1.1220e+00, -6.7381e-01,  3.7882e-02, -5.5881e-01,
        -8.2709e-01,  8.2253e-01, -7.5100e-01,  9.2778e-01, -1.4849e+00,
        -2.1293e-01, -1.1860e+00, -6.6092e-01, -2.3348e-01,  1.5447e+00,
         6.0061e-01, -7.0909e-01],
       [ 1.9217e+00, -1.8182e-01,  1.5220e+00,  5.4644e-01,  4.0858e-01,
        -1.9692e+00, -8.9185e-01,  3.2961e-01, -2.5128e-01,  5.5030e-01,
        -7.5171e-01, -6.5783e-03, -6.3108e-01,  1.3431e+00,  3.8010e-02,
        -7.1654e-01,  1.7206e+00, -5.2149e-01, -2.3248e-01,  1.0774e+00,
        -7.6019e-01,  9.0109e-03, -7.9219e-01,  1.2307e+00, -5.2760e-01,
        -1.3207e+00, -7.0654e-01],
       [-7.7861e-01,  1.2910e+00, -1.5094e+00,  7.4593e-01,  4.8990e-01,
        -1.0034e+00,  9.6407e-01,  2.0990e+00, -3.9870e-01, -7.6635e-01,
        -2.1007e+00,  1.2331e+00,  7.7481e-01,  2.4311e-01, -2.1322e-01,
        -6.9877e-01,  2.0889e-01, -6.2477e-01, -1.0825e-01, -2.1964e+00,
         2.7083e-01,  6.1047e-01, -5.8162e-01, -1.7025e+00, -8.0672e-01,
        -2.4174e-01,  1.5490e+00],
       [-3.4593e-01,  5.4714e-01,  3.1755e-02,  8.1375e-01,  2.6200e-01,
        -6.7101e-01,  2.0656e-02,  7.1300e-01, -4.3997e-02, -5.1944e-01,
         1.1241e-01, -3.9770e-01, -2.7829e-01, -1.5364e-01, -2.5424e+00,
         2.5033e-01,  1.1056e-01, -2.0366e+00, -9.2735e-01, -6.9350e-01,
        -5.2788e-01, -8.7438e-01, -1.0102e+00, -1.0522e+00,  1.2348e+00,
         2.5907e-02, -9.6676e-01],
       [ 1.0904e+00,  5.3966e-01,  6.6741e-01, -2.2316e+00, -1.1603e+00,
        -4.2560e-01,  5.9547e-01, -1.0887e+00,  2.4324e-01, -2.1021e+00,
        -2.9289e-01, -7.0682e-01,  9.5190e-01, -1.1583e+00, -1.2844e+00,
         1.0193e+00,  1.6851e+00,  8.3422e-01,  1.7113e+00,  4.4456e-01,
        -7.1861e-01, -7.0343e-01, -7.1332e-01,  9.9760e-01, -6.1980e-01,
         1.9522e+00,  1.4311e-01],
       [ 1.8765e-01,  7.5974e-01, -2.6387e-01, -7.3048e-01,  6.1955e-01,
         3.5577e-02, -7.6459e-02, -1.2306e+00,  1.3419e+00,  1.1878e+00,
        -1.0672e+00, -2.1507e+00,  6.7082e-01,  1.1614e+00, -2.4155e-01,
         9.5907e-01,  3.8262e-02,  3.9877e-02, -7.7180e-01,  2.9251e-01,
        -6.0606e-01, -1.5136e+00, -2.7143e+00, -4.1164e-01, -1.2273e+00,
        -4.1746e-01,  1.5021e+00],
       [-6.2849e-01, -4.4247e-01,  5.6885e-01,  1.2803e+00, -5.5397e-01,
         1.1179e+00, -6.0053e-01, -5.8619e-01, -2.8277e-01,  5.3390e-01,
        -9.9388e-01, -1.6996e+00,  1.8362e+00,  4.2016e-01, -6.8729e-01,
        -3.5060e-01,  7.5598e-01, -9.3632e-01, -8.4109e-02, -1.6361e+00,
         1.0224e+00,  1.0733e+00, -5.7453e-01,  4.9668e-02,  7.2379e-01,
         5.9746e-01,  2.6966e+00],
       [ 2.7930e+00, -2.2745e+00, -2.3912e-01,  8.7498e-02,  1.4967e+00,
        -5.7016e-01, -5.7248e-01,  1.9909e+00, -7.4416e-01,  7.2960e-01,
         6.4083e-01,  1.6075e+00, -8.8810e-01,  2.7359e-01, -1.3257e-01,
         1.2710e+00,  1.7234e+00,  1.1180e-01,  2.6952e-01,  1.1835e+00,
         1.2575e+00,  1.3969e-01,  4.7259e-01,  7.9025e-01,  1.0811e+00,
        -9.1965e-01, -4.0503e-01],
       [ 4.5696e-01, -5.4184e-01, -2.3025e+00,  2.0127e+00, -4.6452e-01,
        -5.8270e-01,  2.0863e+00, -4.7729e-02, -4.4920e-01,  9.5566e-01,
        -1.4708e-01, -1.2532e+00, -1.1850e+00,  3.6583e-01, -1.4049e-01,
         3.5252e-01, -5.2400e-01, -6.2844e-01, -9.3792e-01,  1.6772e+00,
         3.8554e-03, -7.3685e-01, -9.3514e-01,  1.0465e-01, -4.6464e-01,
         1.6676e+00,  1.3931e+00],
       [ 6.5398e-01, -2.2449e-01,  1.2831e+00, -9.1787e-01, -3.3916e-01,
        -1.8058e+00,  6.0518e-01, -5.6252e-01, -7.8933e-01,  1.2767e+00,
        -1.0143e+00,  4.1611e-01, -7.5348e-01,  1.7128e+00, -8.7554e-01,
         3.9714e-01,  8.4326e-01,  3.7988e-01, -1.1670e+00,  5.5228e-01,
        -1.0279e+00, -3.9554e-01, -7.1410e-01, -8.7456e-02, -3.3361e-01,
        -1.8798e-01, -1.2647e+00],
       [ 2.0021e+00, -2.3470e-01, -1.3765e+00,  9.3426e-01,  1.0880e+00,
         1.9179e-01,  3.0114e-01,  8.9896e-01, -8.4454e-01,  2.3267e-01,
        -3.9205e-01, -2.5081e-01,  8.7124e-02,  1.3769e+00, -8.3358e-01,
        -8.9400e-01,  1.1744e+00, -6.0779e-01, -1.1493e-01, -7.8077e-01,
         1.9660e+00,  6.1175e-01,  3.6039e-01, -1.0274e+00,  1.1495e+00,
         4.5111e-01,  6.4420e-01],
       [ 2.1635e-01, -7.8731e-01, -3.3005e-01,  3.2877e-01, -1.6332e+00,
         1.0807e+00,  3.3638e-01,  1.1536e-01,  3.2834e-01,  5.3447e-02,
         1.4224e+00, -8.3957e-01, -2.4956e-01, -8.9778e-01, -8.6583e-01,
        -1.0786e+00, -1.8384e-01,  7.1622e-01,  1.8175e-01,  1.1053e+00,
         1.7003e+00, -1.6965e-01,  1.6293e-01,  1.3413e+00, -2.6301e-01,
        -7.5521e-01,  8.1911e-01],
       [ 7.4140e-01, -5.8787e-01, -4.6505e-01,  5.3112e-02,  2.2190e+00,
        -3.5158e-01,  3.6381e-01,  2.5769e+00,  1.4544e+00, -6.1003e-01,
        -5.9961e-01, -5.8392e-01, -1.8104e-02, -9.5177e-01, -9.6400e-01,
        -2.8183e-01,  1.0597e+00, -7.2370e-01,  1.4755e-01, -3.2667e-01,
         2.4958e+00,  1.1088e+00, -8.5476e-01,  1.8443e+00, -1.3881e-01,
         1.3096e+00, -2.5802e-01],
       [ 1.0669e+00,  2.1363e-01, -7.6603e-01, -1.6977e+00, -1.5023e-01,
        -5.2150e-01, -6.3730e-01,  2.6214e-01,  7.6539e-03,  1.3067e+00,
        -6.3482e-01, -1.1042e-04, -6.6158e-01,  1.4723e-01, -6.6036e-02,
         5.2851e-01,  5.7950e-01,  2.1438e-01,  9.2200e-01,  5.2919e-01,
         7.7070e-01,  4.2899e-01,  3.4330e-01,  2.0698e+00,  1.3405e+00,
        -2.1746e-01,  8.6273e-01]];
        let mut final_weights = vec![];
        for outer_vec in from_test_weights {
            for j in outer_vec {
               final_weights.push(j as f32);
            }
        }
        const NUMBER_OF_CHARACTERS: usize = 27;

        let mut weights = Tensor::from_vec(final_weights, Shape::new(vec![NUMBER_OF_CHARACTERS, NUMBER_OF_CHARACTERS]));
        // will need to do a "set_requires_grad" function
        weights.set_requires_grad(true);
    
        let (_stoi, _itos, xs, ys) = generate_dataset();
        const TEST_BATCH: usize = 10; // Using only one batch for simplicity
    
        // Prepare a batch of data
        let combined = xs.iter().take(TEST_BATCH).zip(ys.iter().take(TEST_BATCH));
        let mut inputs = Tensor::zeroes(Shape::new(vec![TEST_BATCH, NUMBER_OF_CHARACTERS]));
        let mut targets = vec![];
        let mut index = 0;
        for (x, y) in combined {
            inputs.set_index([index, *x].into(), vec![1.0].into());
            targets.push(*y);
            index += 1;
        }

        // convert targets into a tensor
        let loops = 10;
        // Assuming we have only one batch, so no need for an outer epoch loop
        for _ in 0..loops {
            {
                let mut singleton = SINGLETON_INSTANCE.lock().unwrap();
                singleton.zero_all_grads();
            }

                let prediction = inputs << weights;

                let counts = prediction.exp();
                let counts_sum = counts.sum();
                let counts_cum_inverted = counts_sum.pow(-1.0);
                let probabilities = counts * counts_cum_inverted;


                let mut logged = vec![];
                for i in 0..TEST_BATCH {
                    logged.push(probabilities.view([i, targets[i]].into()).log());
                }

                let mean = -(Tensor::t_mean(&logged));

                println!("loss: {:?}", mean.item());
                mean.backward();
                /*
                for log in logged.iter() {
                    println!("log: {:?}", log.grad());
                }
                 */

            {
                let mut singleton = SINGLETON_INSTANCE.lock().unwrap();
                singleton.update_parameters(-50.0);
            }
        }

    }

    #[test]
    fn bigram_test() {
        let from_test_weights =vec![[ 
         1.5674e+00, -2.3729e-01, -2.7385e-02, -1.1008e+00,  2.8588e-01,
        -2.9643e-02, -1.5471e+00,  6.0489e-01,  7.9136e-02,  9.0462e-01,
        -4.7125e-01,  7.8682e-01, -3.2843e-01, -4.3297e-01,  1.3729e+00,
         2.9334e+00,  1.5618e+00, -1.6261e+00,  6.7716e-01, -8.4039e-01,
         9.8488e-01, -1.4837e-01, -1.4795e+00,  4.4830e-01, -7.0730e-02,
         2.4968e+00,  2.4448e+00],
       [-6.7006e-01, -1.2199e+00,  3.0314e-01, -1.0725e+00,  7.2762e-01,
         5.1114e-02,  1.3095e+00, -8.0220e-01, -8.5042e-01, -1.8068e+00,
         1.2523e+00, -1.2256e+00,  1.2165e+00, -9.6478e-01, -2.3211e-01,
        -3.4762e-01,  3.3244e-01, -1.3263e+00,  1.1224e+00,  5.9641e-01,
         4.5846e-01,  5.4011e-02, -1.7400e+00,  1.1560e-01,  8.0319e-01,
         5.4108e-01, -1.1646e+00],
       [ 1.4756e-01, -1.0006e+00,  3.8012e-01,  4.7328e-01, -9.1027e-01,
        -7.8305e-01,  1.3506e-01, -2.1161e-01, -1.0406e+00, -1.5367e+00,
         9.3743e-01, -8.8303e-01,  1.7457e+00,  2.1346e+00, -8.5614e-01,
         5.4082e-01,  6.1690e-01,  1.5160e+00, -1.0447e+00, -6.6414e-01,
        -7.2390e-01,  1.7507e+00,  1.7530e-01,  9.9280e-01, -6.2787e-01,
         7.7023e-02, -1.1641e+00],
       [ 1.2473e+00, -2.7061e-01, -1.3635e+00,  1.3066e+00,  3.2307e-01,
         1.0358e+00, -8.6249e-01, -1.2575e+00,  9.4180e-01, -1.3257e+00,
         1.4670e-01,  1.6913e-01, -1.5397e+00, -7.2759e-01,  1.1491e+00,
        -8.7462e-01, -2.9771e-01, -1.3707e+00,  1.1500e-01, -1.0188e+00,
        -8.3777e-01, -2.1057e+00, -2.6044e-01, -1.7149e+00, -3.3787e-01,
        -1.8263e+00, -8.3897e-01],
       [-1.5723e+00,  4.5795e-01, -5.6533e-01,  5.4281e-01,  1.7549e-01,
        -2.2901e+00, -7.0928e-01, -2.9283e-01, -2.1803e+00,  7.9311e-02,
         9.0187e-01,  1.2028e+00, -5.6144e-01, -1.3753e-01, -1.3799e-01,
        -2.0977e+00, -7.9238e-01,  6.0689e-01, -1.4777e+00, -5.1029e-01,
         5.6421e-01,  9.6838e-01, -3.1114e-01, -3.0603e-01, -1.7495e+00,
        -1.6335e+00,  3.8761e-01],
       [ 4.7236e-01,  1.4830e+00,  3.1748e-01,  1.0588e+00,  2.3982e+00,
         4.6827e-01, -6.5650e-01,  6.1662e-01, -6.2197e-01,  5.1007e-01,
         1.3563e+00,  2.3445e-01, -4.5585e-01, -1.3132e-03, -5.1161e-01,
         5.5570e-01,  4.7458e-01, -1.3867e+00,  1.6229e+00,  1.7197e-01,
         9.8846e-01,  5.0657e-01,  1.0198e+00, -1.9062e+00, -4.2753e-01,
        -2.1259e+00,  9.6041e-01],
       [ 1.2482e+00,  2.5341e-01,  2.8188e+00, -3.3983e-01,  7.0311e-01,
         4.0716e-01, -1.9018e-01, -6.9652e-01,  1.7039e+00,  7.4204e-01,
         9.7370e-01,  3.0028e-01, -2.8971e-01, -3.1566e-01, -8.7898e-01,
         1.0661e-01,  1.8598e+00,  5.5752e-02,  1.2815e+00, -6.3182e-01,
        -1.2464e+00,  6.8305e-01, -3.9455e-01,  1.4388e-02,  5.7216e-01,
         8.6726e-01,  6.3149e-01],
       [-1.2230e+00, -2.1286e-01,  5.0950e-01,  3.2713e-01,  1.9661e+00,
        -2.4091e-01, -7.9515e-01,  2.7198e-01, -1.1100e+00, -4.5285e-01,
        -4.9578e-01,  1.2648e+00,  1.4625e+00,  1.1199e+00,  9.9539e-01,
        -1.2353e+00,  7.3818e-01,  8.1415e-01, -7.3806e-01,  5.6714e-01,
        -1.4601e+00, -2.4780e-01,  8.8282e-01, -8.1004e-02, -9.5299e-01,
        -4.8838e-01, -7.3712e-01],
       [ 7.0609e-01, -1.9295e-01,  1.2348e+00,  3.3308e-01,  1.3283e+00,
        -1.0921e+00, -8.3952e-01,  1.9098e-01, -7.1750e-01, -3.8668e-01,
        -1.2542e+00,  1.2068e+00, -1.7102e+00, -4.7701e-01, -1.0527e+00,
        -1.4367e-01, -2.7737e-01,  1.1634e+00, -6.6910e-01,  6.4918e-01,
         5.8243e-01,  1.9264e+00, -3.7846e-01,  7.9577e-03,  5.1068e-01,
         7.5927e-01, -1.6086e+00],
       [-1.6065e-01,  1.3784e+00, -2.7804e-01,  2.0710e-01,  1.0033e+00,
        -5.9772e-01, -3.9771e-01, -1.2801e+00,  9.2445e-02,  1.0526e-01,
        -3.9072e-01, -4.0091e-01,  5.6533e-01, -1.5065e+00,  1.2898e+00,
        -1.5100e+00,  1.0930e+00,  1.0797e+00, -8.6681e-02,  1.3423e+00,
         1.5184e-01,  2.4687e-01,  3.1895e-01, -9.8614e-01, -2.1382e-01,
        -6.4308e-02, -8.5528e-01],
       [ 1.6113e-01,  4.4925e-01,  8.1827e-01, -8.1628e-01, -3.9243e-01,
        -7.4521e-01, -9.4649e-01, -1.5941e-01, -1.5047e+00,  8.4682e-01,
        -4.9158e-02,  9.3866e-02, -6.4533e-01,  1.2108e+00, -7.8198e-01,
         3.8449e-01, -8.5259e-01,  1.0464e+00, -1.8493e+00,  9.1092e-01,
        -9.9360e-01,  6.0195e-01, -1.0890e-01,  5.2587e-01, -9.4046e-01,
        -1.2773e-01, -2.5679e-01],
       [-1.5437e+00,  3.7950e-01, -1.7705e+00, -1.2085e+00,  9.4773e-01,
        -9.1355e-01,  7.1023e-01,  7.9512e-01,  5.7662e-01, -7.3778e-01,
        -1.5264e+00,  7.1173e-01,  1.4056e+00, -4.0636e-01, -7.4648e-01,
         4.9790e-01,  1.1298e-01, -4.1854e-01,  1.7905e-01,  2.3483e-01,
         7.3510e-01, -6.1577e-01,  7.0467e-01,  1.1630e-01,  2.8365e-01,
        -2.5043e+00, -5.1931e-01],
       [-5.9134e-01, -1.1059e-01,  8.3416e-01, -1.0505e+00,  3.6345e-01,
         1.8195e-01, -4.8045e-01,  5.3309e-01,  6.7869e-01, -3.5974e-01,
        -1.3270e+00, -8.2526e-01,  6.3614e-01,  1.9110e-01,  7.5476e-01,
         4.0538e-01,  2.2565e+00,  1.3655e+00, -5.6192e-01, -3.0423e-01,
         2.9894e-01,  1.8784e+00,  5.5958e-01,  1.3388e+00,  4.1606e-01,
         6.8491e-01, -1.4790e-01],
       [ 1.9359e-01,  1.0532e+00,  6.3393e-01,  2.5786e-01,  9.6408e-01,
        -2.4855e-01,  2.4756e-02, -3.0404e-02,  1.5622e+00, -4.4852e-01,
        -1.2345e+00,  1.1220e+00, -6.7381e-01,  3.7882e-02, -5.5881e-01,
        -8.2709e-01,  8.2253e-01, -7.5100e-01,  9.2778e-01, -1.4849e+00,
        -2.1293e-01, -1.1860e+00, -6.6092e-01, -2.3348e-01,  1.5447e+00,
         6.0061e-01, -7.0909e-01],
       [ 1.9217e+00, -1.8182e-01,  1.5220e+00,  5.4644e-01,  4.0858e-01,
        -1.9692e+00, -8.9185e-01,  3.2961e-01, -2.5128e-01,  5.5030e-01,
        -7.5171e-01, -6.5783e-03, -6.3108e-01,  1.3431e+00,  3.8010e-02,
        -7.1654e-01,  1.7206e+00, -5.2149e-01, -2.3248e-01,  1.0774e+00,
        -7.6019e-01,  9.0109e-03, -7.9219e-01,  1.2307e+00, -5.2760e-01,
        -1.3207e+00, -7.0654e-01],
       [-7.7861e-01,  1.2910e+00, -1.5094e+00,  7.4593e-01,  4.8990e-01,
        -1.0034e+00,  9.6407e-01,  2.0990e+00, -3.9870e-01, -7.6635e-01,
        -2.1007e+00,  1.2331e+00,  7.7481e-01,  2.4311e-01, -2.1322e-01,
        -6.9877e-01,  2.0889e-01, -6.2477e-01, -1.0825e-01, -2.1964e+00,
         2.7083e-01,  6.1047e-01, -5.8162e-01, -1.7025e+00, -8.0672e-01,
        -2.4174e-01,  1.5490e+00],
       [-3.4593e-01,  5.4714e-01,  3.1755e-02,  8.1375e-01,  2.6200e-01,
        -6.7101e-01,  2.0656e-02,  7.1300e-01, -4.3997e-02, -5.1944e-01,
         1.1241e-01, -3.9770e-01, -2.7829e-01, -1.5364e-01, -2.5424e+00,
         2.5033e-01,  1.1056e-01, -2.0366e+00, -9.2735e-01, -6.9350e-01,
        -5.2788e-01, -8.7438e-01, -1.0102e+00, -1.0522e+00,  1.2348e+00,
         2.5907e-02, -9.6676e-01],
       [ 1.0904e+00,  5.3966e-01,  6.6741e-01, -2.2316e+00, -1.1603e+00,
        -4.2560e-01,  5.9547e-01, -1.0887e+00,  2.4324e-01, -2.1021e+00,
        -2.9289e-01, -7.0682e-01,  9.5190e-01, -1.1583e+00, -1.2844e+00,
         1.0193e+00,  1.6851e+00,  8.3422e-01,  1.7113e+00,  4.4456e-01,
        -7.1861e-01, -7.0343e-01, -7.1332e-01,  9.9760e-01, -6.1980e-01,
         1.9522e+00,  1.4311e-01],
       [ 1.8765e-01,  7.5974e-01, -2.6387e-01, -7.3048e-01,  6.1955e-01,
         3.5577e-02, -7.6459e-02, -1.2306e+00,  1.3419e+00,  1.1878e+00,
        -1.0672e+00, -2.1507e+00,  6.7082e-01,  1.1614e+00, -2.4155e-01,
         9.5907e-01,  3.8262e-02,  3.9877e-02, -7.7180e-01,  2.9251e-01,
        -6.0606e-01, -1.5136e+00, -2.7143e+00, -4.1164e-01, -1.2273e+00,
        -4.1746e-01,  1.5021e+00],
       [-6.2849e-01, -4.4247e-01,  5.6885e-01,  1.2803e+00, -5.5397e-01,
         1.1179e+00, -6.0053e-01, -5.8619e-01, -2.8277e-01,  5.3390e-01,
        -9.9388e-01, -1.6996e+00,  1.8362e+00,  4.2016e-01, -6.8729e-01,
        -3.5060e-01,  7.5598e-01, -9.3632e-01, -8.4109e-02, -1.6361e+00,
         1.0224e+00,  1.0733e+00, -5.7453e-01,  4.9668e-02,  7.2379e-01,
         5.9746e-01,  2.6966e+00],
       [ 2.7930e+00, -2.2745e+00, -2.3912e-01,  8.7498e-02,  1.4967e+00,
        -5.7016e-01, -5.7248e-01,  1.9909e+00, -7.4416e-01,  7.2960e-01,
         6.4083e-01,  1.6075e+00, -8.8810e-01,  2.7359e-01, -1.3257e-01,
         1.2710e+00,  1.7234e+00,  1.1180e-01,  2.6952e-01,  1.1835e+00,
         1.2575e+00,  1.3969e-01,  4.7259e-01,  7.9025e-01,  1.0811e+00,
        -9.1965e-01, -4.0503e-01],
       [ 4.5696e-01, -5.4184e-01, -2.3025e+00,  2.0127e+00, -4.6452e-01,
        -5.8270e-01,  2.0863e+00, -4.7729e-02, -4.4920e-01,  9.5566e-01,
        -1.4708e-01, -1.2532e+00, -1.1850e+00,  3.6583e-01, -1.4049e-01,
         3.5252e-01, -5.2400e-01, -6.2844e-01, -9.3792e-01,  1.6772e+00,
         3.8554e-03, -7.3685e-01, -9.3514e-01,  1.0465e-01, -4.6464e-01,
         1.6676e+00,  1.3931e+00],
       [ 6.5398e-01, -2.2449e-01,  1.2831e+00, -9.1787e-01, -3.3916e-01,
        -1.8058e+00,  6.0518e-01, -5.6252e-01, -7.8933e-01,  1.2767e+00,
        -1.0143e+00,  4.1611e-01, -7.5348e-01,  1.7128e+00, -8.7554e-01,
         3.9714e-01,  8.4326e-01,  3.7988e-01, -1.1670e+00,  5.5228e-01,
        -1.0279e+00, -3.9554e-01, -7.1410e-01, -8.7456e-02, -3.3361e-01,
        -1.8798e-01, -1.2647e+00],
       [ 2.0021e+00, -2.3470e-01, -1.3765e+00,  9.3426e-01,  1.0880e+00,
         1.9179e-01,  3.0114e-01,  8.9896e-01, -8.4454e-01,  2.3267e-01,
        -3.9205e-01, -2.5081e-01,  8.7124e-02,  1.3769e+00, -8.3358e-01,
        -8.9400e-01,  1.1744e+00, -6.0779e-01, -1.1493e-01, -7.8077e-01,
         1.9660e+00,  6.1175e-01,  3.6039e-01, -1.0274e+00,  1.1495e+00,
         4.5111e-01,  6.4420e-01],
       [ 2.1635e-01, -7.8731e-01, -3.3005e-01,  3.2877e-01, -1.6332e+00,
         1.0807e+00,  3.3638e-01,  1.1536e-01,  3.2834e-01,  5.3447e-02,
         1.4224e+00, -8.3957e-01, -2.4956e-01, -8.9778e-01, -8.6583e-01,
        -1.0786e+00, -1.8384e-01,  7.1622e-01,  1.8175e-01,  1.1053e+00,
         1.7003e+00, -1.6965e-01,  1.6293e-01,  1.3413e+00, -2.6301e-01,
        -7.5521e-01,  8.1911e-01],
       [ 7.4140e-01, -5.8787e-01, -4.6505e-01,  5.3112e-02,  2.2190e+00,
        -3.5158e-01,  3.6381e-01,  2.5769e+00,  1.4544e+00, -6.1003e-01,
        -5.9961e-01, -5.8392e-01, -1.8104e-02, -9.5177e-01, -9.6400e-01,
        -2.8183e-01,  1.0597e+00, -7.2370e-01,  1.4755e-01, -3.2667e-01,
         2.4958e+00,  1.1088e+00, -8.5476e-01,  1.8443e+00, -1.3881e-01,
         1.3096e+00, -2.5802e-01],
       [ 1.0669e+00,  2.1363e-01, -7.6603e-01, -1.6977e+00, -1.5023e-01,
        -5.2150e-01, -6.3730e-01,  2.6214e-01,  7.6539e-03,  1.3067e+00,
        -6.3482e-01, -1.1042e-04, -6.6158e-01,  1.4723e-01, -6.6036e-02,
         5.2851e-01,  5.7950e-01,  2.1438e-01,  9.2200e-01,  5.2919e-01,
         7.7070e-01,  4.2899e-01,  3.4330e-01,  2.0698e+00,  1.3405e+00,
        -2.1746e-01,  8.6273e-01]];
        let mut final_weights = vec![];
        for outer_vec in from_test_weights {
            for j in outer_vec {
               final_weights.push(j as f32);
            }
        }
        const NUMBER_OF_CHARACTERS: usize = 27;

        let mut weights = Tensor::from_vec(final_weights, Shape::new(vec![NUMBER_OF_CHARACTERS, NUMBER_OF_CHARACTERS]));
        // will need to do a "set_requires_grad" function
        weights.set_requires_grad(true);

        let (_stoi, _itos, xs, ys) = generate_dataset();
        let TEST_BATCH : usize = 10;//xs.len();
        // Prepare a batch of data
        let combined = xs.iter().take(TEST_BATCH).zip(ys.iter().take(TEST_BATCH));
        let mut inputs = vec![];
        let mut targets = vec![];
        for (x, y) in combined {
            let mut input = Tensor::zeroes(Shape::new(vec![1, NUMBER_OF_CHARACTERS]));
            input.set_index([0, (*x)].into(), vec![1.0].into());
            inputs.push(input);
            let mut target = Tensor::zeroes(Shape::new(vec![1, NUMBER_OF_CHARACTERS]));
            target.set_index([0, (*y)].into(), vec![1.0].into());
            targets.push(target);
        }
        const EPOCHS : usize = 250;

        for _ in 0..EPOCHS {
            let mut running_loss = 0.0;

            for (input, target) in inputs.iter().zip(&targets) {
                {
                    let mut singleton = SINGLETON_INSTANCE.lock().unwrap();
                    singleton.zero_all_grads();
                }
                let prediction = *input << weights;
                let counts = prediction.exp();
                let counts_sum = counts.sum();
                let probabilities = counts / counts_sum;
                // The below could be made redudent if we did not turn the target into a one hot encouder
                let target_index = target.item().iter().position(|&x| x == 1.0).unwrap();

                let view = probabilities.view([0, target_index].into());
                let bob = view;
                let logged = -(bob.log());
                logged.backward();

                running_loss += logged.item()[[0]];
                let mut singleton = SINGLETON_INSTANCE.lock().unwrap();
                singleton.update_parameters(-50.0);
            }
            println!("Loss: {}", running_loss / TEST_BATCH as f32);
        }
    }
}
