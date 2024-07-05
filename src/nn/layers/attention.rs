use crate::nn::layers::LinearLayer;
use crate::nn::Module;
use crate::Tensor;

pub struct AttentionHead {
    pub q: LinearLayer,
    pub k: LinearLayer,
    pub v: LinearLayer,
    //    pub attention: Attention,
    pub output: LinearLayer,
}

impl AttentionHead {
    pub fn new(d_model: usize, d_k: usize, d_v: usize) -> Self {
        AttentionHead {
            q: LinearLayer::new(d_model, d_k),
            k: LinearLayer::new(d_model, d_k),
            v: LinearLayer::new(d_model, d_v),
            //            attention: Attention::new(),
            output: LinearLayer::new(d_v, d_model),
        }
    }
}

impl Module for AttentionHead {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let q = self.q.forward(input);
        let k = self.k.forward(input);
        let v = self.v.forward(input);

        let attention = q << k.tranpose_with_provided_axis(1, 2);
        let attention = attention / (k.shape.indices[1] as f32).sqrt();

        let attention = attention.softmax(2);
        let attention = attention << v;

        let output = self.output.forward(&attention);
        return output;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.extend(self.q.get_parameters());
        parameters.extend(self.k.get_parameters());
        parameters.extend(self.v.get_parameters());
        parameters.extend(self.output.get_parameters());
        return parameters;
    }
}
