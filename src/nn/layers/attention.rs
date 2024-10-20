use crate::nn::layers::LinearLayer;
use crate::nn::Module;
use crate::Tensor;
use crate::nn::layers::linear::LinearLayerConfig;

pub struct AttentionHead {
    pub q: LinearLayer,
    pub k: LinearLayer,
    pub v: LinearLayer,
    pub mask: Option<Tensor>
}

impl AttentionHead {
    pub fn new(d_model: usize, heads: usize) -> Self {
        let common_config = LinearLayerConfig {
            number_of_inputs: d_model,
            number_of_weights: heads
        };
        AttentionHead {
            q: LinearLayer::new(common_config),
            k: LinearLayer::new(common_config),
            v: LinearLayer::new(common_config),
            mask: None
        }
    }

    pub fn from_pretrained(q: LinearLayer, k: LinearLayer, v: LinearLayer) -> Self {
        AttentionHead {
            q,
            k,
            v,
            mask: None
        }
    }

    pub fn set_mask(&mut self, mask: Tensor) {
        self.mask = Some(mask);
    }
}

impl Module for AttentionHead {
    fn forward(&mut self, input: &Tensor) -> Tensor {

        let q = self.q.forward(input);
        let k = self.k.forward(input);
        let v = self.v.forward(input);
        let attention = q << k.tranpose_with_provided_axis(1, 0);
        let mut attention = attention / (k.shape.indices[1] as f32).sqrt();

        if self.mask.is_some() {
            attention = attention * self.mask.unwrap();
        }
        println!("{:?}", attention.shape);
        let attention = attention.softmax(attention.shape.number_of_indices - 1);

        let attention = attention << v;

        return attention;
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        let mut parameters = Vec::new();
        parameters.extend(self.q.get_parameters());
        parameters.extend(self.k.get_parameters());
        parameters.extend(self.v.get_parameters());
        return parameters;
    }
}
