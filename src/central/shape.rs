#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Shape {
    pub indices: Vec<usize>
}

impl Shape {
    pub fn new(indices: Vec<usize>) -> Shape {
        Shape {
            indices
        }
    }

    pub fn total_size(&self) -> usize {
        let mut total = 1;
        for i in &self.indices {
            total *= i;
        }
        total
    }

    pub fn as_ndarray_shape(&self) -> Vec<usize> {
        self.indices.clone()
    }
}