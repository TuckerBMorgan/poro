use crate::Indexable;

pub const MAX_NUMBER_OF_INDICES : usize = 10;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub struct Shape {
    pub number_of_indices: usize,
    pub indices: [usize; MAX_NUMBER_OF_INDICES]
}

impl Shape {
    pub fn new(indices: Vec<usize>) -> Shape {
        assert!(indices.len() <= MAX_NUMBER_OF_INDICES);
        let mut local_indices = [0; MAX_NUMBER_OF_INDICES];
        for (i, index) in indices.iter().enumerate() {
            local_indices[i] = *index;
        }

        Shape {
            number_of_indices: indices.len(),
            indices: local_indices
        }
    }

    pub fn total_size(&self) -> usize {
        let mut total = 1;
        for i in 0..self.number_of_indices {
            total *= self.indices[i];
        }
        total
    }

    pub fn as_ndarray_shape(&self) -> Vec<usize> {
        let mut shape = Vec::new();
        for i in 0..self.number_of_indices {
            shape.push(self.indices[i]);
        }
        shape
    }

    pub fn matmul_shape(&self, other: &Shape) -> Shape {
        if self.number_of_indices == 2 && other.number_of_indices == 2 {
            assert!(self.indices[1] == other.indices[0]);
            let mut new_indices = Vec::new();
            new_indices.push(self.indices[0]);
            new_indices.push(other.indices[1]);
            return Shape::new(new_indices);
        }
        if self.number_of_indices == 3 && other.number_of_indices == 2 {
            assert!(self.indices[2] == other.indices[0]);
            let mut new_indices = Vec::new();
            new_indices.push(self.indices[0]);
            new_indices.push(self.indices[1]);
            new_indices.push(other.indices[1]);
            return Shape::new(new_indices);
        }
        if self.number_of_indices == 3 && other.number_of_indices == 3 {
            assert!(self.indices[2] == other.indices[1]);
            let mut new_indices = Vec::new();
            new_indices.push(self.indices[0]);
            new_indices.push(self.indices[1]);
            new_indices.push(other.indices[2]);
            return Shape::new(new_indices);
        }
        panic!("Not implemented");
    }

    pub fn size(&self) -> usize {
        let mut size = 1;
        for i in 0..self.number_of_indices {
            size *= self.indices[i];
        }
        size
    }

    pub fn get_index(&self, index: Indexable) -> usize {
        match index {
            Indexable::Single(i) => {
                assert!(i < self.number_of_indices);
                self.indices[i]
            },
            Indexable::Double(_a, _b) => {
                panic!("Not implemented");
            },
            Indexable::Mixed(_range_start, _range_end) => {
                panic!("Not implemented");
            },
            Indexable::FromTensor(_) => {
                panic!("Not implemented");
            }
        }
    }

    pub fn subshape_from_indexable(&self, index: Indexable) -> Shape {
        match index {
            Indexable::Single(_i) => {
                assert!(self.indices.len() > 1);
                let mut new_indices = vec![];
                for j in 1..self.number_of_indices {
                    new_indices.push(self.indices[j]);
                }
                Shape::new(new_indices)
            },
            Indexable::Double(_a, _b) => {
                if self.number_of_indices == 2 {
                    return Shape::new(vec![1]);
                }

                let mut new_indices = vec![];
                for j in 1..self.number_of_indices {
                    new_indices.push(self.indices[j]);
                }
                Shape::new(new_indices)

            },
            Indexable::Mixed(_range_start, _range_end) => {
                return Shape::new(self.indices.to_vec());
            },
            Indexable::FromTensor(_tensor) => {
                return Shape::new(self.indices.to_vec());

            }
        }
    
    }
}

impl From<Vec<usize>> for Shape {
    fn from(indices: Vec<usize>) -> Shape {
        Shape::new(indices)
    }
}   