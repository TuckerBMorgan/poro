use crate::Indexable;

pub const MAX_NUMBER_OF_INDICES: usize = 10;

/// Represents the shape of a tensor.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub struct Shape {
    pub number_of_indices: usize,
    pub indices: [usize; MAX_NUMBER_OF_INDICES],
}

impl Shape {
    /// Creates a new `Shape` instance with the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A vector of indices representing the shape.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of indices exceeds the maximum number of indices.
    pub fn new(indices: Vec<usize>) -> Shape {
        assert!(indices.len() <= MAX_NUMBER_OF_INDICES);
        let mut local_indices = [0; MAX_NUMBER_OF_INDICES];
        for (i, index) in indices.iter().enumerate() {
            local_indices[i] = *index;
        }

        Shape {
            number_of_indices: indices.len(),
            indices: local_indices,
        }
    }

    /// Returns the total size of the shape.
    pub fn total_size(&self) -> usize {
        let mut total = 1;
        for i in 0..self.number_of_indices {
            total *= self.indices[i];
        }
        total
    }

    pub fn remove(&self, index: usize) -> Shape {
        let mut new_indices = Vec::new();
        for i in 0..self.number_of_indices {
            if i != index {
                new_indices.push(self.indices[i]);
            }
        }
        Shape::new(new_indices)
    }

    /// Returns the shape as a vector of usize values.
    pub fn as_ndarray_shape(&self) -> Vec<usize> {
        let mut shape = Vec::new();
        for i in 0..self.number_of_indices {
            shape.push(self.indices[i]);
        }
        shape
    }

    /// Returns the shape resulting from matrix multiplication with another shape.
    ///
    /// # Arguments
    ///
    /// * `other` - The shape to multiply with.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes are not compatible for matrix multiplication.
    pub fn matmul_shape(&self, other: &Shape) -> Shape {

        if self.number_of_indices == 1 && other.number_of_indices == 2 {
            assert!(self.indices[0] == other.indices[0]);
            return Shape::new(vec![other.indices[1]]);
        }

        if self.number_of_indices == 2 && other.number_of_indices == 1 {
            assert!(self.indices[1] == other.indices[0], "{} != {}", self.indices[1], other.indices[0]);
            return Shape::new(vec![self.indices[0]]);
        }
        if self.number_of_indices == 2 && other.number_of_indices == 2 {
            assert!(self.indices[1] == other.indices[0], "{} != {}", self.indices[1], other.indices[0]);
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
        if self.number_of_indices == 3 && other.number_of_indices == 1 {
            assert!(self.indices[2] == other.indices[0]);
            let mut new_indices = Vec::new();
            new_indices.push(self.indices[0]);
            new_indices.push(self.indices[1]);
            return Shape::new(new_indices);
        }

        if self.number_of_indices == 4 && other.number_of_indices == 4 {
            assert!(self.indices[3] == other.indices[2]);
            let mut new_indices = Vec::new();
            new_indices.push(self.indices[0]);
            new_indices.push(self.indices[1]);
            new_indices.push(self.indices[2]);
            new_indices.push(other.indices[3]);
            return Shape::new(new_indices);
        }


        panic!("Not implemented");
    }


    pub fn matmul_shape_generic(&self, other: &Shape) -> Shape {
        // Check if both shapes have at least 1 axis
        assert!(self.number_of_indices >= 1 && other.number_of_indices >= 1);
    
        // Matrices must satisfy matrix multiplication rules:
        // The last dimension of self (left) should match the first dimension of other (right).
        assert!(self.indices[self.number_of_indices - 1] == other.indices[0]);
    
        // Build the resulting shape
        let mut new_indices = Vec::new();
    
        // Broadcasting the leading dimensions (all dimensions except the last for `self`
        // and all dimensions except the first for `other`)
        for i in 0..(self.number_of_indices - 1).max(other.number_of_indices - 1) {
            if i < self.number_of_indices - 1 && i < other.number_of_indices - 1 {
                // Both matrices have this dimension, so we need to broadcast them
                assert!(self.indices[i] == 1 || other.indices[i + 1] == 1 || self.indices[i] == other.indices[i + 1]);
                new_indices.push(self.indices[i].max(other.indices[i + 1]));
            } else if i < self.number_of_indices - 1 {
                // Only `self` has this dimension
                new_indices.push(self.indices[i]);
            } else {
                // Only `other` has this dimension
                new_indices.push(other.indices[i + 1]);
            }
        }
    
        // Append the final dimension from the right matrix
        new_indices.push(other.indices[other.number_of_indices - 1]);
    
        Shape::new(new_indices)
    }
    

    /// Returns the size of the shape.
    pub fn size(&self) -> usize {
        let mut size = 1;
        for i in 0..self.number_of_indices {
            size *= self.indices[i];
        }
        size
    }

    /// Returns the index at the specified position.
    ///
    /// # Arguments
    ///
    /// * `index` - The index position.
    ///
    /// # Panics
    ///
    /// This function will panic if the index is out of range.
    pub fn get_index(&self, index: Indexable) -> usize {
        match index {
            Indexable::Single(i) => {
                assert!(i < self.number_of_indices);
                self.indices[i]
            }
            Indexable::Double(_a, _b) => {
                panic!("Not implemented");
            }
            Indexable::Triple(_a, _b, _c) => {
                panic!("Not implemented");
            }
            Indexable::FromTensor(_) => {
                panic!("Not implemented");
            }
        }
    }

    /// Returns a subshape of the current shape based on the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to create the subshape from.
    pub fn subshape_from_indexable(&self, index: Indexable) -> Shape {
        match index {
            Indexable::Single(_i) => {
                assert!(self.indices.len() > 1);
                let mut new_indices = vec![];
                for j in 1..self.number_of_indices {
                    new_indices.push(self.indices[j]);
                }
                Shape::new(new_indices)
            }
            Indexable::Double(_a, _b) => {
                if self.number_of_indices == 2 {
                    return Shape::new(vec![1]);
                }

                let mut new_indices = vec![];
                for j in 1..self.number_of_indices {
                    new_indices.push(self.indices[j]);
                }
                Shape::new(new_indices)
            }
            Indexable::Triple(_a, _b, _c) => {
                if self.number_of_indices == 3 {
                    return Shape::new(vec![1]);
                }

                let mut new_indices = vec![];
                for j in 1..self.number_of_indices {
                    new_indices.push(self.indices[j]);
                }
                Shape::new(new_indices)
            }
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
