use crate::Shape;

use super::tensor::TensorID;
/// Represents an indexable value.
/// This is used to index into a tensor.
/// It can be a single index, a double index, or an index from a tensor.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum Indexable {
    Single(usize),
    Double(usize, usize),
    FromTensor(TensorID),
}

impl Indexable {
    /// Returns the number of indices in the indexable.
    pub fn len(&self) -> usize {
        match self {
            Indexable::Single(_) => 1,
            Indexable::Double(_, _) => 2,
            Indexable::FromTensor(_) => 1,
        }
    }

    /// Returns the index at the given position.
    /// # Arguments
    /// * `shape` - The shape to get the index from.
    /// # Returns
    /// The index at the given position.
    /// # Panics
    /// This function will panic if the index is out of range.
    pub fn get_index(&self, shape: Shape) -> usize {
        match self {
            Indexable::Single(i) => {
                assert!(*i < shape.number_of_indices);
                shape.indices[*i]
            }
            Indexable::Double(_a, _b) => {
                panic!("Double index not implemented");
            }
            Indexable::FromTensor(_a) => {
                panic!("Mixed index not implemented");
            }
        }
    }
}

/// Convert an indexable into a shape.
/// This will return a shape with the indexable as the indices.
/// # Arguments
/// * `indexable` - The indexable to convert.
/// # Returns
/// A shape with the indexable as the indices.
/// # Panics
/// This function will panic if the indexable is a double index or a tensor index.
impl From<Indexable> for Shape {
    fn from(indexable: Indexable) -> Shape {
        match indexable {
            Indexable::Single(a) => vec![a].into(),
            Indexable::Double(a, b) => vec![a, b].into(),
            Indexable::FromTensor(_) => panic!("Mixed index not implemented"),
        }
    }
}

/// Convert an indexable into a vector of indices.
/// This will return a vector of indices with the indexable as the indices.
/// # Arguments
/// * `indexable` - The indexable to convert.
/// # Returns
/// A vector of indices with the indexable as the indices.
impl From<Indexable> for Vec<usize> {
    fn from(indexable: Indexable) -> Vec<usize> {
        match indexable {
            Indexable::Single(index) => vec![index],
            Indexable::Double(a, b) => vec![a, b],
            Indexable::FromTensor(_) => vec![1],
        }
    }
}

/// Convert a usize into an indexable.
/// This will return a single index indexable.
/// # Arguments
/// * `index` - The index to convert.
/// # Returns
/// A single index indexable.
impl From<usize> for Indexable {
    fn from(index: usize) -> Indexable {
        Indexable::Single(index)
    }
}

/// Convert a tuple of two usizes into an indexable.
/// This will return a double index indexable.
/// # Arguments
/// * `index` - The index to convert.
/// # Returns
/// A double index indexable.
impl From<[usize; 2]> for Indexable {
    fn from([a, b]: [usize; 2]) -> Indexable {
        Indexable::Double(a, b)
    }
}
