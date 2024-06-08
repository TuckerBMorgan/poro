use crate::{get_equation, Shape};

use super::tensor::TensorID;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum Indexable {
    Single(usize),
    Double(usize, usize),
    Mixed(TensorID, TensorID),
    FromTensor(TensorID),
}

impl Indexable {
    pub fn len(&self) -> usize {
        match self {
            Indexable::Single(_) => 1,
            Indexable::Double(_, _) => 2,
            Indexable::Mixed(_, _) => 2,
            Indexable::FromTensor(_) => 1,
        }
    }

    pub fn get_index(&self, shape: Shape) -> usize {
        match self {
            Indexable::Single(i) => {
                assert!(*i < shape.number_of_indices);
                shape.indices[*i]
            }
            Indexable::Double(_a, _b) => {
                panic!("Double index not implemented");
            }
            Indexable::Mixed(_a, _b) => {
                panic!("Mixed index not implemented");
            }
            Indexable::FromTensor(_a) => {
                panic!("Mixed index not implemented");
            }
        }
    }
}

impl From<Indexable> for Shape {
    fn from(indexable: Indexable) -> Shape {
        match indexable {
            Indexable::Single(a) => vec![a].into(),
            Indexable::Double(a, b) => vec![a, b].into(),
            Indexable::Mixed(_a, _b) => panic!("Mixed index not implemented"),
            Indexable::FromTensor(_) => panic!("Mixed index not implemented"),
        }
    }
}

impl From<Indexable> for Vec<usize> {
    fn from(indexable: Indexable) -> Vec<usize> {
        match indexable {
            Indexable::Single(index) => vec![index],
            Indexable::Double(a, b) => vec![a, b],
            Indexable::Mixed(_a, _b) => vec![1, 1],
            Indexable::FromTensor(_) => vec![1],
        }
    }
}

impl From<usize> for Indexable {
    fn from(index: usize) -> Indexable {
        Indexable::Single(index)
    }
}

impl From<[usize; 2]> for Indexable {
    fn from([a, b]: [usize; 2]) -> Indexable {
        Indexable::Double(a, b)
    }
}

impl From<[Vec<f32>; 2]> for Indexable {
    fn from([a, b]: [Vec<f32>; 2]) -> Indexable {
        let mut singelton = get_equation();
        let a_as_tensor = singelton.allocate_tensor(
            Shape::new(vec![a.len()]),
            a,
            super::operation::Operation::Nop,
        );
        let b_as_tensor = singelton.allocate_tensor(
            Shape::new(vec![b.len()]),
            b,
            super::operation::Operation::Nop,
        );
        Indexable::Mixed(a_as_tensor, b_as_tensor)
    }
}
