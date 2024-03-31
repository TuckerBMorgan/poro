use crate::Shape;

use super::shape;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum Indexable {
    Single(usize),
    Double(usize, usize),    
    Range(usize, usize, usize)
}

impl Indexable {
    pub fn len(&self) -> usize {
        match self {
            Indexable::Single(_) => 1,
            Indexable::Double(_, _) => 2,
            Indexable::Range(start, end, step) => ((end - start) as f64 / *step as f64).ceil() as usize
        }
    }

    pub fn get_index(&self, shape: Shape) -> usize {
        match self {
            Indexable::Single(i) => {
                assert!(*i < shape.number_of_indices);
                shape.indices[*i]
            },
            Indexable::Double(a, b) => {
                panic!("Double index not implemented");
            },
            Indexable::Range(start, end, step) => {
                ((end - start) as f64 / *step as f64).ceil() as usize
            }
        }
    
    }
}

impl From<Indexable> for Shape {
    fn from(indexable: Indexable) -> Shape {
        match indexable {
            Indexable::Single(a) => vec![a].into(),
            Indexable::Double(a, b) => vec![a, b].into(),
            Indexable::Range(start, end, step) => vec![((end - start) as f64 / step as f64).ceil() as usize].into()
        }
    }
}

impl From<Indexable> for Vec<usize> {
    fn from(indexable: Indexable) -> Vec<usize> {
        match indexable {
            Indexable::Single(index) => vec![index],
            Indexable::Double(a, b) => vec![a, b],
            Indexable::Range(start, end, step) => (start..end).step_by(step).collect()
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