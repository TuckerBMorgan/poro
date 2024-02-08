mod central;
pub use central::*;
pub use ndarray::prelude::*;

#[cfg(test)]
mod tests {

    use crate::*;

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
}
