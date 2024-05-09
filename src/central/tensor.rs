use core::slice;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Shl, Sub};
use ndarray::{prelude::*, Slice};

use super::{operation::Operation, shape::Shape, indexable::Indexable};

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash, Debug)]
pub struct TensorID {
    pub id: usize
}

impl TensorID {
    pub fn new(id: usize) -> TensorID {
        TensorID {
            id
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub tensor_id: TensorID,
    pub shape: Shape,
    pub operation: Operation,
    pub name: [char;10]
}

impl Tensor {

    pub fn randn(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_randn_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }
    /*
    pub fn load_from_weight_file(shape: Shape, path: String) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_tensor_from_weight_file(shape.clone(), path, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }
 */
    pub fn get_id(&self) -> usize {
        self.tensor_id.id
    }

    pub fn set_name(&mut self, name: [char;10]) {
        self.name = name;
    }

    pub fn zeroes(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_zero_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn ones(shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_ones_tensor(shape.clone(), Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn element(shape: Shape, data: f32) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_element_tensor(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_tensor_from_operation(shape.clone(), data, Operation::Nop);
        Tensor {
            tensor_id,
            shape,
            operation: Operation::Nop,
            name: ['a';10]
        }
    }

    pub fn item(&self) -> ArrayD<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone();
        let data = data.as_slice();
        let data = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
        data
    }

    pub fn backward(&self) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.backward(self.tensor_id);
    }

    pub fn grad(&self) -> ArrayD<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_tensor_grad(self.tensor_id).clone();
        let data = data.as_slice().unwrap();
        let data = ArrayD::from_shape_vec(self.shape.as_ndarray_shape(), data.to_vec()).unwrap();
        data
    }

    pub fn exp(&self) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone().into_iter().map(|x| x.exp()).collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data, Operation::Exp(self.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Exp(self.tensor_id),
            name: ['a';10]
        }
    }


    pub fn pow(&self, power: f32) -> Tensor {
        let power_as_tensor = Tensor::element(vec![1].into(), power);
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone().into_iter().map(|x| x.powf(power)).collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data, Operation::Pow(self.tensor_id, power_as_tensor.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Pow(self.tensor_id, power_as_tensor.tensor_id),
            name: ['a';10]
        }
    }

    pub fn tanh(&self) -> Tensor {
        let hold = Tensor::element(Shape::new(vec![1]), 2.0);
        let k = *self * hold;
        let e = k.exp();
        let o = (e - 1.0) / (e + 1.0);
        return o;
    }

    pub fn sum(&self) -> Tensor {

        // I want to sum along the first axis
        let data = self.item().sum_axis(Axis(1)).clone().into_iter().map(|x| x).collect();
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let tensor_id = singleton.allocate_tensor(Shape::new(vec![self.shape.indices[0], 1]), data, Operation::Sum(self.tensor_id));

        Tensor {
            tensor_id,
            shape: Shape::new(vec![ self.shape.indices[0], 1]),
            operation: Operation::Sum(self.tensor_id),
            name: ['a';10]
        }
    }
    
    pub fn mean(&self, ) -> Tensor {
        // a mean is a sum and a div
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data : f32 = singleton.get_item(self.tensor_id).clone().into_iter().sum();
        let data = data / self.shape.size() as f32;
        let tensor_id = singleton.allocate_element_tensor(Shape::new(vec![1]), data, Operation::Mean(self.tensor_id));

        Tensor {
            tensor_id,
            shape: Shape::new(vec![1]),
            operation: Operation::Mean(self.tensor_id),
            name: ['a';10]
        }
    }

    pub fn t_mean(tensors: &Vec<Tensor>) -> Tensor {
        for tensor in tensors {
            if tensor.shape.size() != 1 {
                panic!("All tensors must be of size 1");
            }
        }

        let mut tensor = Tensor::element(vec![1].into(), 0.0);
        for sum_tensor in tensors {
            tensor = tensor + *sum_tensor;
        }

        return tensor / tensors.len() as f32;
    }

    pub fn log(&self) -> Tensor {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data = singleton.get_item(self.tensor_id).clone().into_iter().map(|x| x.ln()).collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data, Operation::Log(self.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Log(self.tensor_id),
            name: ['a';10]
        }
    }

    pub fn set_index(&mut self, indexable: Indexable, data: Vec<f32>) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.set_subset_of_tensor_data(self.tensor_id, indexable, data);
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.set_requires_grad(self.tensor_id, requires_grad);
    }

    pub fn broadcast(&self, shape: Shape) -> Tensor {

        let data: Vec<f32> = self.item().broadcast(shape.as_ndarray_shape()).unwrap().iter().map(|x|*x).collect();

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        // We need to dubpilcate the data to match the new shape

        let tensor_id = singleton.allocate_tensor_from_operation(shape.clone(), data, Operation::Broadcast(self.tensor_id, shape));

        Tensor {
            tensor_id,
            shape,
            operation: Operation::Broadcast(self.tensor_id, shape),
            name: ['a';10]
        }
    }

    pub fn view(&self, index: Indexable) -> Tensor {
        // Allocate a new tensor the size of the view
        // and then set the data of the new tensor to the data of the old tensor
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let data : Vec<f32> = singleton.get_item(self.tensor_id);
        // I now need to get the index subset of data from the old tensor
        let new_shape = self.shape.subshape_from_indexable(index);
        match index {
            Indexable::Single(i) => {
                let data = data[i];
                let tensor_id = singleton.allocate_element_tensor(new_shape, data, Operation::View(self.tensor_id, index));
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a';10]
                }
            },
            Indexable::Double(a, b) => {
                let offset = a * self.shape.indices[1] + b;
                let data = data[offset];
                let tensor_id = singleton.allocate_element_tensor(new_shape, data, Operation::View(self.tensor_id, index));
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a';10]
                }
            },
            Indexable::Mixed(a, b) => {
                // Look up the A and B vectors, and then use the B vector to pick the indices from the A vector
                let a_data = singleton.get_item(a);
                let b_data = singleton.get_item(b);
                let mut new_data = Vec::new();
                for i in 0..b_data.len() {
                    let index = b_data[i] as usize;
                    new_data.push(a_data[index]);
                }
                let tensor_id = singleton.allocate_tensor_from_operation(new_shape, new_data, Operation::View(self.tensor_id, index));
                return Tensor {
                    tensor_id,
                    shape: new_shape,
                    operation: Operation::View(self.tensor_id, index),
                    name: ['a';10]
                }
            }

        }
        /*
        let tensor_id = singleton.allocate_tensor(self.shape.subshape_from_indexable(index), vec![], Operation::View(self.tensor_id));

        Tensor {
            tensor_id,
            shape:vec![1].into(),
            operation: Operation::View(self.tensor_id),
            name: ['a';10]
        }
 */
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let result_data = self.item() + rhs.item();

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

        let data_as_vec = result_data.into_iter().collect();
        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Add(self.tensor_id, rhs.tensor_id));

        Tensor {
            tensor_id,
            shape: self.shape,
            operation: Operation::Add(self.tensor_id, rhs.tensor_id),
            name: ['a';10]
        }
    }
}

impl Add<f32> for Tensor {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self + right_hand_as_tesnor
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            // we need to broadcast the tensors
            let broaded_casted_rhs = rhs.broadcast(self.shape);
            let result_data = self.item() * broaded_casted_rhs.item();

            let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
    
            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Mul(self.tensor_id, broaded_casted_rhs.tensor_id));
    
            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Mul(self.tensor_id, broaded_casted_rhs.tensor_id),
                name: ['a';10]
            }
        }
        else {
            let result_data = self.item() * rhs.item();

            let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
    
            let data_as_vec = result_data.into_iter().collect();
            let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), data_as_vec, Operation::Mul(self.tensor_id, rhs.tensor_id));
    
            Tensor {
                tensor_id,
                shape: self.shape,
                operation: Operation::Mul(self.tensor_id, rhs.tensor_id),
                name: ['a';10]
            }
        }

    }

}


impl Mul<f32> for Tensor {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self * right_hand_as_tesnor
    }
}

impl Neg for Tensor {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}


impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {

        let intermidiate = rhs.pow(-1.0);
        self * intermidiate
    }
}

impl Div<f32> for Tensor {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self / right_hand_as_tesnor
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs * self
    }
}

impl Sub<Tensor> for f32 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(rhs.shape.clone(), self);
        right_hand_as_tesnor - rhs
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Self::Output {
        let right_hand_as_tesnor = Tensor::element(self.shape.clone(), rhs);
        self - right_hand_as_tesnor
    }
}

impl Add<Tensor> for f32 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        rhs + self
    }
} 

// SIN: reusing the Shl opeartor to do the matmul operations
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {

        

        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

        let result_data = singleton.matmul(self.tensor_id, rhs.tensor_id);

        let tensor_id = singleton.allocate_tensor_from_operation(self.shape.clone(), result_data.into_raw_vec(), Operation::MatMul(self.tensor_id, rhs.tensor_id));
        let matmul_shape = self.shape.matmul_shape(&rhs.shape);
        
        Tensor {
            tensor_id,
            shape: matmul_shape,
            operation: Operation::MatMul(self.tensor_id, rhs.tensor_id),
            name: ['a';10]
        }
    }
}
