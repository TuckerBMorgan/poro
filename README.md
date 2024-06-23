# Poro
[![Current Crates.io Version](https://img.shields.io/crates/v/poro.svg?style=for-the-badge&logo=rust)](https://crates.io/crates/poro)
![Professor Poro](./icon.webp)

Poro is a library I am writing to help me better understand modern ML frameworks like [Pytorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/). It is mostly based off of Karpathys Micrograd series of (lectures)[https://www.youtube.com/watch?v=VMj-3S1tku0]. It is in rust because I enjoy its ease of setup.

## Features
- Basic neural network operations
- Frictionless Autograd
- Cuda support (limited by growing!)
- Support for custom layers and operations
- Lightweight and trying to focus on ease of understanding for others

## notes on usage
I work on this lib while I also have a job, and it is done for the enjoyment of learning, so it is likely a good idea to not use this for professional use as I am unlikely to get to any issues you might have in a timely manner :) 

if you run cargo test, some tests might fail, this is not actually them failing
do to the nature of the way I have the equation working as a singelton, it it possible for
test to grab the lock at the wrong time and fail. If you know how to fix this, that would make a great first PR :). You can always run the tests one at a time and see that they work that way

# Getting Started
```bash
cargo test --release
```

### Installation

To use Poro, add the following to your `Cargo.toml`:

```toml
[dependencies]
poro = "0.1.2"
```

### Usage

Here is a simple example to get you started with Poro:

```rust
use Poro::tensor::Tensor;
use ndarray::prelude::*;

fn main() {
    let a = Tensor::ones(Shape::new(vec![2, 2]));
    let b = Tensor::zeroes(Shape::new(vec![2, 2]));
    let c = a + b;
    let result = c.item();
    assert!(result == arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());
}
```

## Planned features

- Optimizer module
- Data Loader Module
- Working with Metal
- Transfomers Layer
- Conv Layer
- Model/Module Configure 
