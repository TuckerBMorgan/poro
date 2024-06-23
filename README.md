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