# Poro
[![Current Crates.io Version](https://img.shields.io/crates/v/poro.svg?style=for-the-badge&logo=rust)](https://crates.io/crates/poro)
![Professor Poro](./icon.webp)

Poro is a simple toy neural network library implemented in Rust. It is designed for educational purposes and provides basic functionality to create, train, and evaluate neural networks. This library is not intended for production use but serves as a learning tool for those interested in understanding the fundamentals of neural networks and their implementation.

## Features

- Basic neural network operations
- Frictionless Autograd
- Support for custom layers and operations
- Lightweight and easy to understand

## Getting Started

### Prerequisites

Ensure you have Rust installed on your system. You can install Rust using [rustup](https://rustup.rs/).

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

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes or additions.

## License

Poro is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
