# Poro

![Professor Poro](./icon.webp)

Poro is a simple toy neural network library implemented in Rust. It is designed for educational purposes and provides basic functionality to create, train, and evaluate neural networks. This library is not intended for production use but serves as a learning tool for those interested in understanding the fundamentals of neural networks and their implementation.

## Features

- Basic neural network operations
- Tensor manipulation
- Support for custom layers and operations
- Lightweight and easy to understand

## Getting Started

### Prerequisites

Ensure you have Rust installed on your system. You can install Rust using [rustup](https://rustup.rs/).

### Installation

To use Poro, add the following to your `Cargo.toml`:

```toml
[dependencies]
poro = "0.1.0"
```

### Usage

Here is a simple example to get you started with Poro:

```rust
use Poro::model::Model;
use Poro::tensor::Tensor;
use Poro::operation::Operation;

fn main() {
    // Create a simple neural network
    let model = Model::new();

    // Define input tensor
    let input = Tensor::new(vec![1.0, 2.0, 3.0]);

    // Perform a forward pass
    let output = model.forward(input);

    // Print the output
    println!("{:?}", output);
}
```

## Modules

### `equation.rs`

Contains the implementation of various mathematical operations required for neural network computations.

### `indexable.rs`

Provides traits and implementations for indexing tensors and other data structures.

### `internal_tensor.rs`

Defines the internal representation of tensors and related operations.

### `mod.rs`

The main module that ties together all components of the Poro library.

### `model.rs`

Defines the structure and behavior of neural network models.

### `module.rs`

Contains the definitions of different layers and modules that can be used to build neural networks.

### `operation.rs`

Implements various operations that can be performed on tensors and within the neural network.

### `shape.rs`

Provides utilities for handling tensor shapes and dimensions.

### `tensor.rs`

Defines the tensor structure and provides methods for tensor manipulation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes or additions.

## License

Poro is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
