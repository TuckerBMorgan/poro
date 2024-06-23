use poro::{update_parameters, Tensor};

fn main() {
    // You allocate tensors just like you would in PyTorch
    // Then automatically calculate the gradients
    // you do need to call set_requires_grad(true) on the tensors you want to calculate the gradients for
    let mut a = Tensor::randn(vec![1].into());
    a.set_requires_grad(true);
    let mut b = Tensor::randn(vec![1].into());
    b.set_requires_grad(true);

    let c = a + b;

    println!(
        "a {:?} + b {:?} = {:?}",
        a.item().to_string(),
        b.item().to_string(),
        c.item().to_string()
    );

    // And then simply call backward() on the result tensor
    c.backward();
    println!("a.grad: {:?}", a.grad().to_string());
    // And then you need to call update_parameters() to update the parameters
    update_parameters(-0.01);
}
