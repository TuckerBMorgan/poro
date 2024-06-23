use poro::Tensor;

fn main() {
    
    // You allocate tensors just like you would in PyTorch
    let a = Tensor::randn(vec![1].into());
    let b = Tensor::randn(vec![1].into()); 
    print!("You can add tensors together a {:?} + b {:?}", a.item().to_string(), b.item().to_string());
    // And then you can treat them like you would in PyTorch
    let c = a + b;

    println!(" = {:?}", c.item().to_string());

    // They support most basic math operations
    let d = Tensor::randn(vec![1].into());
    let e = Tensor::randn(vec![1].into());
    print!("You can multiply them as well d {:?} + e {:?}", d.item().to_string(), e.item().to_string());
    let f = d * e;
    println!(" = {:?}", f.item().to_string());

    // And you can chain them together
    let g = Tensor::randn(vec![1].into());
    let h = Tensor::randn(vec![1].into());
    let i = Tensor::randn(vec![1].into());
    print!("You can chain them together g {:?} + h {:?} * i {:?}", g.item().to_string(), h.item().to_string(), i.item().to_string());
    let j = g + h * i;

    println!(" = {:?}", j.item().to_string());
}