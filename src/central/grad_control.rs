pub struct NoGrad {

}

impl NoGrad {
    pub fn new() -> NoGrad {
        let mut equation = super::get_equation();
        equation.disable_grad();
        NoGrad {}
    }
}

impl Drop for NoGrad {
    fn drop(&mut self) {
        let mut equation = super::get_equation();
        equation.enable_grad();
    }
}