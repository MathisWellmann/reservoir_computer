/// The possible activation functions to apply to the output of reservoir computers
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// The identity function
    Identity,
    /// The hyperbolic tangent
    Tanh,
    /// The rectified linear unit
    Relu,
}

impl Activation {
    /// Perform the activation function over all elements
    pub fn activate(&self, vals: &mut [f64]) {
        match self {
            Activation::Identity => {}
            Activation::Tanh => {
                for v in vals {
                    *v = v.tanh();
                }
            }
            Activation::Relu => {
                for v in vals {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
    }
}
