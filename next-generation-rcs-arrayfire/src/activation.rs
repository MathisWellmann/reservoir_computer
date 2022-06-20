/// Lists all possible actiation functions
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Identity,
    Relu,
    Tanh,
}

impl Activation {
    /// Applies the activation function to the input value
    pub fn activate(&self, val: f32) -> f32 {
        match self {
            Self::Identity => val,
            Self::Relu => {
                if val < 0.0 {
                    0.0
                } else {
                    val
                }
            }
            Self::Tanh => val.tanh(),
        }
    }
}
