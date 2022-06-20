use crate::Activation;

/// The parameters required for any next generation reservoir computer
#[derive(Debug, Clone)]
pub struct Params {
    /// Number of taps from which input values are sampled
    pub num_time_delay_taps: usize,

    /// Number of input samples to skip. Basically spaces taps out
    pub num_samples_to_skip: usize,

    /// Activation function of output
    pub output_activation: Activation,

    /// Ridge regression regularization coefficient
    pub regularization_coeff: f32,
}

impl Params {
    /// Computes the number of data steps required for warmup
    pub fn warmup_steps(&self) -> usize {
        self.num_time_delay_taps * self.num_samples_to_skip
    }
}
