use common::{Activation, RCParams};

/// The parameters required for any next generation reservoir computer
#[derive(Debug, Clone)]
pub struct Params {
    /// Number of taps from which input values are sampled
    pub num_time_delay_taps: usize,

    /// Number of input samples to skip. Basically spaces taps out
    pub num_samples_to_skip: usize,

    /// Activation function of output
    pub output_activation: Activation,

    /// The state size of reservoir, dictated by constructor used
    pub reservoir_size: usize,
}

impl RCParams for Params {
    #[inline(always)]
    fn initial_state_value(&self) -> f64 {
        0.0
    }

    #[inline(always)]
    fn reservoir_size(&self) -> usize {
        self.reservoir_size
    }
}
