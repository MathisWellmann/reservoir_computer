use common::{Activation, RCParams};

/// The parameters of the Echo State Network
#[derive(Debug, Clone)]
pub struct Params {
    /// Probability of inputs connecting to state
    pub input_sparsity: f64,
    /// Activation function applied to result of input transformation
    pub input_activation: Activation,
    /// Scales the input weight matrix
    pub input_weight_scaling: f64,

    /// Number of nodes in the reservoir
    pub reservoir_size: usize,
    /// Scales the reservoir biases
    pub reservoir_bias_scaling: f64,
    /// Connection probability within the reservoir
    pub reservoir_sparsity: f64,
    /// Activation function of reservoir state transition
    pub reservoir_activation: Activation,

    /// Controls the retention of information from previous time steps.
    /// The spectral radius determines how fast the influence of an input
    /// dies out in a reservoir with time, and how stable the reservoir
    /// activations are. The spectral radius should be greater in tasks
    /// requiring longer memory of the input.
    pub spectral_radius: f64,
    /// Tunes the decay time of internal activity of the network
    /// The leaking rate a can be regarded as the speed of the reservoir
    /// update dynamics discretized in time. This can be adapted online
    /// to deal with time wrapping of the signals. Set the leaking rate
    /// to match the speed of the dynamics of input / target This sort
    /// of acts as a EMA smoothing filter, so other MAs could be used
    /// such as ALMA
    pub leaking_rate: f64,
    /// Ridge regression regulazation applied in training
    pub regularization_coeff: f64,
    /// Fraction of initial state transitions to disregard in training
    pub washout_pct: f64,
    /// Activation function of networks readout
    pub output_activation: Activation,
    /// Optional seed for Rng
    pub seed: Option<u64>,
    /// Fraction of noise to add to state update equation
    pub state_update_noise_frac: f64,
    /// Initial value of state
    pub initial_state_value: f64,
}

impl RCParams for Params {
    #[inline(always)]
    fn initial_state_value(&self) -> f64 {
        self.initial_state_value
    }

    #[inline(always)]
    fn reservoir_size(&self) -> usize {
        self.reservoir_size
    }
}

/*
/// Dimensionality of the parameter space
pub const PARAM_DIM: usize = 7;

pub struct ParamMapper {
    /// Parameter ranges
    pub input_sparsity_range: Range,
    pub input_activation: Activation,
    pub input_weight_scaling_range: Range,
    pub reservoir_size_range: Range,
    pub reservoir_bias_scaling_range: Range,
    pub reservoir_sparsity_range: Range,
    pub reservoir_activation: Activation,
    pub spectral_radius: f64,
    pub leaking_rate_range: Range,
    pub regularization_coeff_range: Range,
    pub washout_pct: f64,
    pub output_activation: Activation,
    pub seed: Option<u64>,
    pub state_update_noise_frac: f64,
    pub initial_state_value: f64,
}

impl OptParamMapper<PARAM_DIM> for ParamMapper {
    type Params = Params;

    fn map(&self, params: &[f64; PARAM_DIM]) -> Params {
        Params {
            input_sparsity: scale(
                0.0,
                1.0,
                self.input_sparsity_range.0,
                self.input_sparsity_range.1,
                params[0],
            ),
            input_activation: self.input_activation,
            input_weight_scaling: scale(
                0.0,
                1.0,
                self.input_weight_scaling_range.0,
                self.input_weight_scaling_range.1,
                params[1],
            ),
            reservoir_size: scale(
                0.0,
                1.0,
                self.reservoir_size_range.0,
                self.reservoir_size_range.1,
                params[2],
            ) as usize,
            reservoir_bias_scaling: scale(
                0.0,
                1.0,
                self.reservoir_bias_scaling_range.0,
                self.reservoir_bias_scaling_range.1,
                params[3],
            ),
            reservoir_sparsity: scale(
                0.0,
                1.0,
                self.reservoir_sparsity_range.0,
                self.reservoir_sparsity_range.1,
                params[4],
            ),
            reservoir_activation: self.reservoir_activation,
            spectral_radius: self.spectral_radius,
            leaking_rate: scale(
                0.0,
                1.0,
                self.leaking_rate_range.0,
                self.leaking_rate_range.1,
                params[5],
            ),
            regularization_coeff: scale(
                0.0,
                1.0,
                self.regularization_coeff_range.0,
                self.regularization_coeff_range.1,
                params[6],
            ),
            washout_pct: self.washout_pct,
            output_activation: self.output_activation,
            seed: self.seed,
            state_update_noise_frac: self.state_update_noise_frac,
            initial_state_value: self.initial_state_value,
        }
    }
}
*/
