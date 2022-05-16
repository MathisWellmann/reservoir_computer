use crate::{Activation, OptParamMapper, RCParams};

pub(crate) const PARAM_DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct Params {
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_time_delay_taps: usize,
    pub num_samples_to_skip: usize,
    pub output_activation: Activation,
}

impl RCParams for Params {
    #[inline(always)]
    fn initial_state_value(&self) -> f64 {
        0.0
    }

    #[inline(always)]
    fn reservoir_size(&self) -> usize {
        // TODO: enable this to work in more dimensions
        const INPUT_DIM: usize = 1;
        let d_lin = self.num_time_delay_taps * INPUT_DIM;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        d_lin + d_nonlin
    }
}

pub struct ParamMapper {}

impl OptParamMapper<PARAM_DIM> for ParamMapper {
    type Params = Params;

    fn map(&self, params: &[f64; PARAM_DIM]) -> Self::Params {
        todo!()
    }
}
