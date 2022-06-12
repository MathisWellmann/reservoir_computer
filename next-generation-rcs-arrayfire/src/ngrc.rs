use arrayfire::{Array, Dim4};

use crate::params::Params;

pub struct NextGenerationRC {
    params: Params,
}

impl NextGenerationRC {
    pub fn new(params: Params) -> Self {
        Self {
            params,
        }
    }

    pub fn construct_lin_part(&self, inputs: &[f32]) -> Array<f32> {
        let nrows = inputs.len();
        let ncols = self.params.num_time_delay_taps;

        let mut values: Vec<f32> = Vec::with_capacity(nrows * ncols);

        for delay in 0..self.params.num_time_delay_taps {
            let mut column = vec![0.0; nrows];
            for j in delay * self.params.num_samples_to_skip..nrows {
                column[j] = inputs[j - delay * self.params.num_samples_to_skip];
            }
            values.append(&mut column);
        }

        info!("values.len(): {}", values.len());

        Array::new(&values, Dim4::new(&[nrows as u64, ncols as u64, 1, 1]))
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32]) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayfire::*;
    use common::Activation;

    const NUM_VALS: usize = 9;

    fn get_inputs() -> [f32; NUM_VALS] {
        [0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0]
    }

    #[test]
    fn ngrc_arrayfire_lin_part() {
        if let Err(_) = pretty_env_logger::try_init() {}

        set_device(0);
        info();

        let d_lin = 3;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let reservoir_size = d_lin + d_nonlin;

        let params = Params {
            num_time_delay_taps: d_lin,
            num_samples_to_skip: 1,
            output_activation: Activation::Tanh,
            reservoir_size,
        };
        let rc = NextGenerationRC::new(params);

        let inputs = get_inputs();
        let lin_part = rc.construct_lin_part(&inputs);

        af_print!("lin_part", lin_part);

        let goal = vec![
            0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0, 0.0, 0.0, 0.55, 1.0, 0.45, 0.0,
            -0.55, -1.0, -0.45, 0.0, 0.0, 0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0,
        ];
        // TODO: how to compare two Arrays?
    }
}
