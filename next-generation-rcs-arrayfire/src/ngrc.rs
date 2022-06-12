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
