use std::time::Instant;

use arrayfire::{col, mul, rows, set_col, set_cols, Array, Dim4};

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

    fn construct_lin_part(&self, inputs: &[f32]) -> Array<f32> {
        let t0 = Instant::now();

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

        info!("consturct_lin_part took {}ms", t0.elapsed().as_millis());

        Array::new(&values, Dim4::new(&[nrows as u64, ncols as u64, 1, 1]))
    }

    fn construct_full_features(&self, inputs: &[f32]) -> Array<f32> {
        let t0 = Instant::now();

        let lin_part = self.construct_lin_part(inputs);

        let num_input_rows = lin_part.dims().get()[0];
        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;
        let nrows = num_input_rows - warmup as u64;
        let d_lin = self.params.num_time_delay_taps;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let ncols = d_lin + d_nonlin;

        let mut full_features = Array::new_empty(Dim4::new(&[nrows, ncols as u64, 1, 1]));

        // Copy over lin part
        set_cols(
            &mut full_features,
            &rows(&lin_part, warmup as i64, (num_input_rows - 1) as i64),
            0,
            d_lin as i64 - 1,
        );

        let mut cnt: usize = 0;
        for i in 0..d_lin {
            for j in i..d_lin {
                for span in j..d_lin {
                    let column = mul(
                        &mul(&col(&lin_part, i as i64), &col(&lin_part, j as i64), true),
                        &col(&lin_part, span as i64),
                        true,
                    );

                    set_col(
                        &mut full_features,
                        &rows(&column, warmup as i64, (num_input_rows - 1) as i64),
                        (d_lin + cnt) as i64,
                    );
                    cnt += 1;
                }
            }
        }

        info!("construct_full_features took {}ms", t0.elapsed().as_millis());

        full_features
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

    #[test]
    fn ngrc_arrayfire_full_features() {
        if let Err(_) = pretty_env_logger::try_init() {}

        set_device(0);
        info();

        let d_lin = 2;
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
        let full_features = rc.construct_full_features(&inputs);

        af_print!("full_features", full_features);
    }
}
