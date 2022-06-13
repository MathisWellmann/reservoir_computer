use std::time::Instant;

use arrayfire::{
    col, diag_create, inverse, mul, rows, set_col, set_cols, transpose, Array, Dim4, MatProp,
};

use crate::params::Params;

pub struct NextGenerationRC {
    params: Params,
    readout_matrix: Array<f32>,
}

impl NextGenerationRC {
    pub fn new(params: Params) -> Self {
        let d_lin = params.num_time_delay_taps;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let d_total = d_lin + d_nonlin;

        Self {
            params,
            readout_matrix: Array::new_empty(Dim4::new(&[1, d_total as u64, 1, 1])),
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

    fn construct_design_matrix(&self, inputs: &[f32]) -> Array<f32> {
        let t0 = Instant::now();

        let lin_part = self.construct_lin_part(inputs);

        let num_input_rows = lin_part.dims().get()[0];
        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;
        let nrows = num_input_rows - warmup as u64;
        let d_lin = self.params.num_time_delay_taps;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let ncols = d_lin + d_nonlin + 1; // Add column of 1s here already

        let mut design = Array::new_empty(Dim4::new(&[nrows, ncols as u64, 1, 1]));

        // Insert column of 1s
        let column: Vec<f32> = vec![1.0; nrows as usize];
        set_col(&mut design, &Array::new(&column, Dim4::new(&[nrows, 1, 1, 1])), 0);

        // Copy over lin part
        set_cols(
            &mut design,
            &rows(&lin_part, warmup as i64, (num_input_rows - 1) as i64),
            1,
            d_lin as i64,
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
                        &mut design,
                        &rows(&column, warmup as i64, (num_input_rows - 1) as i64),
                        (d_lin + cnt) as i64,
                    );
                    cnt += 1;
                }
            }
        }

        info!("construct_full_features took {}ms", t0.elapsed().as_millis());

        design
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32]) {
        let design = self.construct_design_matrix(inputs);

        // Fit linear regression
        let design_ncols = design.dims().get()[1];
        let reg_m = diag_create(
            &Array::new(
                &vec![self.params.regularization_coeff; design_ncols as usize],
                Dim4::new(&[design_ncols, 1, 1, 1]),
            ),
            0,
        );

        let targets = Array::new(&targets, Dim4::new(&[targets.len() as u64, 1, 1, 1]));

        let t0 = transpose(&design, false);
        let p0 = &t0 * &design;
        let p1 = inverse(&(p0 + reg_m), MatProp::NONE);
        let p2 = t0 * targets;

        self.readout_matrix = p1 * p2;
    }
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
            regularization_coeff: 0.1,
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
            regularization_coeff: 0.1,
        };
        let rc = NextGenerationRC::new(params);

        let inputs = get_inputs();
        let full_features = rc.construct_design_matrix(&inputs);

        af_print!("full_features", full_features);
    }
}
