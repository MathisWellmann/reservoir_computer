use std::{collections::VecDeque, time::Instant};

use arrayfire::{
    col, diag_create, inverse, matmul, mul, row, rows, set_col, set_cols, sum_all, transpose, view,
    Array, Dim4, MatProp,
};

use crate::Params;

pub struct NGRCArrayfire {
    params: Params,
    readout_matrix: Array<f32>,
    state_matrix: Array<f32>,
    inputs: VecDeque<f32>,
}

impl NGRCArrayfire {
    pub fn new(params: Params) -> Self {
        let d_lin = params.num_time_delay_taps;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let d_total = (d_lin + d_nonlin) as u64;

        let cap = params.num_samples_to_skip * params.num_time_delay_taps;

        Self {
            params,
            readout_matrix: Array::new_empty(Dim4::new(&[1, d_total, 1, 1])),
            state_matrix: Array::new_empty(Dim4::new(&[1, d_total + 1, 1, 1])),
            inputs: VecDeque::with_capacity(cap),
        }
    }

    #[inline(always)]
    pub fn params(&self) -> &Params {
        &self.params
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
        debug_assert!(inputs.len() > self.params.warmup_steps());

        let t0 = Instant::now();

        let lin_part = self.construct_lin_part(inputs);

        let num_input_rows = lin_part.dims().get()[0];
        let warmup = self.params.warmup_steps();
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
        info!("inputs.len(): {}, targets.len(): {}", inputs.len(), targets.len());
        let t = Instant::now();

        let design = self.construct_design_matrix(inputs);

        let design_ncols = design.dims().get()[1];
        let reg_m = diag_create(
            &Array::new(
                &vec![self.params.regularization_coeff; design_ncols as usize],
                Dim4::new(&[design_ncols, 1, 1, 1]),
            ),
            0,
        );

        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;
        let ts: Vec<f32> = targets.iter().skip(warmup).cloned().collect();
        let targets = Array::new(&ts, Dim4::new(&[ts.len() as u64, 1, 1, 1]));

        let t0 = transpose(&design, false);
        debug!("design dims: {:?}, t0 dims: {:?}", design.dims(), t0.dims());
        let p0 = matmul(&t0, &design, MatProp::NONE, MatProp::NONE);
        let p1 = inverse(&(p0 + reg_m), MatProp::NONE);
        debug!("targets dims: {:?}", targets.dims());
        let p2 = matmul(&t0, &targets, MatProp::NONE, MatProp::NONE);

        self.readout_matrix = matmul(&p1, &p2, MatProp::NONE, MatProp::NONE);

        info!("train took {} ms", t.elapsed().as_millis());
    }

    pub fn update_state(&mut self, input: f32) {
        self.inputs.push_back(input);

        if self.inputs.len() > self.params.warmup_steps() + 1 {
            let _ = self.inputs.pop_front();
        } else {
            debug!("not enough datapoints to create design matrix");
            return;
        }

        // TODO: PERF: there is room for improvement here
        let design_matrix =
            self.construct_design_matrix(&self.inputs.iter().cloned().collect::<Vec<f32>>());
        self.state_matrix = row(&design_matrix, design_matrix.dims().get()[0] as i64 - 1);
    }

    /// Perform a readout of the current state
    pub fn readout(&self) -> f32 {
        let pred = matmul(&self.state_matrix, &self.readout_matrix, MatProp::NONE, MatProp::NONE);

        // There is only one element in pred,
        // which i want to extract and this seems to be the only way to do it
        let val = sum_all(&pred);
        info!("readout val: {:?}", val);
        self.params.output_activation.activate(val.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Activation;
    use arrayfire::*;

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

        let params = Params {
            num_time_delay_taps: d_lin,
            num_samples_to_skip: 1,
            output_activation: Activation::Tanh,
            regularization_coeff: 0.1,
        };
        let rc = NGRCArrayfire::new(params);

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

        let params = Params {
            num_time_delay_taps: d_lin,
            num_samples_to_skip: 1,
            output_activation: Activation::Tanh,
            regularization_coeff: 0.1,
        };
        let rc = NGRCArrayfire::new(params);

        let inputs = get_inputs();
        let full_features = rc.construct_design_matrix(&inputs);

        af_print!("full_features", full_features);
    }
}
