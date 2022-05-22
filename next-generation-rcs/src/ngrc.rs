use std::collections::VecDeque;

use common::{RCParams, ReservoirComputer};
use lin_reg::LinReg;
use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, VecStorage};

use super::{params::Params, FullFeatureConstructor};

type StateMatrix = Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>;

#[derive(Debug, Clone)]
pub struct NextGenerationRC<R, C> {
    params: Params,
    inputs: VecDeque<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    readout_matrix: DMatrix<f64>,
    state: StateMatrix,
    // size of the linear part of feature vector
    d_lin: usize,
    // Total size of the feature vector
    d_total: usize,
    // capacity of sliding window of inputs
    window_cap: usize,
    regressor: R,
    full_feature_constructor: C,
}

impl<R, C> NextGenerationRC<R, C>
where
    R: LinReg,
    C: FullFeatureConstructor,
{
    pub fn new(params: Params, regressor: R, full_feature_constructor: C) -> Self {
        let d_lin = params.num_time_delay_taps * params.input_dim;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let d_total = d_lin + d_nonlin;

        let readout_matrix = Matrix::from_element_generic(
            Dim::from_usize(params.output_dim),
            Dim::from_usize(d_total),
            0.0,
        );
        let state = Matrix::from_element_generic(Dim::from_usize(1), Dim::from_usize(d_total), 0.0);

        let window_cap = params.num_samples_to_skip * params.num_time_delay_taps + 1;

        Self {
            params,
            inputs: VecDeque::with_capacity(window_cap),
            readout_matrix,
            d_lin,
            d_total,
            state,
            window_cap,
            regressor,
            full_feature_constructor,
        }
    }

    /// Construct the linear part of feature vector
    ///
    /// # Arguments
    /// inputs: Number of rows are the observed datapoints and number of columns
    /// represent the features at each timestep
    fn construct_lin_part<'a>(
        &self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) -> DMatrix<f64> {
        assert_eq!(inputs.ncols(), 1, "more than 1 input dimension not implemented yet");

        let mut lin_part: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(inputs.nrows()),
            Dim::from_usize(self.d_lin),
            0.0,
        );

        for delay in 0..self.params.num_time_delay_taps {
            let mut column = vec![0.0; inputs.nrows()];
            for j in delay * self.params.num_samples_to_skip..inputs.nrows() {
                // TODO: support for more than 1 input dimension
                column[j] = *inputs
                    .column(0)
                    .get(j - delay * self.params.num_samples_to_skip)
                    .unwrap_or(&0.0);
            }
            let col_idx = inputs.ncols() * delay;
            let column: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                Matrix::from_vec_generic(Dim::from_usize(column.len()), Dim::from_usize(1), column);
            lin_part.set_column(col_idx, &column);
        }

        lin_part
    }
}

impl<R, C> ReservoirComputer<R> for NextGenerationRC<R, C>
where
    R: LinReg,
    C: FullFeatureConstructor,
{
    #[inline(always)]
    fn params(&self) -> &dyn RCParams {
        &self.params
    }

    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) {
        let lin_part = self.construct_lin_part(inputs);
        let full_features =
            <C as FullFeatureConstructor>::construct_full_features(&self.params, &lin_part);

        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;

        let mut design: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(full_features.nrows()),
            Dim::from_usize(full_features.ncols() + 1),
            0.0,
        );
        // add column of 1s here
        let col: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_element_generic(
                Dim::from_usize(full_features.nrows()),
                Dim::from_usize(1),
                1.0,
            );
        design.set_column(0, &col);
        for j in 1..full_features.ncols() + 1 {
            design.set_column(j, &full_features.column(j - 1));
        }

        self.readout_matrix = self.regressor.fit_readout(
            &design.rows(0, design.nrows()),
            &targets.rows(warmup + 1, targets.nrows() - warmup - 1),
        );
    }

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<1>, Dynamic, Const<1>, Dynamic>,
    ) {
        let input =
            <Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>::from_row_slice_generic(
                Dim::from_usize(1),
                Dim::from_usize(self.params.input_dim),
                input.iter().cloned().collect::<Vec<f64>>().as_slice(),
            );
        self.inputs.push_back(input);

        if self.inputs.len() > self.window_cap {
            let _ = self.inputs.pop_front();
        } else {
            debug!("not enough datapoints available yet to update state");
            return;
        }

        let mut inputs: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(self.window_cap),
            Dim::from_usize(self.params.input_dim),
            0.0,
        );
        for (i, col) in self.inputs.iter().enumerate() {
            inputs.set_row(i, col);
        }
        let lin_part = self.construct_lin_part(&inputs.rows(0, inputs.nrows()));
        let full_features =
            <C as FullFeatureConstructor>::construct_full_features(&self.params, &lin_part);

        // extract the state from the last full_feature column
        let mut state: Vec<f64> = vec![0.0; self.d_total + 1];
        state[0] = 1.0;
        for (i, f) in full_features.row(full_features.nrows() - 1).iter().enumerate() {
            state[i + 1] = *f;
        }
        self.state =
            <Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>::from_vec_generic(
                Dim::from_usize(1),
                Dim::from_usize(self.d_total + 1), // +1 as the first column is always a 1
                state,
            );
    }

    #[inline(always)]
    fn readout(&self) -> Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> {
        if self.inputs.len() < self.window_cap {
            return Matrix::from_element_generic(
                Dim::from_usize(self.params.output_dim),
                Dim::from_usize(1),
                0.0,
            );
        }
        debug!(
            "readout: dims of state: ({}, {}), readout: ({}, {})",
            self.state.nrows(),
            self.state.ncols(),
            self.readout_matrix.nrows(),
            self.readout_matrix.ncols()
        );

        let mut pred = &self.state * &self.readout_matrix;
        self.params.output_activation.activate(pred.as_mut_slice());

        pred
    }

    #[inline(always)]
    fn set_state(
        &mut self,
        state: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    ) {
        // Prepending the 1 required for proper readout
        // NOTE: this assumes the input state does not have the 1 column yet
        // It's done to maintain compatibility with classic-rcs
        let state: StateMatrix = Matrix::from_fn_generic(
            Dim::from_usize(1),
            Dim::from_usize(state.ncols() + 1),
            |_, j| {
                if j == 0 {
                    1.0
                } else {
                    *state.get(j - 1).unwrap()
                }
            },
        );
        self.state = state;
    }

    #[inline(always)]
    fn readout_matrix(&self) -> &DMatrix<f64> {
        &self.readout_matrix
    }
}

#[cfg(test)]
mod tests {
    use common::Activation;
    use lin_reg::TikhonovRegularization;
    use round::round;

    use super::*;
    use crate::NGRCConstructor;

    const NUM_VALS: usize = 9;

    fn get_inputs() -> DMatrix<f64> {
        Matrix::from_vec_generic(
            Dim::from_usize(NUM_VALS),
            Dim::from_usize(1),
            vec![0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0],
        )
    }

    #[test]
    fn ngrc_lin_part_1d_skip1() {
        if let Err(_) = pretty_env_logger::try_init() {}

        let inputs = get_inputs();

        const K: usize = 3;
        let params = Params {
            input_dim: 1,
            output_dim: 1,
            num_time_delay_taps: K,
            num_samples_to_skip: 1,
            output_activation: Activation::Tanh,
        };
        let regressor = TikhonovRegularization {
            regularization_coeff: 0.001,
        };
        let fc = NGRCConstructor::default();
        let ngrc =
            NextGenerationRC::<TikhonovRegularization, NGRCConstructor>::new(params, regressor, fc);

        let lin_part = ngrc.construct_lin_part(&inputs.rows(0, NUM_VALS));
        info!("inputs: {}", inputs);
        info!("lin_part: {}", lin_part);

        let goal_part: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(NUM_VALS),
            Dim::from_usize(K),
            vec![
                0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0, 0.0, 0.0, 0.55, 1.0, 0.45, 0.0,
                -0.55, -1.0, -0.45, 0.0, 0.0, 0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0,
            ],
        );
        info!("goal_part: {}", goal_part);

        assert_eq!(lin_part, goal_part)
    }

    #[test]
    fn ngrc_lin_part_1d_skip2() {
        if let Err(_) = pretty_env_logger::try_init() {}

        let inputs = get_inputs();

        const K: usize = 2;
        let params = Params {
            input_dim: 1,
            output_dim: 1,
            num_time_delay_taps: K,
            num_samples_to_skip: 2,
            output_activation: Activation::Tanh,
        };
        let regressor = TikhonovRegularization {
            regularization_coeff: 0.001,
        };
        let fc = NGRCConstructor::default();
        let ngrc =
            NextGenerationRC::<TikhonovRegularization, NGRCConstructor>::new(params, regressor, fc);

        let lin_part = ngrc.construct_lin_part(&inputs.rows(0, NUM_VALS));
        info!("inputs: {}", inputs);
        info!("lin_part: {}", lin_part);

        let goal_part: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(NUM_VALS),
            Dim::from_usize(K),
            vec![
                0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0, 0.0, 0.0, 0.0, 0.55, 1.0, 0.45,
                0.0, -0.55, -1.0,
            ],
        );
        info!("goal_part: {}", goal_part);

        assert_eq!(lin_part, goal_part)
    }

    #[test]
    fn ngrc_lin_part_2d() {
        todo!()
    }

    #[test]
    fn ngrc_nonlin_part_1d() {
        if let Err(_) = pretty_env_logger::try_init() {}

        let inputs = get_inputs();

        const I: usize = 1;
        const K: usize = 2;
        let params = Params {
            input_dim: 1,
            output_dim: 1,
            num_time_delay_taps: K,
            num_samples_to_skip: 1,
            output_activation: Activation::Tanh,
        };
        let regressor = TikhonovRegularization {
            regularization_coeff: 0.001,
        };
        let fc = NGRCConstructor::default();
        let ngrc = NextGenerationRC::<TikhonovRegularization, NGRCConstructor>::new(
            params.clone(),
            regressor,
            fc,
        );

        let lin_part = ngrc.construct_lin_part(&inputs.rows(0, inputs.nrows()));
        let mut full_features = NGRCConstructor::construct_full_features(&params, &lin_part);
        info!("inputs: {}", inputs);

        let d_lin = K * I;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let d_total = d_lin + d_nonlin;
        let warmup = I * K;

        let goal_features: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(inputs.nrows() - warmup),
            Dim::from_usize(d_total),
            vec![
                1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45,
                1.0, 0.091125, 0.0, -0.166375, -1.0, -0.091125, 0.0, 0.55, 0.2025, 0.0, 0.0, -0.55,
                -0.2025, 0.0, 0.3025, 0.45, 0.0, 0.0, -0.3025, -0.45, 0.0, 0.166375, 1.0, 0.091125,
                0.0, -0.166375, -1.0, -0.091125,
            ],
        );
        info!("goal_features: {}", goal_features);

        // round all values to 6 decimal places
        full_features.iter_mut().for_each(|v| *v = round(*v, 6));
        info!("full_features: {}", full_features);

        assert_eq!(full_features, goal_features);
    }
}
