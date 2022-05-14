use std::collections::VecDeque;

use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, VecStorage};

use super::{OptParamMapper, StateMatrix};
use crate::{activation::Activation, lin_reg::LinReg, RCParams, ReservoirComputer};

const PARAM_DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct Params {
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_time_delay_taps: usize,
    pub num_samples_to_skip: usize,
    pub output_activation: Activation,
}

#[derive(Debug, Clone)]
pub struct NextGenerationRC<R> {
    params: Params,
    inputs: VecDeque<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    readout_matrix: DMatrix<f64>,
    state: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    // size of the linear part of feature vector
    d_lin: usize,
    // Total size of the feature vector
    d_total: usize,
    // capacity of sliding window of inputs
    window_cap: usize,
    regressor: R,
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

impl<R> NextGenerationRC<R>
where
    R: LinReg,
{
    /// Construct the linear part of feature vector
    ///
    /// # Arguments
    /// inputs: Number of rows are the observed datapoints and number of columns represent the features at each timestep
    ///
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

    /// Construct the nonlinear part of feature matrix from linear part
    ///
    /// # Arguments
    /// inputs: Number of rows are the observed datapoints and number of columns represent the features at each timestep
    ///
    fn construct_full_features<'a>(
        &self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) -> DMatrix<f64> {
        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;

        let lin_part = self.construct_lin_part(inputs);

        // manually copy over elements while skipping the warmup columns
        let mut full_features: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(lin_part.nrows() - warmup),
            Dim::from_usize(self.d_total),
            0.0,
        );
        for i in warmup..lin_part.nrows() {
            full_features
                .set_row(i - warmup, &lin_part.row(i).resize_horizontally(self.d_total, 0.0));
        }

        let mut cnt: usize = 0;
        for i in 0..self.d_lin {
            for j in i..self.d_lin {
                for span in j..self.d_lin {
                    let column: Vec<f64> = lin_part
                        .column(i)
                        .iter()
                        .skip(warmup)
                        .zip(lin_part.column(j).iter().skip(warmup))
                        .zip(lin_part.column(span).iter().skip(warmup))
                        .map(|((v_i, v_j), v_s)| v_i * v_j * v_s)
                        .collect();
                    let column: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                        Matrix::from_vec_generic(
                            Dim::from_usize(lin_part.nrows() - warmup),
                            Dim::from_usize(1),
                            column,
                        );
                    full_features.set_column(self.d_lin + cnt, &column);
                    cnt += 1;
                }
            }
        }

        full_features
    }
}

impl<R> ReservoirComputer<PARAM_DIM, R> for NextGenerationRC<R>
where
    R: LinReg,
{
    type ParamMapper = ParamMapper;

    fn new(params: Params, regressor: R) -> Self {
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
        }
    }

    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) {
        let full_features = self.construct_full_features(inputs);

        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;

        // TODO: add column of 1s here
        let mut design: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(full_features.nrows()),
            Dim::from_usize(full_features.ncols() + 1),
            0.0,
        );
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
        _prev_pred: &Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
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
        let full_features = self.construct_full_features(&inputs.columns(0, inputs.ncols()));

        // extract the state from the last full_feature column
        let mut state: Vec<f64> = vec![0.0; self.d_total + 1];
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

        let mut pred = &self.state * &self.readout_matrix;
        self.params.output_activation.activate(pred.as_mut_slice());

        pred
    }

    #[inline(always)]
    fn set_state(
        &mut self,
        state: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    ) {
        self.state = state;
    }

    #[inline(always)]
    fn params(&self) -> &Params {
        &self.params
    }

    #[inline(always)]
    fn readout_matrix(&self) -> &DMatrix<f64> {
        &self.readout_matrix
    }
}

#[cfg(test)]
mod tests {
    use round::round;

    use super::*;
    use crate::lin_reg::TikhonovRegularization;

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
        let ngrc = NextGenerationRC::<TikhonovRegularization>::new(params, regressor);

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
        let ngrc = NextGenerationRC::<TikhonovRegularization>::new(params, regressor);

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
        let ngrc = NextGenerationRC::<TikhonovRegularization>::new(params, regressor);

        let mut full_features = ngrc.construct_full_features(&inputs.rows(0, inputs.nrows()));
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
