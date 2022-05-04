use std::collections::VecDeque;

use nalgebra::{ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, VecStorage};

use super::{OptParamMapper, StateMatrix};
use crate::{activation::Activation, RCParams, ReservoirComputer};

const PARAM_DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct Params {
    pub num_time_delay_taps: usize,
    pub num_samples_to_skip: usize,
    pub regularization_coeff: f64,
    pub output_activation: Activation,
}

#[derive(Debug, Clone)]
pub struct NextGenerationRC<const I: usize, const O: usize> {
    params: Params,
    inputs: VecDeque<StateMatrix>,
    readout_matrix: Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    state: StateMatrix,
    // size of the linear part of feature vector
    d_lin: usize,
    // Total size of the feature vector
    d_total: usize,
    // capacity of sliding window of inputs
    window_cap: usize,
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

impl<const I: usize, const O: usize> NextGenerationRC<I, O> {
    fn construct_lin_part<'a>(
        &self,
        inputs: &'a MatrixSlice<'a, f64, Const<I>, Dynamic, Const<1>, Const<I>>,
    ) -> DMatrix<f64> {
        assert_eq!(I, 1, "more than 1 input dimension not implemented yet");

        let mut lin_part: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(self.d_lin),
            Dim::from_usize(inputs.ncols()),
            0.0,
        );

        for delay in 0..self.params.num_time_delay_taps {
            let mut row = vec![0.0; inputs.ncols()];
            for j in delay..inputs.ncols() {
                // TODO: support for more than 1 input dimension
                row[j] = *inputs.row(0).get(j - delay).unwrap();
            }
            let row_idx = I * delay;
            let row: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                Matrix::from_vec_generic(Dim::from_usize(1), Dim::from_usize(row.len()), row);
            lin_part.set_row(row_idx, &row);
        }

        lin_part
    }

    /// Construct the nonlinear part of feature matrix from linear part
    fn construct_full_features<'a>(
        &self,
        inputs: &'a MatrixSlice<'a, f64, Const<I>, Dynamic, Const<1>, Const<I>>,
    ) -> DMatrix<f64> {
        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;

        let lin_part = self.construct_lin_part(inputs);

        // manually copy over elements while skipping the warmup columns
        let mut full_features: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(self.d_total),
            Dim::from_usize(lin_part.ncols() - warmup),
            0.0,
        );
        for j in warmup..lin_part.ncols() {
            full_features
                .set_column(j - warmup, &lin_part.column(j).resize_vertically(self.d_total, 0.0));
        }

        let mut cnt: usize = 0;
        for i in 0..self.d_lin {
            for j in i..self.d_lin {
                for span in j..self.d_lin {
                    let row: Vec<f64> = lin_part
                        .row(i)
                        .iter()
                        .skip(warmup)
                        .zip(lin_part.row(j).iter().skip(warmup))
                        .zip(lin_part.row(span).iter().skip(warmup))
                        .map(|((v_i, v_j), v_s)| v_i * v_j * v_s)
                        .collect();
                    let row: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                        Matrix::from_vec_generic(
                            Dim::from_usize(1),
                            Dim::from_usize(lin_part.ncols() - warmup),
                            row,
                        );
                    full_features.set_row(self.d_lin + cnt, &row);
                    cnt += 1;
                }
            }
        }

        full_features
    }
}

impl<const I: usize, const O: usize> ReservoirComputer<I, O, PARAM_DIM> for NextGenerationRC<I, O> {
    type ParamMapper = ParamMapper;

    fn new(params: Params) -> Self {
        let d_lin = params.num_time_delay_taps * I;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let d_total = d_lin + d_nonlin;

        let readout_matrix =
            Matrix::from_element_generic(Dim::from_usize(O), Dim::from_usize(d_total), 0.0);
        let state = Matrix::from_element_generic(Dim::from_usize(d_total), Dim::from_usize(1), 0.0);

        let window_cap = params.num_samples_to_skip * params.num_time_delay_taps + 1;
        Self {
            params,
            inputs: VecDeque::with_capacity(window_cap),
            readout_matrix,
            d_lin,
            d_total,
            state,
            window_cap,
        }
    }

    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Const<I>, Dynamic, Const<1>, Const<I>>,
        targets: &'a MatrixSlice<'a, f64, Const<O>, Dynamic, Const<1>, Const<I>>,
    ) {
        let full_features = self.construct_full_features(inputs);

        let warmup = self.params.num_time_delay_taps * self.params.num_samples_to_skip;

        // Tikhonov regularization aka ridge regression
        let reg_m: DMatrix<f64> = Matrix::from_diagonal_element_generic(
            Dim::from_usize(self.d_total),
            Dim::from_usize(self.d_total),
            self.params.regularization_coeff,
        );
        debug!("warmup: {}, targets.ncols(): {}", warmup, targets.ncols());
        debug!("features.ncols(): {}", full_features.ncols());
        let p_0 =
            targets.columns(warmup + 1, targets.ncols() - warmup - 1) * full_features.transpose();
        let p_1 = &full_features * full_features.transpose();
        let r: DMatrix<f64> = p_1 + reg_m;
        self.readout_matrix = p_0 * r.try_inverse().unwrap();
    }

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<I>, Const<1>, Const<1>, Const<I>>,
        _prev_pred: &Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>>,
    ) {
        let input = <Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>>::from_column_slice_generic(
                Dim::from_usize(I),
                Dim::from_usize(1),
                input.as_slice()
            );
        self.inputs.push_back(input);

        if self.inputs.len() > self.window_cap {
            let _ = self.inputs.pop_front();
        } else {
            debug!("not enough datapoints available yet to update state");
            return;
        }

        let mut inputs: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
            Matrix::from_element_generic(Dim::from_usize(I), Dim::from_usize(self.window_cap), 0.0);
        for (j, col) in self.inputs.iter().enumerate() {
            inputs.set_column(j, col);
        }
        let full_features = self.construct_full_features(&inputs.columns(0, inputs.ncols()));

        // extract the state from the last full_feature column
        self.state = <Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>>::from_column_slice_generic(
            Dim::from_usize(self.d_total),
            Dim::from_usize(1),
            full_features.column(full_features.ncols() - 1).as_slice(),
        );
    }

    #[inline(always)]
    fn readout(&self) -> Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>> {
        if self.inputs.len() < self.window_cap {
            return Matrix::from_element_generic(Dim::from_usize(O), Dim::from_usize(1), 0.0);
        }

        let mut pred = &self.readout_matrix * &self.state;
        self.params.output_activation.activate(pred.as_mut_slice());

        pred
    }

    #[inline(always)]
    fn set_state(&mut self, state: StateMatrix) {
        self.state = state;
    }

    #[inline(always)]
    fn params(&self) -> &Params {
        &self.params
    }

    #[inline(always)]
    fn readout_matrix(
        &self,
    ) -> &Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>> {
        &self.readout_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use round::round;

    const NUM_VALS: usize = 9;

    fn get_inputs() -> Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> {
        Matrix::from_vec_generic(
            Dim::from_usize(1),
            Dim::from_usize(NUM_VALS),
            vec![0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0],
        )
    }

    #[test]
    fn ngrc_lin_part_1d_skip1() {
        if let Err(_) = pretty_env_logger::try_init() {}

        let inputs = get_inputs();

        const K: usize = 3;
        let params = Params {
            num_time_delay_taps: K,
            num_samples_to_skip: 1,
            regularization_coeff: 0.0001,
            output_activation: Activation::Tanh,
        };
        let ngrc = NextGenerationRC::<1, 1>::new(params);

        let lin_part = ngrc.construct_lin_part(&inputs.columns(0, NUM_VALS));
        info!("inputs: {}", inputs);
        info!("lin_part: {}", lin_part);

        let goal_part: Matrix<f64, Const<K>, Dynamic, VecStorage<f64, Const<K>, Dynamic>> =
            Matrix::from_vec_generic(
                Dim::from_usize(K),
                Dim::from_usize(NUM_VALS),
                vec![
                    0.0, 0.0, 0.0, 0.55, 0.0, 0.0, 1.0, 0.55, 0.0, 0.45, 1.0, 0.55, 0.0, 0.45, 1.0,
                    -0.55, 0.0, 0.45, -1.0, -0.55, 0.0, -0.45, -1.0, -0.55, 0.0, -0.45, -1.0,
                ],
            );
        info!("goal_part: {}", goal_part);

        assert_eq!(lin_part, goal_part)
    }

    #[test]
    fn ngrc_lin_part_1d_skip2() {
        todo!()
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
            num_time_delay_taps: K,
            num_samples_to_skip: 1,
            regularization_coeff: 0.0001,
            output_activation: Activation::Tanh,
        };
        let ngrc = NextGenerationRC::<1, 1>::new(params);

        let mut full_features = ngrc.construct_full_features(&inputs.columns(0, inputs.ncols()));
        info!("inputs: {}", inputs);

        let d_lin = K * I;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let d_total = d_lin + d_nonlin;
        let warmup = I * K;

        let goal_features: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(d_total),
            Dim::from_usize(inputs.ncols() - warmup),
            vec![
                1.0, 0.55, 1.0, 0.55, 0.3025, 0.166375, 0.45, 1.0, 0.091125, 0.2025, 0.45, 1.0,
                0.0, 0.45, 0.0, 0.0, 0.0, 0.091125, -0.55, 0.0, -0.166375, 0.0, 0.0, 0.0, -1.0,
                -0.55, -1.0, -0.55, -0.3025, -0.166375, -0.45, -1.0, -0.091125, -0.2025, -0.45,
                -1.0, 0.0, -0.45, 0.0, 0.0, 0.0, -0.091125,
            ],
        );
        info!("goal_features: {}", goal_features);

        // round all values to 6 decimal places
        full_features.iter_mut().for_each(|v| *v = round(*v, 6));
        info!("full_features: {}", full_features);

        assert_eq!(full_features, goal_features);
    }
}
