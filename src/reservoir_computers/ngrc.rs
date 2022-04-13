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
        0
    }
}

pub struct ParamMapper {}

impl OptParamMapper<PARAM_DIM> for ParamMapper {
    type Params = Params;

    fn map(&self, params: &[f64; PARAM_DIM]) -> Self::Params {
        todo!()
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

    fn train(
        &mut self,
        inputs: &Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>>,
        targets: &Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    ) {
        let nvals = inputs.ncols() - self.params.num_time_delay_taps;
        let mut lin_part: DMatrix<f64> =
            Matrix::from_element_generic(Dim::from_usize(self.d_lin), Dim::from_usize(nvals), 0.0);
        for j in (self.params.num_time_delay_taps * self.params.num_samples_to_skip)..nvals {
            let mut col = vec![];
            for delay in 0..self.params.num_time_delay_taps {
                col.append(
                    &mut inputs
                        .column(j - (delay * self.params.num_samples_to_skip))
                        .as_slice()
                        .to_vec(),
                );
            }
            let col: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                Matrix::from_vec_generic(Dim::from_usize(self.d_lin), Dim::from_usize(1), col);
            lin_part.set_column(j, &col);
        }

        let col_start = self.params.num_time_delay_taps;
        let full_features = lin_part.clone();
        let mut full_features = full_features.resize_generic::<Dynamic, Dynamic>(
            Dim::from_usize(self.d_total),
            Dim::from_usize(nvals - col_start),
            0.0,
        );

        // Fill in the nonlinear part
        let mut cnt: usize = 0;
        for i in 0..self.d_lin {
            for j in i..self.d_lin {
                for span in j..self.d_lin {
                    let row: Vec<f64> = lin_part
                        .row(i)
                        .iter()
                        .skip(col_start)
                        .zip(lin_part.row(j).iter().skip(col_start))
                        .zip(lin_part.row(span).iter().skip(col_start))
                        .map(|((v_i, v_j), v_s)| v_i * v_j * v_s)
                        .collect();
                    let row: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                        Matrix::from_vec_generic(
                            Dim::from_usize(1),
                            Dim::from_usize(nvals - col_start),
                            row,
                        );
                    full_features.set_row(self.d_lin + cnt, &row);
                    cnt += 1;
                }
            }
        }

        // Tikhonov regularization aka ridge regression
        let reg_m: DMatrix<f64> = Matrix::from_diagonal_element_generic(
            Dim::from_usize(self.d_total),
            Dim::from_usize(self.d_total),
            self.params.regularization_coeff,
        );
        let p_0 = targets.columns(col_start, nvals - col_start) * full_features.transpose();
        let p_1 = &full_features * full_features.transpose();
        let r: DMatrix<f64> = p_1 + reg_m;
        self.readout_matrix = p_0 * r.try_inverse().unwrap();
    }

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<I>, Const<1>, Const<1>, Const<I>>,
        _prev_pred: &Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>>,
    ) {
        if self.inputs.len() > self.window_cap {
            let _ = self.inputs.pop_front();
        }
        let input = <Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>>::from_column_slice_generic(
                Dim::from_usize(I),
                Dim::from_usize(1),
                input.as_slice()
            );
        self.inputs.push_back(input);

        if self.inputs.len() < self.window_cap {
            return;
        }

        let nvals = self.inputs.len();
        let mut lin_part: DMatrix<f64> =
            Matrix::from_element_generic(Dim::from_usize(self.d_lin), Dim::from_usize(nvals), 0.0);
        for j in (self.params.num_time_delay_taps * self.params.num_samples_to_skip)..nvals {
            let mut col = vec![];
            for delay in 0..self.params.num_time_delay_taps {
                col.append(
                    &mut self.inputs[j - (delay * self.params.num_samples_to_skip)]
                        .as_slice()
                        .to_vec(),
                );
            }
            let col: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                Matrix::from_vec_generic(Dim::from_usize(self.d_lin), Dim::from_usize(1), col);
            lin_part.set_column(j, &col);
        }

        let col_start = self.params.num_time_delay_taps;
        let full_features = lin_part.clone();
        let mut full_features = full_features.resize_generic::<Dynamic, Dynamic>(
            Dim::from_usize(self.d_total),
            Dim::from_usize(nvals - col_start),
            0.0,
        );

        // Fill in the nonlinear part
        let mut cnt: usize = 0;
        for i in 0..self.d_lin {
            for j in i..self.d_lin {
                for span in j..self.d_lin {
                    let row: Vec<f64> = lin_part
                        .row(i)
                        .iter()
                        .skip(col_start)
                        .zip(lin_part.row(j).iter().skip(col_start))
                        .zip(lin_part.row(span).iter().skip(col_start))
                        .map(|((v_i, v_j), v_s)| v_i * v_j * v_s)
                        .collect();
                    let row: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                        Matrix::from_vec_generic(
                            Dim::from_usize(1),
                            Dim::from_usize(nvals - col_start),
                            row,
                        );
                    full_features.set_row(self.d_lin + cnt, &row);
                    cnt += 1;
                }
            }
        }

        // extract the state from the last full_feature column
        self.state = <Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>>::from_column_slice_generic(
            Dim::from_usize(self.d_total),
            Dim::from_usize(1),
            full_features.column(nvals - col_start - 1).as_slice(),
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
