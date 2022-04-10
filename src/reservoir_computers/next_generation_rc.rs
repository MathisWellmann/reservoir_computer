use std::collections::VecDeque;

use nalgebra::{ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, VecStorage};
use nanorand::{Rng, WyRand};

use super::{OptParamMapper, StateMatrix};
use crate::{activation::Activation, RCParams, ReservoirComputer};

const PARAM_DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct Params {
    pub num_time_delay_taps: usize,
    pub num_samples_to_skip: usize,
    pub regularization_coeff: f64,
    pub output_activation: Activation,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct NextGenerationRC<const I: usize, const O: usize> {
    params: Params,
    readout_matrix: Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    state: StateMatrix,
    // size of the linear part of feature vector
    d_lin: usize,
    // Total size of the feature vector
    d_total: usize,
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

        let mut rng = match params.seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };

        let readout_matrix =
            Matrix::from_fn_generic(Dim::from_usize(O), Dim::from_usize(d_total), |_, _| {
                rng.generate::<f64>() * 2.0 - 1.0
            });
        let state = Matrix::from_element_generic(Dim::from_usize(d_total), Dim::from_usize(1), 0.0);

        Self {
            params,
            readout_matrix,
            d_lin,
            d_total,
            state,
        }
    }

    fn train(
        &mut self,
        inputs: &Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>>,
        targets: &Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    ) {
        let nvals = inputs.ncols() - self.params.num_time_delay_taps;
        let mut lin_part: StateMatrix =
            Matrix::from_element_generic(Dim::from_usize(self.d_lin), Dim::from_usize(nvals), 0.0);
        for j in self.params.num_time_delay_taps..nvals {
            let mut col = vec![];
            for delay in 0..self.params.num_time_delay_taps {
                col.append(&mut inputs.column(j - delay).as_slice().to_vec());
            }
            let col: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                Matrix::from_vec_generic(Dim::from_usize(self.d_lin), Dim::from_usize(1), col);
            lin_part.set_column(j, &col);
        }

        let full_features = lin_part.clone();
        let mut full_features = full_features.resize_generic::<Dynamic, Dynamic>(
            Dim::from_usize(self.d_total),
            Dim::from_usize(nvals),
            0.0,
        );

        // Fill in the nonlinear part
        let mut cnt: usize = 0;
        for i in 0..self.d_lin {
            for j in i..self.d_lin {
                for span in j..self.d_lin {
                    let col_start = self.params.num_time_delay_taps;
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

        let reg_m: DMatrix<f64> = Matrix::from_diagonal_element_generic(
            Dim::from_usize(self.d_total),
            Dim::from_usize(self.d_total),
            self.params.regularization_coeff,
        );
        let t_x_ft = targets * full_features.transpose();
        let f_x_ft = &full_features * full_features.transpose();
        let r: DMatrix<f64> = f_x_ft * reg_m;
        self.readout_matrix = t_x_ft * r.try_inverse().unwrap();
    }

    fn update_state<'a>(
        &mut self,
        _input: &'a MatrixSlice<'a, f64, Const<I>, Const<1>, Const<1>, Const<I>>,
        _prev_pred: &Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>>,
    ) {
        panic!("there is not need to update state with the next generation reservoir computer")
    }

    #[inline(always)]
    fn readout(&self) -> Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>> {
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
