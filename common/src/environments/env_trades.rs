use std::sync::Arc;

use crate::{RCParams, ReservoirComputer};
use lin_reg::LinReg;
use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, VecStorage};

use super::PlotGather;

pub struct EnvTrades {
    values: Arc<DMatrix<f64>>,
    train_len: usize,
}

impl EnvTrades {
    #[inline(always)]
    pub fn new(values: Arc<DMatrix<f64>>, train_len: usize) -> Self {
        Self {
            values,
            train_len,
        }
    }
}

impl EnvTrades {
    pub fn evaluate<RC, R>(&self, rc: &mut RC, mut plot: Option<&mut PlotGather>) -> f64
    where
        RC: ReservoirComputer<R>,
        R: LinReg,
    {
        rc.train(&self.values.rows(0, self.train_len - 1), &self.values.rows(1, self.train_len));

        let vals: Vec<f64> = vec![rc.params().initial_state_value(); rc.params().reservoir_size()];
        let state: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
            Matrix::from_vec_generic(
                Dim::from_usize(1),
                Dim::from_usize(rc.params().reservoir_size()),
                vals,
            );
        rc.set_state(state);

        let mut rmse: f64 = 0.0;
        for i in 1..self.values.nrows() {
            if let Some(plot) = plot.as_mut() {
                plot.push_target(i as f64, *self.values.row(i).get(0).unwrap());
            }
            let predicted_out = rc.readout();
            let last_prediction = *predicted_out.get(0).unwrap();

            // To begin forecasting, replace target input with it's own prediction
            let m: DMatrix<f64> =
                Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |i, _| {
                    *predicted_out.get(i).unwrap()
                });
            let target = *self.values.row(i).get(0).unwrap();
            rmse += (last_prediction - target).powi(2);

            let input = if i > self.train_len {
                if let Some(plot) = plot.as_mut() {
                    plot.push_test_pred(i as f64, last_prediction);
                }
                m.row(0)
            } else {
                if let Some(plot) = plot.as_mut() {
                    plot.push_train_pred(i as f64, last_prediction);
                }
                self.values.row(i - 1)
            };

            rc.update_state(&input);
        }

        rmse
    }
}
