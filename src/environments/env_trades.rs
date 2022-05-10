use std::sync::Arc;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use super::{OptEnvironment, PlotGather};
use crate::{reservoir_computers::StateMatrix, LinReg, RCParams, ReservoirComputer, SingleDimIo};

pub struct EnvTrades {
    values: Arc<SingleDimIo>,
    train_len: usize,
}

impl EnvTrades {
    #[inline(always)]
    pub fn new(values: Arc<SingleDimIo>, train_len: usize) -> Self {
        Self {
            values,
            train_len,
        }
    }
}

impl<RC, const N: usize, R> OptEnvironment<RC, 1, 1, N, R> for EnvTrades
where
    RC: ReservoirComputer<1, 1, N, R>,
    R: LinReg,
{
    fn evaluate(&self, rc: &mut RC, mut plot: Option<&mut PlotGather>) -> f64 {
        rc.train(
            &self.values.columns(0, self.train_len - 1),
            &self.values.columns(1, self.train_len),
        );

        let init_val = *self.values.column(0).get(0).unwrap();
        let state: StateMatrix = Matrix::from_element_generic(
            Dim::from_usize(rc.params().reservoir_size()),
            Dim::from_usize(1),
            init_val,
        );
        rc.set_state(state);

        let mut rmse: f64 = 0.0;
        for j in 1..self.values.ncols() {
            if let Some(plot) = plot.as_mut() {
                plot.push_target(j as f64, *self.values.column(j).get(0).unwrap());
            }
            let predicted_out = rc.readout();
            let last_prediction = *predicted_out.get(0).unwrap();

            // To begin forecasting, replace target input with it's own prediction
            let m: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |i, _| {
                    *predicted_out.get(i).unwrap()
                });
            let target = *self.values.column(j).get(0).unwrap();
            rmse += (last_prediction - target).powi(2);

            let input = if j > self.train_len {
                if let Some(plot) = plot.as_mut() {
                    plot.push_test_pred(j as f64, last_prediction);
                }
                m.column(0)
            } else {
                if let Some(plot) = plot.as_mut() {
                    plot.push_train_pred(j as f64, last_prediction);
                }
                self.values.column(j - 1)
            };

            rc.update_state(&input, &predicted_out);
        }

        rmse
    }
}
