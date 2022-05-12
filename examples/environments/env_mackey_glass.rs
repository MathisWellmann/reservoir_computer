use std::sync::Arc;

use nalgebra::{DMatrix, Dim, Matrix};

use super::PlotGather;
use reservoir_computer::{LinReg, RCParams, ReservoirComputer};

pub struct EnvMackeyGlass {
    values: Arc<DMatrix<f64>>,
    train_len: usize,
}

impl EnvMackeyGlass {
    #[inline]
    pub fn new(values: Arc<DMatrix<f64>>, train_len: usize) -> Self {
        assert!(values.ncols() > train_len, "make sure train_len < number of datapoints");

        Self {
            values,
            train_len,
        }
    }
}

impl EnvMackeyGlass {
    pub fn evaluate<RC, const N: usize, R>(
        &self,
        rc: &mut RC,
        mut plot: Option<&mut PlotGather>,
    ) -> f64
    where
        RC: ReservoirComputer<N, R>,
        R: LinReg,
    {
        rc.train(
            &self.values.columns(0, self.train_len - 1),
            &self.values.columns(1, self.train_len),
        );

        let state = Matrix::from_element_generic(
            Dim::from_usize(rc.params().reservoir_size()),
            Dim::from_usize(1),
            rc.params().initial_state_value(),
        );
        rc.set_state(state);

        let mut rmse = 0.0;
        for j in 1..self.values.ncols() {
            if let Some(plot) = plot.as_mut() {
                plot.push_target(j as f64, *self.values.column(j).get(0).unwrap());
            }

            let predicted_out = rc.readout();
            let mut last_prediction = *predicted_out.get(0).unwrap();
            if !last_prediction.is_finite() {
                last_prediction = 0.0;
            }

            // To begin forecasting, replace target input with it's own prediction
            let m: DMatrix<f64> =
                Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |i, _| {
                    *predicted_out.get(i).unwrap()
                });
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
            rmse += (*self.values.column(j).get(0).unwrap() - last_prediction).powi(2);

            rc.update_state(&input, &predicted_out);
        }

        rmse
    }
}
