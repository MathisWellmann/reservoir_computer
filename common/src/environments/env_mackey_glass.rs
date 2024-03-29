use std::sync::Arc;

use lin_reg::LinReg;
use nalgebra::{DMatrix, Dim, Matrix};

use super::PlotGather;
use crate::ReservoirComputer;

/// A mackey glass environment.
/// See: <https://en.wikipedia.org/wiki/Mackey%E2%80%93Glass_equations>
pub struct EnvMackeyGlass {
    values: Arc<DMatrix<f64>>,
    train_len: usize,
}

impl EnvMackeyGlass {
    /// Construct a new instance of this environment
    pub fn new(values: Arc<DMatrix<f64>>, train_len: usize) -> Self {
        assert!(
            values.nrows() > train_len,
            "make sure train_len < number of datapoints"
        );

        Self { values, train_len }
    }
}

impl EnvMackeyGlass {
    /// Evaluate a reservoir computer on the mackey glass benchmark
    pub fn evaluate<RC, R>(&self, rc: &mut RC, mut plot: Option<&mut PlotGather>) -> f64
    where
        RC: ReservoirComputer<R>,
        R: LinReg,
    {
        rc.train(
            &self.values.rows(0, self.train_len - 1),
            &self.values.rows(1, self.train_len),
        );

        let vals: Vec<f64> = vec![rc.params().initial_state_value(); rc.params().reservoir_size()];
        let state = Matrix::from_vec_generic(
            Dim::from_usize(1),
            Dim::from_usize(rc.params().reservoir_size()),
            vals,
        );
        rc.set_state(state);

        let mut rmse = 0.0;
        for i in 1..self.values.nrows() {
            if let Some(plot) = plot.as_mut() {
                plot.push_target(i as f64, *self.values.row(i).get(0).unwrap());
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
            rmse += (*self.values.row(i).get(0).unwrap() - last_prediction).powi(2);

            rc.update_state(&input);
        }

        rmse
    }
}
