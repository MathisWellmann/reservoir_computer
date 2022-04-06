use std::sync::Arc;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use super::PlotGather;
use crate::{OptEnvironment, RCParams, ReservoirComputer, SingleDimIo};

pub struct EnvMackeyGlass {
    train_inputs: Arc<SingleDimIo>,
    train_targets: Arc<SingleDimIo>,
    inputs: Arc<SingleDimIo>,
    targets: Arc<SingleDimIo>,
}

impl EnvMackeyGlass {
    #[inline]
    pub fn new(
        train_inputs: Arc<SingleDimIo>,
        train_targets: Arc<SingleDimIo>,
        inputs: Arc<SingleDimIo>,
        targets: Arc<SingleDimIo>,
    ) -> Self {
        Self {
            train_inputs,
            train_targets,
            inputs,
            targets,
        }
    }
}

impl<R, const N: usize> OptEnvironment<R, 1, 1, N> for EnvMackeyGlass
where R: ReservoirComputer<1, 1, N>
{
    fn evaluate(&self, rc: &mut R, mut plot: Option<&mut PlotGather>) -> f64 {
        rc.train(&self.train_inputs, &self.train_targets);

        let state = Matrix::from_element_generic(
            Dim::from_usize(rc.params().reservoir_size()),
            Dim::from_usize(1),
            rc.params().initial_state_value(),
        );
        rc.set_state(state);

        let train_len = self.train_inputs.ncols();
        let mut rmse = 0.0;
        for j in 0..self.inputs.ncols() {
            if let Some(plot) = plot.as_mut() {
                plot.push_target(j as f64, *self.inputs.column(j).get(0).unwrap());
            }

            let predicted_out = rc.readout();
            let mut last_prediction = *predicted_out.get(0).unwrap();
            if !last_prediction.is_finite() {
                last_prediction = 0.0;
            }

            // To begin forecasting, replace target input with it's own prediction
            let m: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |i, _| {
                    *predicted_out.get(i).unwrap()
                });
            let input = if j > train_len {
                if let Some(plot) = plot.as_mut() {
                    plot.push_test_pred(j as f64, last_prediction);
                }
                m.column(0)
            } else {
                if let Some(plot) = plot.as_mut() {
                    plot.push_train_pred(j as f64, last_prediction);
                }
                self.inputs.column(j)
            };
            rmse += (*self.targets.column(j).get(0).unwrap() - last_prediction).powi(2);

            rc.update_state(&input, &predicted_out);
        }

        rmse
    }
}
