use std::sync::Arc;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use super::{OptEnvironment, PlotGather};
use crate::{reservoir_computers::StateMatrix, RCParams, ReservoirComputer, SingleDimIo};

pub struct EnvTrades {
    train_inputs: Arc<SingleDimIo>,
    train_targets: Arc<SingleDimIo>,
    inputs: Arc<SingleDimIo>,
    targets: Arc<SingleDimIo>,
}

impl EnvTrades {
    #[inline(always)]
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

impl<R, const N: usize> OptEnvironment<R, 1, 1, N> for EnvTrades
where R: ReservoirComputer<1, 1, N>
{
    fn evaluate(&self, rc: &mut R, mut plot: Option<&mut PlotGather>) -> f64 {
        rc.train(&self.train_inputs, &self.train_targets);

        let training_len = self.train_inputs.ncols();

        let n_vals = self.inputs.len();
        let init_val = *self.inputs.column(0).get(0).unwrap();
        let state: StateMatrix = Matrix::from_element_generic(
            Dim::from_usize(rc.params().reservoir_size()),
            Dim::from_usize(1),
            init_val,
        );
        rc.set_state(state);

        let mut rmse: f64 = 0.0;
        for j in 0..n_vals {
            if let Some(plot) = plot.as_mut() {
                plot.push_target(j as f64, *self.inputs.column(j).get(0).unwrap());
            }
            let predicted_out = rc.readout();
            let last_prediction = *predicted_out.get(0).unwrap();

            // To begin forecasting, replace target input with it's own prediction
            let m: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |i, _| {
                    *predicted_out.get(i).unwrap()
                });
            let target = *self.targets.column(j).get(0).unwrap();
            rmse += (last_prediction - target).powi(2);

            let input = if j > training_len {
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

            rc.update_state(&input, &predicted_out);
        }

        rmse
    }
}
