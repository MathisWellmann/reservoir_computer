use std::sync::Arc;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use super::OptEnvironment;
use crate::{reservoir_computers::StateMatrix, RCParams, ReservoirComputer};

pub struct FFEnvTrades {
    pub train_inputs: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub train_targets: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub inputs: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub targets: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
}

impl<R, const N: usize> OptEnvironment<R, 1, 1, N> for FFEnvTrades
where R: ReservoirComputer<1, 1, N>
{
    fn evaluate(&self, rc: &mut R) -> f64 {
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
                m.column(0)
            } else {
                self.inputs.column(j)
            };

            rc.update_state(&input, &predicted_out);
        }

        rmse
    }
}
