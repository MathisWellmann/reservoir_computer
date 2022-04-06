use std::sync::Arc;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use crate::{OptEnvironment, RCParams, ReservoirComputer};

pub struct EnvMackeyGlass {
    pub train_inputs: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub train_targets: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub inputs: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub targets: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
}

impl<R, const N: usize> OptEnvironment<R, 1, 1, N> for EnvMackeyGlass
where R: ReservoirComputer<1, 1, N>
{
    fn evaluate(&self, rc: &mut R) -> f64 {
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
                // TODO: plot support
                //test_predictions.push((j as f64, last_prediction));
                m.column(0)
            } else {
                // TODO: plot support
                //train_predictions.push((j as f64, last_prediction));
                self.inputs.column(j)
            };
            rmse += (*self.inputs.column(j).get(0).unwrap() - last_prediction).powi(2);

            rc.update_state(&input, &predicted_out);
        }

        rmse
    }
}
