use std::sync::Arc;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use super::OptEnvironment;
use crate::{
    activation::Activation,
    reservoir_computers::{esn, StateMatrix},
    utils::scale,
    RCParams, ReservoirComputer,
};

pub type Range = (f64, f64);

pub struct FFEnvTradesESN {
    pub train_inputs: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub train_targets: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub inputs: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,
    pub targets: Arc<Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>>,

    /// Parameter ranges
    pub input_sparsity_range: Range,
    pub input_activation: Activation,
    pub input_weight_scaling_range: Range,
    pub reservoir_size: usize,
    pub reservoir_bias_scaling_range: Range,
    pub reservoir_sparsity_range: Range,
    pub reservoir_activation: Activation,
    pub feedback_gain: f64,
    pub spectral_radius: f64,
    pub leaking_rate: f64,
    pub regularization_coeff: f64,
    pub washout_pct: f64,
    pub output_activation: Activation,
    pub seed: Option<u64>,
    pub state_update_noise_frac: f64,
    pub initial_state_value: f64,
    pub readout_from_input_as_well: bool,
}

impl FFEnvTradesESN {
    pub fn map_params(&self, params: &[f64; 4]) -> esn::Params {
        esn::Params {
            input_sparsity: scale(
                0.0,
                1.0,
                self.input_sparsity_range.0,
                self.input_sparsity_range.1,
                params[0],
            ),
            input_activation: self.input_activation,
            input_weight_scaling: scale(
                0.0,
                1.0,
                self.input_weight_scaling_range.0,
                self.input_weight_scaling_range.1,
                params[1],
            ),
            reservoir_size: self.reservoir_size,
            reservoir_bias_scaling: scale(
                0.0,
                1.0,
                self.reservoir_bias_scaling_range.0,
                self.reservoir_bias_scaling_range.1,
                params[2],
            ),
            reservoir_sparsity: scale(
                0.0,
                1.0,
                self.reservoir_sparsity_range.0,
                self.reservoir_sparsity_range.1,
                params[3],
            ),
            reservoir_activation: self.reservoir_activation,
            feedback_gain: self.feedback_gain,
            spectral_radius: self.spectral_radius,
            leaking_rate: self.leaking_rate,
            regularization_coeff: self.regularization_coeff,
            washout_pct: self.washout_pct,
            output_activation: self.output_activation,
            seed: self.seed,
            state_update_noise_frac: self.state_update_noise_frac,
            initial_state_value: self.initial_state_value,
            readout_from_input_as_well: self.readout_from_input_as_well,
        }
    }
}

impl OptEnvironment<4> for FFEnvTradesESN {
    fn evaluate(&self, params: &[f64; 4]) -> f64 {
        let params = self.map_params(params);
        let mut rc = esn::ESN::<1, 1>::new(params);

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
