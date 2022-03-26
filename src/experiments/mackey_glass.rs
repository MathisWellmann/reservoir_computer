use std::{collections::VecDeque, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use nanorand::{Rng, WyRand};

use crate::{
    activation::Activation,
    plot::plot,
    reservoir_computers::{esn, eusn, RCParams, ReservoirComputer},
    Series,
};

const TRAIN_LEN: usize = 5000;
const TEST_LEN: usize = 1000;
const SEED: Option<u64> = Some(0);

pub(crate) fn start() {
    let total_len = TRAIN_LEN + TEST_LEN;
    let values = mackey_glass_series(total_len, 30, SEED);
    info!("values: {:?}", values);

    let train_inputs = Matrix::from_vec_generic(
        Dim::from_usize(TRAIN_LEN),
        Dim::from_usize(1),
        values.iter().take(TRAIN_LEN).cloned().collect::<Vec<f64>>(),
    );
    let train_targets = Matrix::from_vec_generic(
        Dim::from_usize(TRAIN_LEN),
        Dim::from_usize(1),
        values.iter().skip(1).take(TRAIN_LEN).cloned().collect::<Vec<f64>>(),
    );

    let rcs = vec!["ESN", "EuSN", "NG-RC"];
    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Reservoir Computer")
        .items(&rcs)
        .default(0)
        .interact()
        .unwrap();
    match e {
        0 => {
            let params = esn::Params {
                input_sparsity: 0.1,
                input_activation: Activation::Identity,
                input_weight_scaling: 0.5,
                reservoir_bias_scaling: 0.01,

                reservoir_size: 200,
                reservoir_fixed_in_degree_k: 4,
                reservoir_activation: Activation::Tanh,

                feedback_gain: 0.0,
                spectral_radius: 0.99,
                leaking_rate: 0.15,
                regularization_coeff: 0.05,
                washout_pct: 0.1,
                output_activation: Activation::Identity,
                seed: SEED,
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
            };

            let mut rc = esn::ESN::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("ESN training done in {}ms", t0.elapsed().as_millis());

            run_rc::<esn::ESN<1, 1>, esn::Params, 1, 1>(
                &mut rc,
                values,
                "img/mackey_glass_esn.png",
            );
        }
        1 => {
            let params = eusn::Params {
                input_sparsity: 0.1,
                input_weight_scaling: 0.5,
                reservoir_size: 500,
                reservoir_weight_scaling: 0.7,
                reservoir_bias_scaling: 0.1,
                reservoir_activation: Activation::Relu,
                initial_state_value: values[0],
                seed: SEED,
                washout_frac: 0.1,
                regularization_coeff: 0.1,
                epsilon: 0.008,
                gamma: 0.05,
            };
            let mut rc = eusn::EulerStateNetwork::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("ESN training done in {}ms", t0.elapsed().as_millis());

            run_rc(&mut rc, values, "img/mackey_glass_eusn");
        }
        2 => {
            todo!()
        }
        _ => panic!("invalid reservoir computer selection"),
    };
}

fn run_rc<R: ReservoirComputer<P, I, O>, P: RCParams, const I: usize, const O: usize>(
    rc: &mut R,
    values: Vec<f64>,
    plot_filename: &str,
) {
    let mut plot_targets: Series = vec![];
    let mut train_predictions: Series = vec![];
    let mut test_predictions: Series = vec![];

    let n_vals = values.len();
    let inputs: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(1), values);
    let state = Matrix::from_element_generic(
        Dim::from_usize(rc.params().reservoir_size()),
        Dim::from_usize(1),
        rc.params().initial_state_value(),
    );
    rc.set_state(state);

    let mut train_rmse = 0.0;
    for j in 0..n_vals {
        plot_targets.push((j as f64, *inputs.row(j).get(0).unwrap()));

        let predicted_out = rc.readout();
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
            Matrix::from_fn_generic(Dim::from_usize(I), Dim::from_usize(1), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if j > TRAIN_LEN {
            test_predictions.push((j as f64, last_prediction));
            m.column(0)
        } else {
            train_rmse += (*inputs.row(j).get(0).unwrap() - last_prediction).powi(2);
            train_predictions.push((j as f64, last_prediction));
            inputs.column(j)
        };

        rc.update_state(&input, &predicted_out);
    }
    info!("train_rmse: {}", train_rmse.sqrt());

    plot(&plot_targets, &train_predictions, &test_predictions, plot_filename, (3840, 1080));
}

/// Mackey glass series
fn mackey_glass_series(sample_len: usize, tau: usize, seed: Option<u64>) -> Vec<f64> {
    let delta_t = 10;
    let mut timeseries = 1.2;
    let history_len = tau * delta_t;

    let mut rng = match seed {
        Some(seed) => WyRand::new_seed(seed),
        None => WyRand::new(),
    };
    let mut history = VecDeque::with_capacity(history_len);
    for _i in 0..history_len {
        let val = 1.2 + 0.2 * (rng.generate::<f64>() - 0.5);
        history.push_back(val);
    }

    let mut inp = vec![0.0; sample_len];

    for timestep in 0..sample_len {
        for _ in 0..delta_t {
            let x_tau = history.pop_front().unwrap();
            history.push_back(timeseries);
            let last_hist = history[history.len() - 1];
            timeseries = last_hist
                + (0.2 * x_tau / (1.0 + x_tau.powi(10)) - 0.1 * last_hist) / delta_t as f64;
        }
        inp[timestep] = timeseries;
    }
    // apply tanh nonlinearity
    inp.iter_mut().for_each(|v| *v = v.tanh());

    inp
}
