use std::collections::VecDeque;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use nanorand::{Rng, WyRand};

use crate::{
    activation::Activation,
    esn::{Inputs, Params, ESN},
    plot::plot,
    Series,
};

pub(crate) fn start() {
    let train_len = 5000;
    let test_len = 1000;
    let total_len = train_len + test_len;
    let values = mackey_glass_series(total_len, 30);
    info!("values: {:?}", values);

    let params = Params {
        input_sparsity: 0.1,
        input_activation: Activation::Identity,
        input_weight_scaling: 1.0,
        input_bias_scaling: 0.2,

        reservoir_size: 100,
        reservoir_fixed_in_degree_k: 2,
        reservoir_activation: Activation::Tanh,

        feedback_gain: 0.0,
        spectral_radius: 0.90,
        leaking_rate: 1.0,
        regularization_coeff: 0.1,
        washout_pct: 0.0,
        output_tanh: false,
        seed: Some(0),
        state_update_noise_frac: 0.001,
        initial_state_value: values[0],
    };

    let mut rc = ESN::new(params);
    let train_inputs = Matrix::from_vec_generic(
        Dim::from_usize(train_len),
        Dim::from_usize(1),
        values.iter().take(train_len).cloned().collect::<Vec<f64>>(),
    );
    let train_targets = Matrix::from_vec_generic(
        Dim::from_usize(train_len),
        Dim::from_usize(1),
        values.iter().skip(1).take(train_len).cloned().collect::<Vec<f64>>(),
    );
    rc.train(&train_inputs, &train_targets);

    let mut plot_targets: Series = vec![];
    let mut train_predictions: Series = vec![];
    let mut test_predictions: Series = vec![];

    let inputs: Inputs =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(1), values);
    rc.reset_state();

    let mut train_rmse = 0.0;
    for i in 0..total_len {
        plot_targets.push((i as f64, *inputs.row(i).get(0).unwrap()));

        let predicted_out = rc.readout();
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if i > train_len {
            test_predictions.push((i as f64, last_prediction));
            m.row(0)
        } else {
            train_rmse += (*inputs.row(i).get(0).unwrap() - last_prediction).powi(2);
            train_predictions.push((i as f64, last_prediction));
            inputs.row(i)
        };

        rc.update_state(&input, &predicted_out);
    }
    info!("test_predictions: {:?}", test_predictions);
    info!("train_rmse: {}", train_rmse.sqrt());

    plot(
        &plot_targets,
        &train_predictions,
        &test_predictions,
        "img/mackey_glass.png",
        (3840, 1080),
    );
}

/// Mackey glass series
pub(crate) fn mackey_glass_series(sample_len: usize, tau: usize) -> Vec<f64> {
    let delta_t = 10;
    let mut timeseries = 1.2;
    let history_len = tau * delta_t;

    let mut rng = WyRand::new();
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
