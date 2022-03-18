use std::time::Instant;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use time_series_generator::generate_sine_wave;

use crate::{
    activation::Activation,
    esn::{Inputs, Params, ESN},
    plot::plot,
    Series, INPUT_DIM, OUTPUT_DIM,
};

pub(crate) fn start() {
    info!("loading sample data");

    const TRAINING_WINDOW: usize = 600;

    let mut values: Vec<f64> = generate_sine_wave(100);
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    values.append(&mut values.clone());

    info!("got {} datapoints", values.len());

    let t0 = Instant::now();

    let params = Params {
        input_sparsity: 0.1,
        input_activation: Activation::Identity,
        input_weight_scaling: 0.5,
        reservoir_bias_scaling: 0.0,

        reservoir_size: 80,
        reservoir_fixed_in_degree_k: 2,
        reservoir_activation: Activation::Tanh,

        feedback_gain: 0.0,
        spectral_radius: 0.9,
        leaking_rate: 0.2,
        regularization_coeff: 0.1,
        washout_pct: 0.1,
        output_tanh: false,
        seed: Some(0),
        state_update_noise_frac: 0.01,
        initial_state_value: 0.0,
    };
    let mut rc = ESN::new(params);
    let train_inputs = Matrix::from_vec_generic(
        Dim::from_usize(TRAINING_WINDOW),
        Dim::from_usize(INPUT_DIM),
        values.iter().take(TRAINING_WINDOW).cloned().collect::<Vec<f64>>(),
    );
    let train_targets = Matrix::from_vec_generic(
        Dim::from_usize(TRAINING_WINDOW),
        Dim::from_usize(OUTPUT_DIM),
        values.iter().skip(1).take(TRAINING_WINDOW).cloned().collect::<Vec<f64>>(),
    );
    rc.train(&train_inputs, &train_targets);
    info!("training done in: {}ms", t0.elapsed().as_millis());

    let mut plot_targets: Series = Vec::with_capacity(1_000_000);

    let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);
    let mut test_predictions: Series = Vec::with_capacity(1_000_000);

    let n_vals = values.len();
    let inputs: Inputs =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(INPUT_DIM), values);
    rc.reset_state();
    for i in 0..n_vals {
        plot_targets.push((i as f64, *inputs.row(i).get(0).unwrap()));

        let predicted_out = rc.readout();
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        if i == TRAINING_WINDOW {
            test_predictions.push((i as f64, last_prediction));
        }
        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(INPUT_DIM), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if i > TRAINING_WINDOW {
            test_predictions.push((i as f64, last_prediction));
            m.row(0)
        } else {
            train_predictions.push((i as f64, last_prediction));
            inputs.row(i)
        };

        rc.update_state(&input, &predicted_out);
    }

    plot(&plot_targets, &train_predictions, &test_predictions, "img/sine.png", (2160, 2160));
}
