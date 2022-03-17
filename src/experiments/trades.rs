use std::time::Instant;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use sliding_features::{Echo, View, ALMA};

use crate::{
    activation::Activation,
    esn::{Inputs, Params, ESN},
    load_sample_data,
    plot::plot,
    utils::scale,
    Series, INPUT_DIM, OUTPUT_DIM,
};

pub(crate) fn start() {
    info!("loading sample data");

    const TRAINING_WINDOW: usize = 10_000;

    let series: Vec<f64> =
        load_sample_data::load_sample_data().iter().take(TRAINING_WINDOW * 2).cloned().collect();
    let mut series_min = series[0];
    let mut series_max = series[0];
    for s in &series {
        if *s < series_min {
            series_min = *s;
        }
        if *s > series_max {
            series_max = *s;
        }
    }
    info!("series_min: {}, series_max: {}", series_min, series_max);

    let mut alma = ALMA::new(Echo::new(), 100);
    let mut values: Vec<f64> = Vec::with_capacity(series.len());
    for s in &series {
        let val = scale(series_min, series_max, -1.0, 1.0, *s);
        alma.update(val);
        values.push(alma.last());
    }
    info!("got {} datapoints", values.len());

    let t0 = Instant::now();

    let params = Params {
        input_sparsity: 0.1,
        input_activation: Activation::Identity,
        input_weight_scaling: 1.0,
        input_bias_scaling: 0.04,

        reservoir_size: 500,
        reservoir_fixed_in_degree_k: 10,
        reservoir_activation: Activation::Tanh,

        feedback_gain: 0.01,
        spectral_radius: 0.90,
        leaking_rate: 0.02,
        regularization_coeff: 0.1,
        washout_pct: 0.3,
        output_tanh: true,
        seed: Some(0),
        state_update_noise_frac: 0.001,
        initial_state_value: values[0],
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
    let mut plot_predictions: Series = Vec::with_capacity(1_000_000);

    let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);

    let n_vals = values.len();
    let inputs: Inputs =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(INPUT_DIM), values);
    rc.reset_state();

    for i in 0..n_vals {
        plot_targets.push((i as f64, *inputs.row(i).get(0).unwrap()));

        let predicted_out = rc.readout();
        let last_prediction = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(INPUT_DIM), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if i > TRAINING_WINDOW {
            plot_predictions.push((i as f64, last_prediction));
            m.row(0)
        } else {
            train_predictions.push((i as f64, last_prediction));
            inputs.row(i)
        };

        rc.update_state(&input, &predicted_out);
    }

    plot(&plot_targets, &train_predictions, &plot_predictions, "img/trades.png", (2160, 2160));
}
