use std::{sync::Arc, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use sliding_features::{Echo, View, ALMA};

use crate::{
    activation::Activation,
    experiments::trades::gif_render_firefly::GifRenderFirefly,
    firefly_optimizer::{FireflyOptimizer, FireflyParams, ParameterMapper},
    load_sample_data,
    plot::plot,
    reservoir_computers::{esn, eusn, RCParams, ReservoirComputer},
    utils::scale,
    Series,
};

const INPUT_DIM: usize = 1;
const OUTPUT_DIM: usize = 1;
const TRAINING_WINDOW: usize = 10_000;
const TEST_WINDOW: usize = 5000;

pub(crate) fn start() {
    info!("loading sample data");

    let series: Vec<f64> = load_sample_data::load_sample_data()
        .iter()
        .take(TRAINING_WINDOW + TEST_WINDOW)
        .cloned()
        .collect();
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

    let train_inputs = Matrix::from_vec_generic(
        Dim::from_usize(INPUT_DIM),
        Dim::from_usize(TRAINING_WINDOW),
        values.iter().take(TRAINING_WINDOW).cloned().collect::<Vec<f64>>(),
    );
    let train_targets = Matrix::from_vec_generic(
        Dim::from_usize(OUTPUT_DIM),
        Dim::from_usize(TRAINING_WINDOW),
        values.iter().skip(1).take(TRAINING_WINDOW).cloned().collect::<Vec<f64>>(),
    );

    let rcs = vec!["ESN", "EuSN", "NG-RC", "ESN-Firefly"];
    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Reservoir Computer")
        .items(&rcs)
        .default(0)
        .interact()
        .unwrap();
    match e {
        0 => {
            let washout_pct = 0.0;
            let params = esn::Params {
                input_sparsity: 0.2,
                input_activation: Activation::Identity,
                input_weight_scaling: 0.2,
                reservoir_bias_scaling: 0.05,

                reservoir_size: 500,
                reservoir_sparsity: 0.02,
                reservoir_activation: Activation::Tanh,

                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate: 0.02,
                regularization_coeff: 0.02,
                washout_pct,
                output_activation: Activation::Identity,
                seed: Some(0),
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
                readout_from_input_as_well: false,
            };
            let mut rc = esn::ESN::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("trained readout_matrix: {}", rc.readout_matrix());
            info!("training done in: {}ms", t0.elapsed().as_millis());

            let mut plot_targets: Series = Vec::with_capacity(1_000_000);
            let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);
            let mut test_predictions: Series = Vec::with_capacity(1_000_000);

            run_trained_rc(
                &mut rc,
                &values,
                washout_pct,
                &mut plot_targets,
                &mut train_predictions,
                &mut test_predictions,
            );

            plot(
                &plot_targets,
                &train_predictions,
                &test_predictions,
                "img/trades_esn.png",
                (2160, 2160),
            );
        }
        1 => {
            let washout_frac = 0.05;
            let params = eusn::Params {
                input_sparsity: 1.0,
                input_weight_scaling: 0.05,
                reservoir_size: 500,
                reservoir_weight_scaling: 0.1,
                reservoir_bias_scaling: 0.04,
                reservoir_activation: Activation::Tanh,
                initial_state_value: values[0],
                seed: Some(0),
                washout_frac: 0.05,
                regularization_coeff: 0.1,
                epsilon: 0.01,
                gamma: 0.001,
            };
            let mut rc = eusn::EulerStateNetwork::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("training done in: {}ms", t0.elapsed().as_millis());

            let mut plot_targets: Series = Vec::with_capacity(1_000_000);
            let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);
            let mut test_predictions: Series = Vec::with_capacity(1_000_000);

            run_trained_rc::<eusn::EulerStateNetwork<1, 1>, eusn::Params, 1, 1>(
                &mut rc,
                &values,
                washout_frac,
                &mut plot_targets,
                &mut train_predictions,
                &mut test_predictions,
            );

            plot(
                &plot_targets,
                &train_predictions,
                &test_predictions,
                "img/trades_esn.png",
                (2160, 2160),
            );
        }
        2 => {
            todo!()
        }
        3 => {
            let washout_frac = 0.0;
            let num_candidates = 96;
            let params = FireflyParams {
                gamma: 50.0,
                alpha: 0.005,
                step_size: 0.005,
                num_candidates,
                param_mapping: ParameterMapper::new(
                    vec![(0.05, 0.15), (0.9, 1.0), (0.0, 0.05), (0.01, 0.1)],
                    Activation::Identity,
                    500,
                    Activation::Tanh,
                    0.02,
                    0.02,
                    Some(0),
                    0.001,
                    0.0,
                ),
            };
            let mut opt = FireflyOptimizer::new(params);

            let inputs: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                Matrix::from_vec_generic(
                    Dim::from_usize(INPUT_DIM),
                    Dim::from_usize(values.len()),
                    values.clone(),
                );
            let targets: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                Matrix::from_vec_generic(
                    Dim::from_usize(OUTPUT_DIM),
                    Dim::from_usize(values.len() - 1),
                    values.iter().skip(1).cloned().collect::<Vec<f64>>(),
                );

            let train_inputs = Arc::new(train_inputs);
            let train_targets = Arc::new(train_targets);
            let inputs = Arc::new(inputs);
            let targets = Arc::new(targets);

            let mut gif_render =
                GifRenderFirefly::new("img/trades_esn_firefly.gif", (1080, 1080), num_candidates);
            for i in 0..1000 {
                opt.step(
                    train_inputs.clone(),
                    train_targets.clone(),
                    inputs.clone(),
                    targets.clone(),
                );
                let mut rc = esn::ESN::new(opt.elite_params().clone());

                let mut plot_targets: Series = Vec::with_capacity(1_000_000);
                let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);
                let mut test_predictions: Series = Vec::with_capacity(1_000_000);

                run_trained_rc::<esn::ESN<1, 1>, esn::Params, 1, 1>(
                    &mut rc,
                    &values,
                    washout_frac,
                    &mut plot_targets,
                    &mut train_predictions,
                    &mut test_predictions,
                );
                gif_render.update(
                    &plot_targets,
                    &train_predictions,
                    &test_predictions,
                    opt.fits(),
                    i,
                    opt.candidates(),
                );
            }
        }
        _ => panic!("invalid reservoir computer selection"),
    }
}

fn run_trained_rc<R: ReservoirComputer<P, I, O>, P: RCParams, const I: usize, const O: usize>(
    rc: &mut R,
    values: &Vec<f64>,
    washout_pct: f64,
    plot_targets: &mut Series,
    train_predictions: &mut Series,
    test_predictions: &mut Series,
) {
    let washout_len = (values.len() as f64 * washout_pct) as usize;

    let n_vals = values.len();
    let init_val = values[washout_len];
    let inputs: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
        Matrix::from_vec_generic(
            Dim::from_usize(I),
            Dim::from_usize(values.len()),
            values.to_vec(),
        );
    let state = Matrix::from_element_generic(
        Dim::from_usize(rc.params().reservoir_size()),
        Dim::from_usize(1),
        init_val,
    );
    rc.set_state(state);

    for j in 0..n_vals {
        plot_targets.push((j as f64, *inputs.column(j).get(0).unwrap()));
        if j < washout_len {
            continue;
        }

        let predicted_out = rc.readout();
        let last_prediction = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
            Matrix::from_fn_generic(Dim::from_usize(I), Dim::from_usize(1), |i, _| {
                *predicted_out.get(i).unwrap()
            });
        let input = if j > TRAINING_WINDOW {
            test_predictions.push((j as f64, last_prediction));
            m.column(0)
        } else {
            train_predictions.push((j as f64, last_prediction));
            inputs.column(j)
        };

        rc.update_state(&input, &predicted_out);
    }
}
