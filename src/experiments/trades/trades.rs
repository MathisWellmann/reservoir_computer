use std::time::Instant;

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use sliding_features::{Echo, View, ALMA};

use crate::{
    activation::Activation,
    load_sample_data,
    plot::plot,
    reservoir_computers::{esn, eusn, RCParams, ReservoirComputer},
    utils::scale,
    Series,
};

const INPUT_DIM: usize = 1;
const OUTPUT_DIM: usize = 1;
const TRAINING_WINDOW: usize = 10_000;

pub(crate) fn start() {
    info!("loading sample data");

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
                input_weight_scaling: 1.0,
                reservoir_bias_scaling: 0.0,

                reservoir_size: 500,
                reservoir_fixed_in_degree_k: 8,
                reservoir_activation: Activation::Tanh,

                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate: 0.02,
                regularization_coeff: 0.1,
                washout_pct: 0.3,
                output_activation: Activation::Identity,
                seed: Some(0),
                state_update_noise_frac: 0.0005,
                initial_state_value: values[0],
            };
            let mut rc = esn::ESN::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("training done in: {}ms", t0.elapsed().as_millis());

            run_rc(&mut rc, values, "img/trades_esn.png");
        }
        1 => {
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

            run_rc::<eusn::EulerStateNetwork<1, 1>, eusn::Params, 1, 1>(
                &mut rc,
                values,
                "img/trades_esn.png",
            );
        }
        2 => {
            todo!()
        }
        _ => panic!("invalid reservoir computer selection"),
    }
}

fn run_rc<R: ReservoirComputer<P, I, O>, P: RCParams, const I: usize, const O: usize>(
    rc: &mut R,
    values: Vec<f64>,
    plot_filename: &str,
) {
    let mut plot_targets: Series = Vec::with_capacity(1_000_000);
    let mut plot_predictions: Series = Vec::with_capacity(1_000_000);

    let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);

    let n_vals = values.len();
    let inputs: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
        Matrix::from_vec_generic(Dim::from_usize(I), Dim::from_usize(values.len()), values);
    let state = Matrix::from_element_generic(
        Dim::from_usize(rc.params().reservoir_size()),
        Dim::from_usize(1),
        rc.params().initial_state_value(),
    );
    rc.set_state(state);

    for j in 0..n_vals {
        plot_targets.push((j as f64, *inputs.column(j).get(0).unwrap()));

        let predicted_out = rc.readout();
        let last_prediction = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
            Matrix::from_fn_generic(Dim::from_usize(I), Dim::from_usize(1), |i, _| {
                *predicted_out.get(i).unwrap()
            });
        let input = if j > TRAINING_WINDOW {
            plot_predictions.push((j as f64, last_prediction));
            m.column(0)
        } else {
            train_predictions.push((j as f64, last_prediction));
            inputs.column(j)
        };

        rc.update_state(&input, &predicted_out);
    }

    plot(&plot_targets, &train_predictions, &plot_predictions, plot_filename, (2160, 2160));
}
