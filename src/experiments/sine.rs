use std::time::Instant;

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use time_series_generator::generate_sine_wave;

use crate::{
    activation::Activation,
    plot::plot,
    reservoir_computers::{esn, RCParams, ReservoirComputer},
    Series,
};

const INPUT_DIM: usize = 1;
const OUTPUT_DIM: usize = 1;
const TRAINING_WINDOW: usize = 600;

pub(crate) fn start() {
    info!("loading sample data");

    const TRAINING_WINDOW: usize = 600;

    let mut values: Vec<f64> = generate_sine_wave(100);
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    values.append(&mut values.clone());
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
                input_weight_scaling: 0.5,
                reservoir_bias_scaling: 0.0,

                reservoir_size: 40,
                reservoir_sparsity: 0.1,
                reservoir_activation: Activation::Tanh,

                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate: 0.2,
                regularization_coeff: 0.1,
                washout_pct: 0.1,
                output_activation: Activation::Identity,
                seed: Some(0),
                state_update_noise_frac: 0.01,
                initial_state_value: 0.0,
            };
            let mut rc = esn::ESN::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("training done in: {}ms", t0.elapsed().as_millis());

            run_rc::<esn::ESN<1, 1>, esn::Params, 1, 1>(&mut rc, values, "img/sine_esn.png");
        }
        1 => {
            todo!()
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
    filename: &str,
) {
    let mut plot_targets: Series = Vec::with_capacity(1_000_000);
    let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);
    let mut test_predictions: Series = Vec::with_capacity(1_000_000);

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
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        if j == TRAINING_WINDOW {
            test_predictions.push((j as f64, last_prediction));
        }
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

    plot(&plot_targets, &train_predictions, &test_predictions, filename, (2160, 2160));
}
