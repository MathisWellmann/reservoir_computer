#[macro_use]
extern crate log;

mod plot;

use std::time::Instant;

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, VecStorage};
use time_series_generator::generate_sine_wave;

use plot::{plot, Series};
use reservoir_computer::{
    esn, ngrc, Activation, LinReg, RCParams, ReservoirComputer, TikhonovRegularization,
};

const TRAIN_LEN: usize = 600;
const SEED: Option<u64> = Some(0);

pub(crate) fn main() {
    info!("loading sample data");

    let mut values: Vec<f64> = generate_sine_wave(100);
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    info!("got {} datapoints", values.len());

    let values: DMatrix<f64> =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(1), values);

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

                reservoir_size: 500,
                reservoir_sparsity: 0.1,
                reservoir_activation: Activation::Tanh,

                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate: 0.2,
                regularization_coeff: 0.1,
                washout_pct: 0.1,
                output_activation: Activation::Identity,
                seed: SEED,
                state_update_noise_frac: 0.005,
                initial_state_value: 0.0,
                readout_from_input_as_well: true,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            let mut rc = esn::ESN::new(params, regressor);

            let t0 = Instant::now();
            rc.train(&values.columns(0, TRAIN_LEN - 1), &values.columns(1, TRAIN_LEN));
            info!("training done in: {}ms", t0.elapsed().as_millis());

            run_rc::<esn::ESN<1, 1, TikhonovRegularization>, 7, TikhonovRegularization>(
                &mut rc,
                &values,
                "img/sine_esn.png",
            );
        }
        1 => {
            todo!()
            /*
            let params = eusn::Params {
                input_sparsity: 0.1,
                input_weight_scaling: 1.0,
                reservoir_size: 100,
                reservoir_weight_scaling: 0.05,
                reservoir_bias_scaling: 0.5,
                reservoir_activation: Activation::Tanh,
                initial_state_value: 0.0,
                seed: SEED,
                washout_frac: 0.1,
                regularization_coeff: 0.01,
                epsilon: 0.01,
                gamma: 0.001,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            let mut rc = eusn::EulerStateNetwork::new(params, regressor);

            let t0 = Instant::now();
            rc.train(&values.columns(0, TRAIN_LEN - 1), &values.columns(1, TRAIN_LEN));
            info!("ESN training done in {}ms", t0.elapsed().as_millis());

            run_rc(&mut rc, &values, "img/sine_eusn.png");
            */
        }
        2 => {
            let params = ngrc::Params {
                input_dim: 1,
                output_dim: 1,
                num_time_delay_taps: 10,
                num_samples_to_skip: 10,
                output_activation: Activation::Identity,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            let mut rc = ngrc::NextGenerationRC::new(params, regressor);
            let t0 = Instant::now();
            rc.train(&values.columns(0, TRAIN_LEN - 1), &values.columns(1, TRAIN_LEN));
            info!("NGRC training took {}ms", t0.elapsed().as_millis());

            run_rc::<ngrc::NextGenerationRC<TikhonovRegularization>, 3, TikhonovRegularization>(
                &mut rc,
                &values,
                "img/sine_ngrc.png",
            );
        }
        _ => panic!("invalid reservoir computer selection"),
    }
}

fn run_rc<RC, const N: usize, R>(rc: &mut RC, values: &DMatrix<f64>, filename: &str)
where
    RC: ReservoirComputer<N, R>,
    R: LinReg,
{
    let mut plot_targets: Series = Vec::with_capacity(1_000_000);
    let mut train_predictions: Series = Vec::with_capacity(TRAIN_LEN);
    let mut test_predictions: Series = Vec::with_capacity(1_000_000);

    let n_vals = values.len();
    let state = Matrix::from_element_generic(
        Dim::from_usize(rc.params().reservoir_size()),
        Dim::from_usize(1),
        rc.params().initial_state_value(),
    );
    rc.set_state(state);
    for j in 1..n_vals {
        plot_targets.push((j as f64, *values.column(j).get(0).unwrap()));

        let predicted_out = rc.readout();
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        if j == TRAIN_LEN {
            test_predictions.push((j as f64, last_prediction));
        }
        // To begin forecasting, replace target input with it's own prediction
        let m: DMatrix<f64> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |i, _| {
                *predicted_out.get(i).unwrap()
            });
        let input = if j > TRAIN_LEN {
            test_predictions.push((j as f64, last_prediction));
            m.column(0)
        } else {
            train_predictions.push((j as f64, last_prediction));
            values.column(j - 1)
        };

        rc.update_state(&input, &predicted_out);
    }

    plot(&plot_targets, &train_predictions, &test_predictions, filename, (2160, 2160));
}
