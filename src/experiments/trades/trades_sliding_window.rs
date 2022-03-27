use std::time::Instant;

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use sliding_features::{Constant, Echo, Multiply, View, ALMA, VSCT};

use crate::{
    activation::Activation,
    experiments::trades::gif_render::GifRender,
    load_sample_data,
    reservoir_computers::{esn, eusn, RCParams, ReservoirComputer},
    Series,
};

const INPUT_DIM: usize = 1;
const SEED: Option<u64> = Some(0);
pub(crate) const TRAIN_LEN: usize = 10_000;
pub(crate) const VALIDATION_LEN: usize = 2_000;

pub(crate) fn start() {
    info!("loading sample data");

    let series: Vec<f64> = load_sample_data::load_sample_data();

    let mut feature =
        Multiply::new(VSCT::new(ALMA::new(Echo::new(), 100), TRAIN_LEN), Constant::new(0.2));
    let mut values: Vec<f64> = Vec::with_capacity(series.len());
    for s in &series {
        feature.update(*s);
        values.push(feature.last());
    }
    info!("got {} datapoints", values.len());

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
                washout_pct: 0.05,
                output_activation: Activation::Identity,
                seed: Some(0),
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
                readout_from_input_as_well: false,
            };

            let mut rc = esn::ESN::new(params);

            run_sliding::<esn::ESN<1, 1>, esn::Params, 1, 1>(
                &mut rc,
                values,
                "img/trades_sliding_window_esn.gif",
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

            run_sliding::<eusn::EulerStateNetwork<1, 1>, eusn::Params, 1, 1>(
                &mut rc,
                values,
                "img/trades_sliding_window_eusn.gif",
            );
        }
        2 => {
            todo!()
        }
        _ => panic!("invalid reservoir computer selection"),
    }
}

fn run_sliding<R: ReservoirComputer<P, I, O>, P: RCParams, const I: usize, const O: usize>(
    rc: &mut R,
    values: Vec<f64>,
    filename: &str,
) {
    let t0 = Instant::now();
    /*
    let params = FireflyParams {
        gamma: 50.0,
        alpha: 0.005,
        step_size: 0.005,
        num_candidates,
        param_mapping: ParameterMapper::new(
            vec![(0.05, 0.15), (0.9, 1.0), (0.0, 0.05), (7.0, 9.0)],
            Activation::Identity,
            100,
            Activation::Tanh,
            0.02,
            0.1,
            Some(0),
            0.0005,
            0.0,
        ),
    };
    let mut opt = FireflyOptimizer::<R, I, O>::new(params);
    */

    let mut gif_render = GifRender::new(filename, (1080, 1080));
    // TODO: iterate over all data
    for i in (TRAIN_LEN + VALIDATION_LEN + 1)..100_000 {
        if i % 100 == 0 {
            info!("step @ {}", i);
            let t1 = Instant::now();

            let train_inputs: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
                Matrix::from_vec_generic(
                    Dim::from_usize(INPUT_DIM),
                    Dim::from_usize(TRAIN_LEN),
                    values[i - TRAIN_LEN - VALIDATION_LEN - 1..i - VALIDATION_LEN - 1].to_vec(),
                );
            let train_targets: Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>> =
                Matrix::from_vec_generic(
                    Dim::from_usize(INPUT_DIM),
                    Dim::from_usize(TRAIN_LEN),
                    values[i - TRAIN_LEN - VALIDATION_LEN..i - VALIDATION_LEN].to_vec(),
                );
            let inputs: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
                Matrix::from_vec_generic(
                    Dim::from_usize(INPUT_DIM),
                    Dim::from_usize(TRAIN_LEN + VALIDATION_LEN),
                    values[i - TRAIN_LEN - VALIDATION_LEN - 1..i - 1].to_vec(),
                );
            let targets: Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>> =
                Matrix::from_vec_generic(
                    Dim::from_usize(INPUT_DIM),
                    Dim::from_usize(TRAIN_LEN + VALIDATION_LEN),
                    values[i - TRAIN_LEN - VALIDATION_LEN..i].to_vec(),
                );

            /*
            opt.step(
                Arc::new(train_inputs),
                Arc::new(train_targets),
                Arc::new(inputs),
                Arc::new(targets),
            );
            let mut rc = opt.elite();
             */

            rc.train(&train_inputs, &train_targets);

            let vals_matrix: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
                Matrix::from_vec_generic(
                    Dim::from_usize(INPUT_DIM),
                    Dim::from_usize(TRAIN_LEN + VALIDATION_LEN + 1),
                    values[i - TRAIN_LEN - VALIDATION_LEN - 1..i].to_vec(),
                );

            let state = Matrix::from_element_generic(
                Dim::from_usize(rc.params().reservoir_size()),
                Dim::from_usize(1),
                values[i - TRAIN_LEN - VALIDATION_LEN - 1],
            );
            rc.set_state(state);

            let (plot_targets, train_preds, test_preds) = gather_plot_data(&vals_matrix, rc);
            gif_render.update(&plot_targets, &train_preds, &test_preds);

            info!("step took {}s", t1.elapsed().as_secs());
        }
    }

    info!("took {}s", t0.elapsed().as_secs());
}

fn gather_plot_data<R: ReservoirComputer<P, I, O>, P: RCParams, const I: usize, const O: usize>(
    values: &Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>>,
    rc: &mut R,
) -> (Series, Series, Series) {
    let mut plot_targets = Vec::with_capacity(values.len());
    let mut train_preds = vec![];
    let mut test_preds = vec![];
    for j in 0..values.ncols() {
        plot_targets.push((j as f64, *values.column(j).get(0).unwrap()));

        let predicted_out = rc.readout();
        let last_prediction = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>> =
            Matrix::from_fn_generic(Dim::from_usize(I), Dim::from_usize(1), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if j > TRAIN_LEN {
            test_preds.push((j as f64, last_prediction));
            m.column(0)
        } else {
            train_preds.push((j as f64, last_prediction));
            values.column(j)
        };

        rc.update_state(&input, &predicted_out);
    }

    (plot_targets, train_preds, test_preds)
}
