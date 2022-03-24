use std::{sync::Arc, time::Instant};

use dialoguer::{Select, theme::ColorfulTheme};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use sliding_features::{Constant, Echo, Multiply, View, ALMA, VSCT};

use crate::{Series, activation::Activation, experiments::trades::gif_render::GifRender, firefly_optimizer::{FireflyOptimizer, FireflyParams, ParameterMapper}, load_sample_data, reservoir_computers::esn};

const INPUT_DIM: usize = 1;
const OUTPUT_DIM: usize = 1;
const SEED: Option<u64> = Some(0)
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

    let rcs = vec![
        "ESN",
        "EuSN",
        "NG-RC",
    ];
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
                output_tanh: false,
                seed: SEED,
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
            };

            let mut rc = esn::ESN::new(params);

        },
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

        }
        2 => {
            todo!()
        }
        _ => panic!("invalid reservoir computer selection")
    }
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
    let mut opt = FireflyOptimizer::new(params);
    */

    let num_candidates = 96;

    let mut gif_render =
        GifRender::new("img/trades_sliding_window.gif", (1080, 1080), num_candidates);
    // TODO: iterate over all data
    for i in (TRAIN_LEN + VALIDATION_LEN + 1)..100_000 {
        if i % 100 == 0 {
            info!("step @ {}", i);
            let t1 = Instant::now();

            let train_inputs: Inputs = Matrix::from_vec_generic(
                Dim::from_usize(TRAIN_LEN),
                Dim::from_usize(INPUT_DIM),
                values[i - TRAIN_LEN - VALIDATION_LEN - 1..i - VALIDATION_LEN - 1].to_vec(),
            );
            let train_targets: Targets = Matrix::from_vec_generic(
                Dim::from_usize(TRAIN_LEN),
                Dim::from_usize(INPUT_DIM),
                values[i - TRAIN_LEN - VALIDATION_LEN..i - VALIDATION_LEN].to_vec(),
            );
            let inputs: Inputs = Matrix::from_vec_generic(
                Dim::from_usize(TRAIN_LEN + VALIDATION_LEN),
                Dim::from_usize(INPUT_DIM),
                values[i - TRAIN_LEN - VALIDATION_LEN - 1..i - 1].to_vec(),
            );
            let targets: Targets = Matrix::from_vec_generic(
                Dim::from_usize(TRAIN_LEN + VALIDATION_LEN),
                Dim::from_usize(INPUT_DIM),
                values[i - TRAIN_LEN - VALIDATION_LEN..i].to_vec(),
            );

            opt.step(
                Arc::new(train_inputs),
                Arc::new(train_targets),
                Arc::new(inputs),
                Arc::new(targets),
            );
            let mut rc = opt.elite();

            let vals_matrix: Inputs = Matrix::from_vec_generic(
                Dim::from_usize(TRAIN_LEN + VALIDATION_LEN + 1),
                Dim::from_usize(INPUT_DIM),
                values[i - TRAIN_LEN - VALIDATION_LEN - 1..i].to_vec(),
            );

            let (plot_targets, train_preds, test_preds) = gather_plot_data(&vals_matrix, &mut rc);
            gif_render.update(
                &plot_targets,
                &train_preds,
                &test_preds,
                opt.fits(),
                i,
                opt.candidates(),
            );

            info!("step took {}s", t1.elapsed().as_secs());
        }
    }

    info!("took {}s", t0.elapsed().as_secs());
}

fn sliding() {

}

fn run_rc<R: ReservoirComputer<P, I, O>, P, const I: usize, const O: usize>(r: R, values: Vec<f64>) {

}

fn gather_plot_data(values: &Inputs, rc: &mut ESN) -> (Series, Series, Series) {
    let mut plot_targets = Vec::with_capacity(values.len());
    let mut train_preds = vec![];
    let mut test_preds = vec![];
    for i in 0..values.nrows() {
        plot_targets.push((i as f64, *values.row(i).get(0).unwrap()));

        let predicted_out = rc.readout();
        let last_prediction = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(INPUT_DIM), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if i > TRAIN_LEN {
            test_preds.push((i as f64, last_prediction));
            m.row(0)
        } else {
            train_preds.push((i as f64, last_prediction));
            values.row(i)
        };

        rc.update_state(&input, &predicted_out);
    }

    (plot_targets, train_preds, test_preds)
}
