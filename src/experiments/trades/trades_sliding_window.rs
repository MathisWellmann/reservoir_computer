use std::time::Instant;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use sliding_features::{Echo, HLNormalizer, View, ALMA};

use crate::{
    activation::Activation,
    esn::{Inputs, Targets, ESN},
    experiments::trades::gif_render::GifRender,
    firefly_optimizer::{FireflyOptimizer, FireflyParams, ParameterMapper},
    load_sample_data, Series, INPUT_DIM,
};

const TRAIN_LEN: usize = 10_000;
const VALIDATION_LEN: usize = 2_000;

pub(crate) fn start() {
    info!("loading sample data");

    let series: Vec<f64> = load_sample_data::load_sample_data();

    let mut feature = HLNormalizer::new(ALMA::new(Echo::new(), 100), TRAIN_LEN);
    let mut values: Vec<f64> = Vec::with_capacity(series.len());
    for s in &series {
        feature.update(*s);
        values.push(feature.last());
    }
    info!("got {} datapoints", values.len());

    let t0 = Instant::now();

    let num_candidates = 10;
    let params = FireflyParams {
        gamma: 0.03,
        alpha: 0.005,
        step_size: 0.005,
        num_candidates,
        param_mapping: ParameterMapper::new(
            vec![(0.01, 0.2), (0.5, 1.0), (0.0, 0.5), (2.0, 10.0)],
            Activation::Identity,
            500,
            Activation::Tanh,
            0.02,
            0.1,
            None,
            0.0005,
            values[0],
        ),
    };
    let mut opt = FireflyOptimizer::new(params);

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

            opt.step(&train_inputs, &train_targets, &inputs, &targets);
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
