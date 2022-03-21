use std::time::Instant;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use sliding_features::{ALMA, Echo, HLNormalizer, View};

use crate::{INPUT_DIM, Series, esn::{ESN, Inputs, Targets}, experiments::trades::gif_render::GifRender, firefly_optimizer::{FireflyOptimizer, FireflyParams}, load_sample_data};

const WINDOW_LEN: usize = 10_000;
const TRAIN_LEN: usize = 8_000;

pub(crate) fn start() {
    info!("loading sample data");


    let series: Vec<f64> = load_sample_data::load_sample_data();

    let mut feature = HLNormalizer::new(ALMA::new(Echo::new(), 100), 100);
    let mut values: Vec<f64> = Vec::with_capacity(series.len());
    for s in &series {
        feature.update(*s);
        values.push(feature.last());
    }
    info!("got {} datapoints", values.len());

    let t0 = Instant::now();

    let n_vals = values.len();

    let params = FireflyParams {

    };
    let mut opt = FireflyOptimizer::new(params);


    let mut gif_render = GifRender::new("img/trades_sliding_window.gif", (1080, 1080));
    for i in (WINDOW_LEN + 1)..n_vals {
        if i % 50 == 0 {
            let vals = &values[i - WINDOW_LEN - 1..i - 1];

            let inputs: Inputs = Matrix::from_vec_generic(
                Dim::from_usize(WINDOW_LEN),
                Dim::from_usize(INPUT_DIM),
                vals.to_vec(),
            );
            let targets: Targets = Matrix::from_vec_generic(
                Dim::from_usize(WINDOW_LEN),
                Dim::from_usize(INPUT_DIM),
                values[i - WINDOW_LEN..i].to_vec()
            );
            opt.step(&inputs, &targets);
            let mut rc = opt.elite();

            let predicted_out = rc.readout();
            let last_prediction = *predicted_out.get(0).unwrap();

            let vals_matrix: Inputs =
                Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(INPUT_DIM), vals.to_vec());

            let (plot_targets, train_preds, test_preds) = gather_plot_data(&vals_matrix, &mut rc);
            gif_render.update(&plot_targets, &train_preds, &test_preds);
        }
    }
}

fn gather_plot_data(values: &Inputs, rc: &mut ESN) -> (Series, Series, Series) {
    let mut plot_targets = Vec::with_capacity(values.len());
    let mut train_preds = vec![];
    let mut test_preds = vec![];
    for i in 0..values.len() {
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
