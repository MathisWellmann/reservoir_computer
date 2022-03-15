#[macro_use]
extern crate log;

mod activation;
mod errors;
mod esn;
mod load_sample_data;

use std::time::Instant;

use nalgebra::{Const, Dim, Matrix};
use plotters::prelude::*;
use time_series_generator::generate_sine_wave;

use crate::{
    activation::Activation,
    esn::{Inputs, Params, Targets, ESN},
};

type Series = Vec<(f64, f64)>;

pub(crate) const INPUT_DIM: usize = 1;
pub(crate) const OUTPUT_DIM: usize = 1;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    info!("loading sample data");
    /*
    let series = load_sample_data::load_sample_data();
    let values: Vec<f64> =
        series.iter().skip(1).zip(series.iter()).map(|(a, b)| (a / b).ln()).collect();
    */
    let mut values: Vec<f64> = generate_sine_wave(100);
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    info!("got {} datapoints", values.len());

    const TRAINING_WINDOW: usize = 400;

    let t0 = Instant::now();

    let params = Params {
        input_sparsity: 0.5,
        input_activation: Activation::Tanh,
        input_weight_scaling: 1.0,
        input_bias_scaling: 0.1,

        reservoir_size: 20,
        reservoir_fixed_in_degree_k: 2,
        reservoir_activation: Activation::Tanh,

        feedback_gain: 0.1,
        spectral_radius: 0.8,
        leaking_rate: 0.05,
        regularization_coeff: 0.1,
        washout_pct: 0.25,
        output_tanh: false,
        seed: None,
    };
    let mut rc = ESN::new(params);
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
    rc.train(&train_inputs, &train_targets);
    info!("training done in: {}ms", t0.elapsed().as_millis());

    let mut plot_targets: Series = Vec::with_capacity(1_000_000);
    let mut plot_predictions: Series = Vec::with_capacity(1_000_000);

    let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);

    let n_vals = values.len();
    let inputs: Inputs =
        Matrix::from_vec_generic(Dim::from_usize(INPUT_DIM), Dim::from_usize(values.len()), values);
    rc.reset_state();
    for j in 0..n_vals {
        plot_targets.push((j as f64, *inputs.column(j).get(0).unwrap()));

        let predicted_out = rc.readout();
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        if j == TRAINING_WINDOW {
            plot_predictions.push((j as f64, last_prediction));
        }
        // TO begin forecasting, replace target input with it's own prediction
        let input = if j > TRAINING_WINDOW {
            plot_predictions.push((j as f64, last_prediction));
            predicted_out.column(0)
        } else {
            train_predictions.push((j as f64, last_prediction));
            inputs.column(j)
        };

        rc.update_state(&input, &predicted_out);
    }

    //let targets = series.iter().enumerate().take(TRAINING_WINDOW * 2).map(|(i,
    // y)| (i as f64, *y as f64)).collect();
    plot(&plot_targets, &train_predictions, &plot_predictions, "img/plot.png");
}

fn plot(targets: &Series, train_preds: &Series, test_preds: &Series, filename: &str) {
    info!("train_preds: {:?}", train_preds);
    let ts_min = targets[0].0;
    let ts_max = targets[targets.len() - 1].0;
    let mut target_min: f64 = targets[0].1;
    let mut target_max: f64 = targets[targets.len() - 1].1;
    for t in targets {
        if t.1 < target_min {
            target_min = t.1;
        }
        if t.1 > target_max {
            target_max = t.1;
        }
    }

    let dims = (2160, 2160);
    let root_area = BitMapBackend::new(filename, dims).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let root_area = root_area.titled(filename, ("sans-serif", 20).into_font()).unwrap();

    let areas = root_area.split_evenly((1, 1));

    let mut cc0 = ChartBuilder::on(&areas[0])
        .margin(5)
        .set_all_label_area_size(50)
        .caption("price", ("sans-serif", 30).into_font().with_color(&BLACK))
        .build_cartesian_2d(ts_min..ts_max, target_min..target_max)
        .unwrap();
    cc0.configure_mesh()
        .x_labels(20)
        .y_labels(20)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.4}", v))
        .draw()
        .unwrap();

    cc0.draw_series(LineSeries::new(targets.clone(), &BLACK))
        .unwrap()
        .label("targets")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    cc0.draw_series(LineSeries::new(train_preds.clone(), &RED))
        .unwrap()
        .label("train_preds")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    cc0.draw_series(LineSeries::new(test_preds.clone(), &GREEN))
        .unwrap()
        .label("test_preds")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    cc0.configure_series_labels().border_style(&BLACK).draw().unwrap();
}
