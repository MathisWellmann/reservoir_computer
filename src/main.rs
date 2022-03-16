#[macro_use]
extern crate log;

mod activation;
mod errors;
mod esn;
mod load_sample_data;

use std::time::Instant;

use nalgebra::{ArrayStorage, Const, Dim, Dynamic, Matrix, VecStorage};
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

    const TRAINING_WINDOW: usize = 600;

    /*
    let series = load_sample_data::load_sample_data();
    let values: Vec<f64> =
        series.iter()
              .skip(1)
              .zip(series.iter())
              .take(TRAINING_WINDOW * 2)
              .map(|(a, b)| (a / b).ln() * 100.0)
              .collect();
     */

    let mut values: Vec<f64> = generate_sine_wave(100);
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    info!("got {} datapoints", values.len());

    let t0 = Instant::now();

    let params = Params {
        input_sparsity: 0.3,
        input_activation: Activation::Identity,
        input_weight_scaling: 1.0,
        input_bias_scaling: 1.0,

        reservoir_size: 30,
        reservoir_fixed_in_degree_k: 2,
        reservoir_activation: Activation::Tanh,

        feedback_gain: 0.05,
        spectral_radius: 0.90,
        leaking_rate: 0.05,
        regularization_coeff: 0.1,
        washout_pct: 0.2,
        output_tanh: true,
        seed: None,
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
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        if i == TRAINING_WINDOW {
            plot_predictions.push((i as f64, last_prediction));
        }
        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(INPUT_DIM), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if i > TRAINING_WINDOW {
            plot_predictions.push((i as f64, last_prediction));
            m.row(0)
            //predicted_out.row(0)
        } else {
            train_predictions.push((i as f64, last_prediction));
            inputs.row(i)
        };

        rc.update_state(&input, &predicted_out);
    }

    //let targets = series.iter().enumerate().take(TRAINING_WINDOW * 2).map(|(i,
    // y)| (i as f64, *y as f64)).collect();
    plot(&plot_targets, &train_predictions, &plot_predictions, "img/plot.png");
}

fn plot(targets: &Series, train_preds: &Series, test_preds: &Series, filename: &str) {
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
