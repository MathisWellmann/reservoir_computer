#[macro_use]
extern crate log;

mod errors;
mod load_sample_data;
mod reservoir;

use std::time::Instant;

use plotters::prelude::*;

use crate::reservoir::ReservoirComputer;

type Series = Vec<(f32, f32)>;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    info!("loading sample data");
    let series = load_sample_data::load_sample_data();
    let rets: Vec<f32> =
        series.iter().skip(1).zip(series.iter()).map(|(a, b)| (a / b).ln()).collect();
    info!("got {} datapoints", series.len());

    const TRAINING_WINDOW: usize = 10_000;

    let t0 = Instant::now();

    let leaking_rate = 0.01;
    let mut rc = ReservoirComputer::new(20, 5, 0.3, 0.0, 0.0, 0.95, leaking_rate, Some(1));
    rc.train(&rets.iter().take(TRAINING_WINDOW).cloned().collect::<Vec<f32>>());

    let mut targets: Series = Vec::with_capacity(1_000_000);
    let mut predictions: Series = Vec::with_capacity(1_000_000);

    let mut curr_val = series[0];
    let mut curr_pred = series[0];
    info!("curr_pred: {}", curr_pred);
    targets.push((0_f32, curr_val));

    let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);

    let mut state = rc.state();
    for (i, val) in rets.iter().enumerate().skip(1).take(TRAINING_WINDOW * 2) {
        curr_val *= val.exp();
        targets.push((i as f32, curr_val));

        let predicted_out = rc.readout_matrix() * &state;
        let pred = predicted_out.get(0).unwrap();
        curr_pred *= pred.exp();

        if i == TRAINING_WINDOW {
            curr_pred = series[TRAINING_WINDOW];
            predictions.push((i as f32, curr_pred));
        }
        // TO begin forecasting, replace target input with it's own prediction
        let val: f32 = if i > TRAINING_WINDOW {
            predictions.push((i as f32, curr_pred));
            *pred
        } else {
            train_predictions.push((i as f32, curr_pred));
            *val
        };

        let a = (1.0 - leaking_rate) * &state;
        let mut b = rc.reservoir() * &state + rc.input_matrix() * val;
        b.iter_mut().for_each(|v| *v = v.tanh());
        state = a + leaking_rate * b;
    }

    info!("t_diff: {}ms", t0.elapsed().as_millis());

    //let targets = series.iter().enumerate().take(TRAINING_WINDOW * 2).map(|(i,
    // y)| (i as f32, *y as f32)).collect();
    plot(&targets, &train_predictions, &predictions, "img/plot.png");
}

fn plot(targets: &Series, train_preds: &Series, test_preds: &Series, filename: &str) {
    let ts_min = targets[0].0;
    let ts_max = targets[targets.len() - 1].0;
    let mut target_min: f32 = targets[0].1;
    let mut target_max: f32 = targets[targets.len() - 1].1;
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
