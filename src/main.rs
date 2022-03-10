#[macro_use]
extern crate log;

//mod reservoir;
mod errors;
mod load_sample_data;

use std::time::Instant;

use nalgebra::{Const, Dim, OMatrix, SymmetricEigen};
use nanorand::{Rng, WyRand};
use plotters::prelude::*;

type Series = Vec<(f32, f32)>;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    info!("loading sample data");
    let series = load_sample_data::load_sample_data();
    info!("got {} datapoints", series.len());

    const N: usize = 50;
    const M: usize = 1;
    const TRAINING_WINDOW: usize = 1000;

    let sparsity = 0.1;
    let alpha = 0.1;
    let train_frac = 0.7;
    let split_idx = (series.len() as f64 * train_frac) as usize;

    let t0 = Instant::now();

    let mut rng = WyRand::new();
    let mut reservoir: OMatrix<f32, Const<N>, Const<N>> =
        OMatrix::from_fn_generic(Dim::from_usize(N), Dim::from_usize(N), |_, _| {
            if rng.generate::<f32>() < sparsity {
                rng.generate::<f32>() * 2.0 - 1.0
            } else {
                0.0
            }
        });
    let eigen = SymmetricEigen::new(reservoir);
    let spectral_radius = eigen.eigenvalues.abs().max();
    reservoir /= spectral_radius;

    let input_matrix: OMatrix<f32, Const<N>, Const<M>> =
        OMatrix::from_fn_generic(Dim::from_usize(N), Dim::from_usize(M), |_, _| rng.generate());
    let readout_matrix: OMatrix<f32, Const<M>, Const<N>> =
        OMatrix::from_fn_generic(Dim::from_usize(M), Dim::from_usize(N), |_, _| rng.generate());
    let mut state: OMatrix<f32, Const<N>, Const<1>> =
        OMatrix::from_fn_generic(Dim::from_usize(N), Dim::from_usize(1), |_, _| 1.0);

    let mut rmse: f32 = 0.0;
    let mut step_wise_state: OMatrix<f32, Const<N>, Const<TRAINING_WINDOW>> =
        OMatrix::from_fn_generic(Dim::from_usize(N), Dim::from_usize(TRAINING_WINDOW), |_, _| 0.0);
    let mut step_wise_predictions: OMatrix<f32, Const<M>, Const<TRAINING_WINDOW>> =
        OMatrix::from_fn_generic(Dim::from_usize(M), Dim::from_usize(TRAINING_WINDOW), |_, _| 0.0);
    for (j, val) in series.iter().enumerate().take(TRAINING_WINDOW) {
        let predicted_out = readout_matrix * state;
        step_wise_predictions.set_column(j, &predicted_out);

        step_wise_state.set_column(j, &state);

        let mut b = alpha * (reservoir * state + input_matrix * *val);
        b.iter_mut().for_each(|v| *v = v.tanh());
        let next_state = (1.0 - alpha) * state + b;
        state = next_state;

        rmse += (predicted_out.get(0).unwrap() - *val).powi(2);
    }
    info!("rmse: {}", rmse);

    let mut targets: Series = Vec::with_capacity(1_000_000);
    let mut predictions: Series = Vec::with_capacity(1_000_000);
    for (i, val) in series.iter().enumerate() {
        targets.push((i as f32, *val));

        let predicted_out = readout_matrix * state;
        debug!("predicted_out: {:?}", predicted_out);
        predictions.push((i as f32, *predicted_out.get(0).unwrap()));
    }

    info!("t_diff: {}ms", t0.elapsed().as_millis());

    plot(&targets, &predictions, "img/plot.png");
}

fn plot(targets: &Series, predictions: &Series, filename: &str) {
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
    root_area.fill(&BLACK).unwrap();
    let root_area = root_area.titled(filename, ("sans-serif", 20).into_font()).unwrap();

    let areas = root_area.split_evenly((1, 1));

    let mut cc0 = ChartBuilder::on(&areas[0])
        .margin(5)
        .set_all_label_area_size(50)
        .caption("price", ("sans-serif", 30).into_font().with_color(&WHITE))
        .build_cartesian_2d(0_f32..1_000_000_f32, target_min..target_max)
        .unwrap();
    cc0.configure_mesh()
        .x_labels(20)
        .y_labels(20)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .draw()
        .unwrap();

    cc0.draw_series(LineSeries::new(targets.clone(), &WHITE))
        .unwrap()
        .label("targets")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &WHITE));
    cc0.draw_series(LineSeries::new(predictions.clone(), &RED))
        .unwrap()
        .label("predictions")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    cc0.configure_series_labels().border_style(&BLACK).draw().unwrap();
}
