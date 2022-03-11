#[macro_use]
extern crate log;

//mod reservoir;
mod errors;
mod load_sample_data;

use std::time::Instant;

use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, SymmetricEigen, VecStorage};
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
    let rets: Vec<f32> =
        series.iter().skip(1).zip(series.iter()).map(|(a, b)| (a / b).ln()).collect();
    info!("got {} datapoints", series.len());

    const N: usize = 300; // reservoir size of hidden neurons
    const M: usize = 1; // input and output dimension
    const TRAINING_WINDOW: usize = 10_000;

    // For each node in the reservoir, select k nodesn the network without
    // replacement and use those as inputs to the current node.
    let fixed_in_degree_k = 50;
    let input_sparsity = 0.3;
    let input_scaling = 0.1;
    // The spectral radius determines how fast the influence of an input
    // dies out in a reservoir with time, and how stable the reservoir activations
    // are. The spectral radius should be greater in tasks requiring longer
    // memory of the input.
    let spectral_radius = 0.95;
    // The leaking rate a can be regarded as the speed of the reservoir update
    // dynamics discretized in time. This can be adapted online to deal with
    // time wrapping of the signals. Set the leaking rate to match the speed of
    // the dynamics of input / target This sort of acts as a EMA smoothing
    // filter, so other MAs could be used such as ALMA
    let alpha = 0.005;
    let train_frac = 0.7;
    let split_idx = (series.len() as f64 * train_frac) as usize;

    let t0 = Instant::now();

    //let mut rng = WyRand::new_seed(0);
    let mut rng = WyRand::new();
    let mut weights: Vec<Vec<f32>> = vec![vec![0.0; N]; N];
    for j in 0..weights.len() {
        // Choose random input node
        let i = rng.generate_range(0..N);
        weights[i][j] = rng.generate::<f32>() * 2.0 - 1.0;
    }
    let mut reservoir: DMatrix<f32> = DMatrix::from_vec_generic(
        Dim::from_usize(N),
        Dim::from_usize(N),
        weights.iter().cloned().flatten().collect(),
    );
    debug!("reservoir: {}", reservoir);
    let eigen = SymmetricEigen::new(reservoir.clone());
    let spec_rad = eigen.eigenvalues.abs().max();
    reservoir *= (1.0 / spec_rad) * spectral_radius;

    let input_matrix: Matrix<f32, Dynamic, Const<M>, VecStorage<f32, Dynamic, Const<M>>> =
        Matrix::from_fn_generic(Dim::from_usize(N), Dim::from_usize(M), |_, _| {
            if rng.generate::<f32>() < input_sparsity {
                rng.generate::<f32>() * input_scaling
            } else {
                0.0
            }
        });
    let mut readout_matrix: Matrix<f32, Const<M>, Dynamic, VecStorage<f32, Const<M>, Dynamic>> =
        Matrix::from_fn_generic(Dim::from_usize(M), Dim::from_usize(N), |_, _| {
            rng.generate::<f32>() * 2.0 - 1.0
        });
    let mut state: Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>> =
        Matrix::from_fn_generic(Dim::from_usize(N), Dim::from_usize(1), |_, _| {
            rng.generate::<f32>() * 2.0 - 1.0
        });
    debug!(
        "input_matrix: {}\nreservoir: {}\nreadout_matrix: {}\nstate: {}",
        input_matrix, reservoir, readout_matrix, state
    );

    let mut rmse: f32 = 0.0;
    let mut step_wise_design: DMatrix<f32> =
        DMatrix::from_fn_generic(Dim::from_usize(N), Dim::from_usize(TRAINING_WINDOW), |_, _| 0.0);
    let mut step_wise_predictions: Matrix<
        f32,
        Const<M>,
        Dynamic,
        VecStorage<f32, Const<M>, Dynamic>,
    > = Matrix::from_fn_generic(Dim::from_usize(M), Dim::from_usize(TRAINING_WINDOW), |_, _| 0.0);
    let step_wise_target: Matrix<f32, Const<M>, Dynamic, VecStorage<f32, Const<M>, Dynamic>> =
        Matrix::from_vec_generic(
            Dim::from_usize(M),
            Dim::from_usize(TRAINING_WINDOW),
            rets.iter().take(TRAINING_WINDOW).cloned().collect(),
        );
    for (j, val) in rets.iter().enumerate().take(TRAINING_WINDOW) {
        let predicted_out = &readout_matrix * &state;
        step_wise_predictions.set_column(j, &predicted_out);

        step_wise_design.set_column(j, &state);

        let a = (1.0 - alpha) * &state;
        let mut b = &reservoir * &state + &input_matrix * *val;
        b.iter_mut().for_each(|v| *v = v.tanh());
        state = a + alpha * b;

        rmse += (predicted_out.get(0).unwrap() - *val).powi(2);
    }
    info!("rmse: {}", rmse);

    // compute optimal readout matrix
    let design_t = step_wise_design.transpose();
    // Use regularizaion whenever there is a danger of overfitting or feedback
    // instability
    let regularization_coeff: f32 = 0.2;
    let identity_m: DMatrix<f32> =
        DMatrix::from_diagonal_element_generic(Dim::from_usize(N), Dim::from_usize(N), 1.0);
    let b = step_wise_design * &design_t + regularization_coeff * identity_m;
    let b = b.transpose();
    readout_matrix = step_wise_target * &design_t * b;

    let mut targets: Series = Vec::with_capacity(1_000_000);
    let mut predictions: Series = Vec::with_capacity(1_000_000);

    let mut curr_val = series[0];
    let mut curr_pred = series[0];
    info!("curr_pred: {}", curr_pred);
    targets.push((0_f32, curr_val));

    let mut train_predictions: Series = Vec::with_capacity(TRAINING_WINDOW);

    for (i, val) in rets.iter().enumerate().skip(1).take(TRAINING_WINDOW * 2) {
        curr_val *= val.exp();
        targets.push((i as f32, curr_val));

        let predicted_out = &readout_matrix * &state;
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

        let a = (1.0 - alpha) * &state;
        let mut b = &reservoir * &state + &input_matrix * val;
        b.iter_mut().for_each(|v| *v = v.tanh());
        state = a + alpha * b;
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
