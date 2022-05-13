#[macro_use]
extern crate log;

mod environments;
mod load_sample_data;
mod plot;

use std::time::Instant;

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice};
use plot::{GifRender, Series};
use reservoir_computer::{
    esn, Activation, LinReg, RCParams, ReservoirComputer, TikhonovRegularization,
};
use sliding_features::{Constant, Echo, Multiply, View, ALMA, VSCT};

// const SEED: Option<u64> = Some(0);
pub(crate) const TRAIN_LEN: usize = 10_000;
pub(crate) const VALIDATION_LEN: usize = 2_000;

pub(crate) fn main() {
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

    let rcs = vec!["ESN", "EuSN", "NG-RC", "ESN-Firefly"];
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

            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            let mut rc = esn::ESN::new(params, regressor);

            run_sliding::<esn::ESN<1, 1, TikhonovRegularization>, 7, TikhonovRegularization>(
                &mut rc,
                values,
                "img/trades_sliding_window_esn.gif",
            );
        }
        1 => {
            // TODO:
            /*
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
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            let mut rc = eusn::EulerStateNetwork::new(params, regressor);

            run_sliding::<
                eusn::EulerStateNetwork<1, 1, TikhonovRegularization>,
                7,
                TikhonovRegularization,
            >(&mut rc, values, "img/trades_sliding_window_eusn.gif");
            */
        }
        2 => {
            todo!()
        }
        3 => {
            // TODO:
            /*
            let param_mapper = esn::ParamMapper {
                input_sparsity_range: (0.15, 0.25),
                input_activation: Activation::Identity,
                input_weight_scaling_range: (0.15, 0.25),
                reservoir_size_range: (200.0, 700.0),
                reservoir_bias_scaling_range: (0.0, 0.1),
                reservoir_sparsity_range: (0.01, 0.03),
                reservoir_activation: Activation::Tanh,
                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate_range: (0.0, 0.1),
                regularization_coeff_range: (0.0, 0.1),
                washout_pct: 0.0,
                output_activation: Activation::Identity,
                seed: Some(0),
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
                readout_from_input_as_well: false,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            run_sliding_opt_firefly::<
                esn::ESN<1, 1, TikhonovRegularization>,
                7,
                TikhonovRegularization,
            >(
                values, "img/trades_sliding_window_esn_firefly.gif", &param_mapper, regressor
            );
            */
        }
        _ => panic!("invalid reservoir computer selection"),
    }
}

/*
fn run_sliding_opt_firefly<RC, const N: usize, R>(
    values: Vec<f64>,
    filename: &str,
    param_mapper: &RC::ParamMapper,
    regressor: R,
) where
    RC: ReservoirComputer<1, 1, N, R> + Send + Sync + 'static,
    R: LinReg + Send + Sync + 'static,
{
    let t0 = Instant::now();

    let values: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
        Matrix::from_vec_generic(Dim::from_usize(1), Dim::from_usize(values.len()), values);

    let num_candidates = 96;
    let params = FireflyParams {
        gamma: 10.0,
        alpha: 0.005,
        step_size: 0.01,
        num_candidates,
    };
    let mut opt = FireflyOptimizer::<N>::new(params);

    let mut gif_render = GifRenderOptimizer::new(filename, (1080, 1080), num_candidates);
    // TODO: iterate over all data
    for j in (TRAIN_LEN + VALIDATION_LEN + 1)..100_000 {
        if j % 100 == 0 {
            info!("step @ {}", j);
            let t1 = Instant::now();

            let env = EnvTrades::new(
                Arc::new(values.columns(j - TRAIN_LEN - VALIDATION_LEN, j).into()),
                TRAIN_LEN,
            );
            let env = Arc::new(env);

            opt.step::<RC, 1, 1, R>(env.clone(), &param_mapper, regressor.clone());
            let params = param_mapper.map(opt.elite_params());
            let mut rc = RC::new(params, regressor.clone());
            rc.train(
                &values.columns(j - TRAIN_LEN - VALIDATION_LEN - 1, j - VALIDATION_LEN - 1),
                &values.columns(j - TRAIN_LEN - VALIDATION_LEN, j - VALIDATION_LEN),
            );

            let state = Matrix::from_element_generic(
                Dim::from_usize(rc.params().reservoir_size()),
                Dim::from_usize(1),
                values[j - TRAIN_LEN - VALIDATION_LEN - 1],
            );
            rc.set_state(state);

            let (plot_targets, train_preds, test_preds) =
                gather_plot_data(&values.columns(j - TRAIN_LEN - VALIDATION_LEN, j), &mut rc);
            gif_render.update(
                &plot_targets,
                &train_preds,
                &test_preds,
                opt.rmses(),
                j,
                opt.candidates(),
            );

            info!("step took {}s", t1.elapsed().as_secs());
        }
    }

    info!("took {}s", t0.elapsed().as_secs());
}
*/

fn run_sliding<RC, const N: usize, R>(rc: &mut RC, values: Vec<f64>, filename: &str)
where
    RC: ReservoirComputer<N, R>,
    R: LinReg,
{
    let t0 = Instant::now();

    let values: DMatrix<f64> =
        Matrix::from_vec_generic(Dim::from_usize(1), Dim::from_usize(values.len()), values);

    let mut gif_render = GifRender::new(filename, (1080, 1080));
    for j in (TRAIN_LEN + VALIDATION_LEN + 1)..100_000 {
        if j % 100 == 0 {
            info!("step @ {}", j);
            let t1 = Instant::now();

            rc.train(
                &values.columns(j - TRAIN_LEN - VALIDATION_LEN - 1, j - 1),
                &values.columns(j - TRAIN_LEN - VALIDATION_LEN, j),
            );

            let state = Matrix::from_element_generic(
                Dim::from_usize(rc.params().reservoir_size()),
                Dim::from_usize(1),
                values[j - TRAIN_LEN - VALIDATION_LEN - 1],
            );
            rc.set_state(state);

            let (plot_targets, train_preds, test_preds) =
                gather_plot_data(&values.columns(j - TRAIN_LEN - VALIDATION_LEN, j), rc);
            gif_render.update(&plot_targets, &train_preds, &test_preds);

            info!("step took {}s", t1.elapsed().as_secs());
        }
    }

    info!("took {}s", t0.elapsed().as_secs());
}

fn gather_plot_data<'a, RC, const N: usize, R>(
    values: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    rc: &mut RC,
) -> (Series, Series, Series)
where
    RC: ReservoirComputer<N, R>,
    R: LinReg,
{
    let mut plot_targets = Vec::with_capacity(values.len());
    let mut train_preds = vec![];
    let mut test_preds = vec![];
    for j in 1..values.ncols() {
        plot_targets.push((j as f64, *values.column(j).get(0).unwrap()));

        let predicted_out = rc.readout();
        let last_prediction = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: DMatrix<f64> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if j > TRAIN_LEN {
            test_preds.push((j as f64, last_prediction));
            m.column(0)
        } else {
            train_preds.push((j as f64, last_prediction));
            values.column(j - 1)
        };

        rc.update_state(&input, &predicted_out);
    }

    (plot_targets, train_preds, test_preds)
}
