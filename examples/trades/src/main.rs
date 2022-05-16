#[macro_use]
extern crate log;

mod environments;
mod plot;

use std::{sync::Arc, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use environments::env_trades::EnvTrades;
use nalgebra::{DMatrix, Dim, Matrix};
use plot::{plot, GifRenderOptimizer, PlotGather};
use reservoir_computer::{
    ngrc, Activation, OptParamMapper, ReservoirComputer, TikhonovRegularization,
};
use sliding_features::{Echo, RoofingFilter, View};
use trade_aggregation::{aggregate_all_trades, load_trades_from_csv, TimeAggregator};

const INPUT_DIM: usize = 1;
const TRAIN_LEN: usize = 5000;
const TEST_WINDOW: usize = 600;
const NUM_GENS: usize = 100;

pub(crate) fn main() {
    pretty_env_logger::init();

    info!("loading sample data");

    let trades = load_trades_from_csv("data/Bitmex_XBTUSD_1M.csv");

    // Aggregate candles
    let mut agg = TimeAggregator::new(60);
    let candles = aggregate_all_trades(&trades, &mut agg);

    // pre-processing
    let mut values: Vec<f64> = Vec::with_capacity(TRAIN_LEN + TEST_WINDOW);
    let mut feature = RoofingFilter::new(Echo::new(), 100, 50);
    for c in candles.iter() {
        feature.update(c.weighted_price);
        values.push(feature.last() / 100.0);
    }
    info!("got {} datapoints", values.len());

    let values: DMatrix<f64> =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(INPUT_DIM), values);

    let rcs = vec![
        "ESN",
        "EuSN",
        "NG-RC",
        "ESN-Firefly",
        "ESN-RandomSearch",
        "EuSN-Firefly",
        "EuSN-RandomSearch",
    ];
    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Reservoir Computer")
        .items(&rcs)
        .default(0)
        .interact()
        .unwrap();
    match e {
        0 => {
            // TODO:
            /*
            let washout_pct = 0.0;
            let params = esn::Params {
                input_sparsity: 0.2,
                input_activation: Activation::Identity,
                input_weight_scaling: 0.2,
                reservoir_size: 500,
                reservoir_bias_scaling: 0.05,
                reservoir_sparsity: 0.02,
                reservoir_activation: Activation::Tanh,
                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate: 0.02,
                regularization_coeff: 0.02,
                washout_pct,
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
            let mut rc = esn::ESN::<1, 1, TikhonovRegularization>::new(params, regressor);

            let env = EnvTrades::new(Arc::new(values), TRAIN_LEN);
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/trades_esn.png",
                (2160, 2160),
            );
            */
        }
        1 => {
            // TODO:
            /*
            let params = eusn::Params {
                input_sparsity: 0.1,
                input_weight_scaling: 0.1,
                reservoir_size: 500,
                reservoir_weight_scaling: 0.1,
                reservoir_bias_scaling: 0.5,
                reservoir_activation: Activation::Tanh,
                initial_state_value: 0.0,
                seed: Some(0),
                washout_frac: 0.1,
                regularization_coeff: 0.01,
                epsilon: 0.01,
                gamma: 0.001,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            let mut rc = eusn::EulerStateNetwork::new(params, regressor);

            let env = EnvTrades::new(Arc::new(values), TRAIN_LEN);
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/trades_eusn.png",
                (2160, 2160),
            );
            */
        }
        2 => {
            let params = ngrc::Params {
                input_dim: 1,
                output_dim: 1,
                num_time_delay_taps: 10,
                num_samples_to_skip: 2,
                output_activation: Activation::Tanh,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.001,
            };
            let mut rc = ngrc::NextGenerationRC::new(params, regressor);
            let t0 = Instant::now();
            rc.train(&values.rows(0, TRAIN_LEN - 1), &values.rows(1, TRAIN_LEN));
            info!("NGRC training took {}ms", t0.elapsed().as_millis());

            let env = EnvTrades::new(Arc::new(values), TRAIN_LEN);
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/trades_ngrc.png",
                (2160, 2160),
            );
        }
        3 => {
            // TODO:
            /*
            let param_mapper = esn::ParamMapper {
                input_sparsity_range: (0.05, 0.25),
                input_activation: Activation::Identity,
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_bias_scaling_range: (0.0, 1.0),
                reservoir_sparsity_range: (0.01, 0.3),
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

            let env = EnvTrades::new(Arc::new(values), TRAIN_LEN);
            let env = Arc::new(env);

            let num_candidates = 96;
            let params = FireflyParams {
                gamma: 10.0,
                alpha: 0.005,
                step_size: 0.01,
                num_candidates,
            };
            let mut opt = FireflyOptimizer::<7>::new(params);

            let mut gif_render =
                GifRenderOptimizer::new("img/trades_esn_firefly.gif", (1080, 1080), num_candidates);
            for i in 0..NUM_GENS {
                let t0 = Instant::now();

                // TODO: choose lin reg
                let regressor = TikhonovRegularization {
                    regularization_coeff: 0.001,
                };
                opt.step::<esn::ESN<1, 1, TikhonovRegularization>, 1, 1, TikhonovRegularization>(
                    env.clone(),
                    &param_mapper,
                    regressor.clone(),
                );
                let params = param_mapper.map(opt.elite_params());
                let mut rc = esn::ESN::new(params, regressor);

                let mut p = PlotGather::default();
                env.evaluate(&mut rc, Some(&mut p));

                gif_render.update(
                    &p.plot_targets(),
                    &p.train_predictions(),
                    &p.test_predictions(),
                    opt.rmses(),
                    i,
                    opt.candidates(),
                );
                info!(
                    "generation {} took {}ms. best rmse: {}",
                    i,
                    t0.elapsed().as_millis(),
                    opt.best_rmse()
                );
            }
            */
        }
        4 => {
            // TODO:
            /*
            let seed = Some(0);

            let param_mapper = esn::ParamMapper {
                input_sparsity_range: (0.05, 0.25),
                input_activation: Activation::Identity,
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_bias_scaling_range: (0.0, 1.0),
                reservoir_sparsity_range: (0.01, 0.3),
                reservoir_activation: Activation::Tanh,
                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate_range: (0.0, 0.1),
                regularization_coeff_range: (0.0, 0.1),
                washout_pct: 0.0,
                output_activation: Activation::Identity,
                seed,
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
                readout_from_input_as_well: false,
            };

            let env = EnvTrades::new(Arc::new(values), TRAIN_LEN);
            let env = Arc::new(env);

            let num_candidates = 96;
            let mut opt = RandomSearch::<7>::new(seed, num_candidates);

            let mut gif_render = GifRenderOptimizer::new(
                "img/trades_esn_random_search.gif",
                (1080, 1080),
                num_candidates,
            );

            for i in 0..NUM_GENS {
                let t0 = Instant::now();

                // TODO: choose lin reg
                let regressor = TikhonovRegularization {
                    regularization_coeff: 0.001,
                };
                opt.step::<esn::ESN<1, 1, TikhonovRegularization>, 1, 1, TikhonovRegularization>(
                    env.clone(),
                    &param_mapper,
                    regressor.clone(),
                );

                let params = param_mapper.map(opt.elite_params());
                let mut rc = esn::ESN::<1, 1, TikhonovRegularization>::new(params, regressor);

                let mut p = PlotGather::default();
                env.evaluate(&mut rc, Some(&mut p));

                gif_render.update(
                    &p.plot_targets(),
                    &p.train_predictions(),
                    &p.test_predictions(),
                    opt.rmses(),
                    i,
                    opt.candidates(),
                );

                info!(
                    "generation {} took {}ms. best rmse: {}",
                    i,
                    t0.elapsed().as_millis(),
                    opt.best_rmse()
                );
            }
            */
        }
        5 => {
            // TODO:
            /*
            let param_mapper = eusn::ParamMapper {
                input_sparsity_range: (0.05, 0.25),
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_weight_scaling_range: (0.01, 1.0),
                reservoir_bias_scaling_range: (0.0, 1.0),
                reservoir_activation: Activation::Tanh,
                initial_state_value: values[0],
                seed: Some(0),
                washout_frac: 0.05,
                regularization_coeff: 0.1,
                epsilon_range: (0.0001, 0.01),
                gamma_range: (0.0001, 0.01),
            };

            let env = EnvTrades::new(Arc::new(values), TRAIN_LEN);
            let env = Arc::new(env);

            let num_candidates = 96;
            let params = FireflyParams {
                gamma: 10.0,
                alpha: 0.005,
                step_size: 0.01,
                num_candidates,
            };
            let mut opt = FireflyOptimizer::new(params);

            let mut gif_render = GifRenderOptimizer::new(
                "img/trades_eusn_firefly.gif",
                (1080, 1080),
                num_candidates,
            );

            for i in 0..NUM_GENS {
                let t0 = Instant::now();

                // TODO: choose lin reg
                let regressor = TikhonovRegularization {
                    regularization_coeff: 0.001,
                };
                opt.step::<eusn::EulerStateNetwork<1, 1, TikhonovRegularization>, 1, 1, TikhonovRegularization>(
                    env.clone(),
                    &param_mapper,
                    regressor.clone()
                );

                let params = param_mapper.map(opt.elite_params());
                let mut rc =
                    eusn::EulerStateNetwork::<1, 1, TikhonovRegularization>::new(params, regressor);

                let mut p = PlotGather::default();
                env.evaluate(&mut rc, Some(&mut p));

                gif_render.update(
                    &p.plot_targets(),
                    &p.train_predictions(),
                    &p.test_predictions(),
                    opt.rmses(),
                    i,
                    opt.candidates(),
                );

                info!(
                    "generation {} took {}ms. best rmse: {}",
                    i,
                    t0.elapsed().as_millis(),
                    opt.best_rmse()
                );
            }
            */
        }
        6 => {
            // TODO:
            /*
            let param_mapper = eusn::ParamMapper {
                input_sparsity_range: (0.05, 0.25),
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_weight_scaling_range: (0.01, 1.0),
                reservoir_bias_scaling_range: (0.0, 1.0),
                reservoir_activation: Activation::Tanh,
                initial_state_value: values[0],
                seed: Some(0),
                washout_frac: 0.05,
                regularization_coeff: 0.1,
                epsilon_range: (0.0001, 0.01),
                gamma_range: (0.0001, 0.01),
            };

            let env = EnvTrades::new(Arc::new(values), TRAIN_LEN);
            let env = Arc::new(env);

            let seed = Some(0);
            let num_candidates = 96;
            let mut opt = RandomSearch::new(seed, num_candidates);

            let mut gif_render = GifRenderOptimizer::new(
                "img/trades_eusn_random_search.gif",
                (1080, 1080),
                num_candidates,
            );

            for i in 0..NUM_GENS {
                let t0 = Instant::now();

                // TODO: choose lin reg
                let regressor = TikhonovRegularization {
                    regularization_coeff: 0.001,
                };
                opt.step::<eusn::EulerStateNetwork<1, 1, TikhonovRegularization>, 1, 1, TikhonovRegularization>(
                    env.clone(),
                    &param_mapper,
                    regressor.clone()
                );

                let params = param_mapper.map(opt.elite_params());
                let mut rc =
                    eusn::EulerStateNetwork::<1, 1, TikhonovRegularization>::new(params, regressor);

                let mut p = PlotGather::default();
                env.evaluate(&mut rc, Some(&mut p));

                gif_render.update(
                    &p.plot_targets(),
                    &p.train_predictions(),
                    &p.test_predictions(),
                    opt.rmses(),
                    i,
                    opt.candidates(),
                );

                info!(
                    "generation {} took {}ms. best rmse: {}",
                    i,
                    t0.elapsed().as_millis(),
                    opt.best_rmse()
                );
            }
            */
        }
        _ => panic!("invalid reservoir computer selection"),
    }
}
