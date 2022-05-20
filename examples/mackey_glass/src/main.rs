#[macro_use]
extern crate log;

use std::{collections::VecDeque, sync::Arc, time::Instant};

use classic_rcs::{ESNConstructor, EUSNConstructor, Params, RC};
use dialoguer::{theme::ColorfulTheme, Select};
use lin_reg::TikhonovRegularization;
use nalgebra::{DMatrix, Dim, Matrix};
use nanorand::{Rng, WyRand};
use rc_plot::{plot, PlotGather};

use common::{environments::EnvMackeyGlass, Activation, ReservoirComputer};
use next_generation_rcs::{NGRCConstructor, NextGenerationRC, Params as NGRCParams};

const TRAIN_LEN: usize = 5000;
const TEST_LEN: usize = 1000;
const SEED: Option<u64> = Some(0);
const NUM_GENS: usize = 100;

pub(crate) fn main() {
    let total_len = TRAIN_LEN + TEST_LEN;
    let values = mackey_glass_series(total_len, 30, SEED);

    let values: DMatrix<f64> =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(1), values);

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
            let params = Params {
                input_activation: Activation::Identity,
                reservoir_size: 200,
                reservoir_activation: Activation::Tanh,
                leaking_rate: 0.1,
                washout_pct: 0.1,
                output_activation: Activation::Identity,
                seed: SEED,
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
            };

            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.1,
            };
            let reservoir_size = 200;
            let spectral_radius = 0.9;
            let reservoir_sparsity = 0.1;
            let reservoir_bias_scaling = 0.4;
            let input_sparsity = 0.1;
            let input_weight_scaling = 0.5;
            let esn_constructor = ESNConstructor::new(
                SEED,
                reservoir_size,
                spectral_radius,
                reservoir_sparsity,
                reservoir_bias_scaling,
                input_sparsity,
                input_weight_scaling,
            );
            let mut rc = RC::<TikhonovRegularization>::new(params, regressor, esn_constructor);

            let t0 = Instant::now();
            rc.train(&values.rows(0, TRAIN_LEN - 1), &values.rows(1, TRAIN_LEN));
            info!("ESN training done in {}ms", t0.elapsed().as_millis());

            let env = EnvMackeyGlass::new(Arc::new(values), TRAIN_LEN);
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/mackey_glass_esn.png",
                (3840, 1080),
            );
        }
        1 => {
            let params = Params {
                reservoir_size: 300,
                reservoir_activation: Activation::Tanh,
                initial_state_value: values[0],
                seed: SEED,
                input_activation: Activation::Identity,
                leaking_rate: 0.1,
                washout_pct: 0.1,
                output_activation: Activation::Identity,
                state_update_noise_frac: 0.001,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.1,
            };
            let reservoir_size = 300;
            let reservoir_weight_scaling = 1.0;
            let reservoir_bias_scaling = 1.0;
            let input_sparsity = 0.1;
            let input_weight_scaling = 1.0;
            let gamma = 0.05;
            let eusn_constructor = EUSNConstructor::new(
                SEED,
                reservoir_size,
                reservoir_weight_scaling,
                reservoir_bias_scaling,
                input_sparsity,
                input_weight_scaling,
                gamma,
            );
            let mut rc = RC::new(params, regressor, eusn_constructor);

            let t0 = Instant::now();
            rc.train(&values.rows(0, TRAIN_LEN - 1), &values.rows(1, TRAIN_LEN));
            info!("ESN training done in {}ms", t0.elapsed().as_millis());

            let env = EnvMackeyGlass::new(Arc::new(values), TRAIN_LEN);
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/mackey_glass_eusn.png",
                (3840, 1080),
            );
        }
        2 => {
            let params = NGRCParams {
                input_dim: 1,
                output_dim: 1,
                num_time_delay_taps: 11,
                num_samples_to_skip: 2,
                output_activation: Activation::Identity,
            };
            // TODO: choose lin reg
            let regressor = TikhonovRegularization {
                regularization_coeff: 95.0,
            };
            let ngrc_constructor = NGRCConstructor::default();
            let mut rc = NextGenerationRC::new(params, regressor, ngrc_constructor);
            let t0 = Instant::now();
            rc.train(&values.rows(0, TRAIN_LEN - 1), &values.rows(1, TRAIN_LEN));
            info!("NGRC training took {}ms", t0.elapsed().as_millis());

            let env = EnvMackeyGlass::new(Arc::new(values), TRAIN_LEN);
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/mackey_glass_ngrc.png",
                (3840, 1080),
            );
        }
        3 => {
            // TODO:
            /*
            let param_mapper = esn::ParamMapper {
                input_sparsity_range: (0.05, 0.2),
                input_activation: Activation::Identity,
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_bias_scaling_range: (0.0, 0.1),
                reservoir_sparsity_range: (0.01, 0.2),
                reservoir_activation: Activation::Tanh,
                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate_range: (0.0, 0.2),
                regularization_coeff_range: (0.0, 0.2),
                washout_pct: 0.0,
                output_activation: Activation::Identity,
                seed: Some(0),
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
                readout_from_input_as_well: false,
            };

            let env = EnvMackeyGlass::new(Arc::new(values), TRAIN_LEN);
            let env = Arc::new(env);

            let num_candidates = 96;
            let params = FireflyParams {
                gamma: 10.0,
                alpha: 0.005,
                step_size: 0.01,
                num_candidates,
            };
            let mut opt = FireflyOptimizer::<7>::new(params);

            let mut gif_render = GifRenderOptimizer::new(
                "img/mackey_glass_esn_firefly.gif",
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
                    regressor,
                );
                let params = param_mapper.map(opt.elite_params());
                // TODO: choose lin reg
                let regressor = TikhonovRegularization {
                    regularization_coeff: 0.001,
                };
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
                input_sparsity_range: (0.05, 0.2),
                input_activation: Activation::Identity,
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_bias_scaling_range: (0.0, 0.1),
                reservoir_sparsity_range: (0.01, 0.2),
                reservoir_activation: Activation::Tanh,
                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate_range: (0.0, 0.2),
                regularization_coeff_range: (0.0, 0.2),
                washout_pct: 0.0,
                output_activation: Activation::Identity,
                seed,
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
                readout_from_input_as_well: false,
            };

            let env = EnvMackeyGlass::new(Arc::new(values), TRAIN_LEN);
            let env = Arc::new(env);

            let num_candidates = 23;
            let mut opt = RandomSearch::<7>::new(seed, num_candidates);

            let mut gif_render = GifRenderOptimizer::new(
                "img/mackey_glass_esn_random_search.gif",
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
                input_sparsity_range: (0.05, 0.2),
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_weight_scaling_range: (0.01, 0.2),
                reservoir_bias_scaling_range: (0.0, 1.0),
                reservoir_activation: Activation::Tanh,
                initial_state_value: values[0],
                seed: Some(0),
                washout_frac: 0.05,
                regularization_coeff: 0.1,
                epsilon_range: (0.0001, 0.01),
                gamma_range: (0.0001, 0.01),
            };

            let env = EnvMackeyGlass::new(Arc::new(values), TRAIN_LEN);
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
                "img/mackey_glass_eusn_firefly.gif",
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
                input_sparsity_range: (0.05, 0.2),
                input_weight_scaling_range: (0.1, 1.0),
                reservoir_size_range: (200.0, 800.0),
                reservoir_weight_scaling_range: (0.01, 0.2),
                reservoir_bias_scaling_range: (0.0, 1.0),
                reservoir_activation: Activation::Tanh,
                initial_state_value: values[0],
                seed: Some(0),
                washout_frac: 0.05,
                regularization_coeff: 0.1,
                epsilon_range: (0.0001, 0.01),
                gamma_range: (0.0001, 0.01),
            };

            let env = EnvMackeyGlass::new(Arc::new(values), TRAIN_LEN);
            let env = Arc::new(env);

            let seed = Some(0);
            let num_candidates = 96;
            let mut opt = RandomSearch::new(seed, num_candidates);

            let mut gif_render = GifRenderOptimizer::new(
                "img/mackey_glass_eusn_random_search.gif",
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
                    regressor.clone(),
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
    };
}

/// Mackey glass series generation
/// # Parameters:
/// sample_len: the number of datapoints to generate
/// tau: roughly correlates to the chaotic and complex behaviour of the sequence
/// seed: An optional seed for the random number generator
pub(crate) fn mackey_glass_series(sample_len: usize, tau: usize, seed: Option<u64>) -> Vec<f64> {
    let delta_t = 10;
    let mut timeseries = 1.2;
    let history_len = tau * delta_t;

    let mut rng = match seed {
        Some(seed) => WyRand::new_seed(seed),
        None => WyRand::new(),
    };
    let mut history = VecDeque::with_capacity(history_len);
    for _i in 0..history_len {
        let val = 1.2 + 0.2 * (rng.generate::<f64>() - 0.5);
        history.push_back(val);
    }

    let mut inp = vec![0.0; sample_len];

    for timestep in 0..sample_len {
        for _ in 0..delta_t {
            let x_tau = history.pop_front().unwrap();
            history.push_back(timeseries);
            let last_hist = history[history.len() - 1];
            timeseries = last_hist
                + (0.2 * x_tau / (1.0 + x_tau.powi(10)) - 0.1 * last_hist) / delta_t as f64;
        }
        inp[timestep] = timeseries;
    }
    // apply tanh nonlinearity
    inp.iter_mut().for_each(|v| *v = v.tanh());

    inp
}
