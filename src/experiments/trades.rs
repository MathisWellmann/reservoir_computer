use std::{sync::Arc, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Dim, Matrix};
use sliding_features::{Echo, View, ALMA};

use crate::{
    activation::Activation,
    environments::{env_trades::EnvTrades, PlotGather},
    load_sample_data,
    optimizers::{
        opt_firefly::{FireflyOptimizer, FireflyParams},
        opt_random_search::RandomSearch,
    },
    plot::{plot, GifRenderOptimizer},
    reservoir_computers::{esn, eusn, ngrc, OptParamMapper, ReservoirComputer},
    utils::scale,
    OptEnvironment, SingleDimIo,
};

const INPUT_DIM: usize = 1;
const OUTPUT_DIM: usize = 1;
const TRAINING_WINDOW: usize = 10_000;
const TEST_WINDOW: usize = 5000;
const NUM_GENS: usize = 100;

pub(crate) fn start() {
    info!("loading sample data");

    let series: Vec<f64> = load_sample_data::load_sample_data()
        .iter()
        .take(TRAINING_WINDOW + TEST_WINDOW)
        .cloned()
        .collect();
    let mut series_min = series[0];
    let mut series_max = series[0];
    for s in &series {
        if *s < series_min {
            series_min = *s;
        }
        if *s > series_max {
            series_max = *s;
        }
    }
    info!("series_min: {}, series_max: {}", series_min, series_max);

    let mut alma = ALMA::new(Echo::new(), 100);
    let mut values: Vec<f64> = Vec::with_capacity(series.len());
    for s in &series {
        let val = scale(series_min, series_max, -1.0, 1.0, *s);
        alma.update(val);
        values.push(alma.last());
    }
    info!("got {} datapoints", values.len());

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
    let inputs: SingleDimIo = Matrix::from_vec_generic(
        Dim::from_usize(INPUT_DIM),
        Dim::from_usize(values.len() - 1),
        values.iter().take(values.len() - 1).cloned().collect(),
    );
    let targets: SingleDimIo = Matrix::from_vec_generic(
        Dim::from_usize(INPUT_DIM),
        Dim::from_usize(values.len() - 1),
        values.iter().skip(1).cloned().collect(),
    );

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
            let mut rc = esn::ESN::new(params);

            let env = EnvTrades::new(
                Arc::new(train_inputs),
                Arc::new(train_targets),
                Arc::new(inputs),
                Arc::new(targets),
            );
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/trades_esn.png",
                (2160, 2160),
            );
        }
        1 => {
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
            let mut rc = eusn::EulerStateNetwork::new(params);

            let env = EnvTrades::new(
                Arc::new(train_inputs),
                Arc::new(train_targets),
                Arc::new(inputs),
                Arc::new(targets),
            );
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/trades_eusn.png",
                (2160, 2160),
            );
        }
        2 => {
            let params = ngrc::Params {
                num_time_delay_taps: 15,
                num_samples_to_skip: 150,
                regularization_coeff: 0.01,
                output_activation: Activation::Identity,
            };
            let mut rc = ngrc::NextGenerationRC::new(params);
            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("NGRC training took {}ms", t0.elapsed().as_millis());

            let env = EnvTrades::new(
                Arc::new(train_inputs),
                Arc::new(train_targets),
                Arc::new(inputs),
                Arc::new(targets),
            );
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

            let env = EnvTrades::new(
                Arc::new(train_inputs.clone()),
                Arc::new(train_targets.clone()),
                Arc::new(inputs.clone()),
                Arc::new(targets),
            );
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

                opt.step::<esn::ESN<1, 1>, 1, 1>(env.clone(), &param_mapper);
                let params = param_mapper.map(opt.elite_params());
                let mut rc = esn::ESN::new(params);

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
        }
        4 => {
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

            let env = EnvTrades::new(
                Arc::new(train_inputs.clone()),
                Arc::new(train_targets.clone()),
                Arc::new(inputs.clone()),
                Arc::new(targets),
            );
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

                opt.step::<esn::ESN<1, 1>, 1, 1>(env.clone(), &param_mapper);

                let params = param_mapper.map(opt.elite_params());
                let mut rc = esn::ESN::<1, 1>::new(params);

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
        }
        5 => {
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

            let env = EnvTrades::new(
                Arc::new(train_inputs.clone()),
                Arc::new(train_targets.clone()),
                Arc::new(inputs.clone()),
                Arc::new(targets),
            );
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

                opt.step::<eusn::EulerStateNetwork<1, 1>, 1, 1>(env.clone(), &param_mapper);

                let params = param_mapper.map(opt.elite_params());
                let mut rc = eusn::EulerStateNetwork::<1, 1>::new(params);

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
        }
        6 => {
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

            let env = EnvTrades::new(
                Arc::new(train_inputs.clone()),
                Arc::new(train_targets.clone()),
                Arc::new(inputs.clone()),
                Arc::new(targets),
            );
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

                opt.step::<eusn::EulerStateNetwork<1, 1>, 1, 1>(env.clone(), &param_mapper);

                let params = param_mapper.map(opt.elite_params());
                let mut rc = eusn::EulerStateNetwork::<1, 1>::new(params);

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
        }
        _ => panic!("invalid reservoir computer selection"),
    }
}
