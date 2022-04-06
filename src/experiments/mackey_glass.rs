use std::{collections::VecDeque, sync::Arc, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Dim, Matrix};
use nanorand::{Rng, WyRand};

use crate::{
    activation::Activation,
    environments::{env_mackey_glass::EnvMackeyGlass, PlotGather},
    plot::plot,
    reservoir_computers::{esn, eusn, ReservoirComputer},
    OptEnvironment,
};

const TRAIN_LEN: usize = 5000;
const TEST_LEN: usize = 1000;
const SEED: Option<u64> = Some(0);

pub(crate) fn start() {
    let total_len = TRAIN_LEN + TEST_LEN;
    let values = mackey_glass_series(total_len, 30, SEED);

    let train_inputs = Matrix::from_vec_generic(
        Dim::from_usize(1),
        Dim::from_usize(TRAIN_LEN),
        values.iter().take(TRAIN_LEN).cloned().collect::<Vec<f64>>(),
    );
    let train_targets = Matrix::from_vec_generic(
        Dim::from_usize(1),
        Dim::from_usize(TRAIN_LEN),
        values.iter().skip(1).take(TRAIN_LEN).cloned().collect::<Vec<f64>>(),
    );
    let inputs = Matrix::from_vec_generic(
        Dim::from_usize(1),
        Dim::from_usize(values.len() - 1),
        values.iter().take(values.len() - 1).cloned().collect::<Vec<f64>>(),
    );
    let targets = Matrix::from_vec_generic(
        Dim::from_usize(1),
        Dim::from_usize(values.len() - 1),
        values.iter().skip(1).cloned().collect::<Vec<f64>>(),
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
            let params = esn::Params {
                input_sparsity: 0.1,
                input_activation: Activation::Identity,
                input_weight_scaling: 0.5,
                reservoir_bias_scaling: 0.4,

                reservoir_size: 200,
                reservoir_sparsity: 0.1,
                reservoir_activation: Activation::Tanh,

                feedback_gain: 0.0,
                spectral_radius: 0.9,
                leaking_rate: 0.1,
                regularization_coeff: 0.1,
                washout_pct: 0.1,
                output_activation: Activation::Identity,
                seed: SEED,
                state_update_noise_frac: 0.001,
                initial_state_value: values[0],
                readout_from_input_as_well: false,
            };

            let mut rc = esn::ESN::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("ESN training done in {}ms", t0.elapsed().as_millis());

            let env = EnvMackeyGlass::new(
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
                "img/mackey_glass_esn.png",
                (3840, 1080),
            );
        }
        1 => {
            let params = eusn::Params {
                input_sparsity: 0.1,
                input_weight_scaling: 1.0,
                reservoir_size: 300,
                reservoir_weight_scaling: 0.1,
                reservoir_bias_scaling: 1.0,
                reservoir_activation: Activation::Tanh,
                initial_state_value: values[0],
                seed: SEED,
                washout_frac: 0.1,
                regularization_coeff: 0.1,
                epsilon: 0.008,
                gamma: 0.05,
            };
            let mut rc = eusn::EulerStateNetwork::new(params);

            let t0 = Instant::now();
            rc.train(&train_inputs, &train_targets);
            info!("ESN training done in {}ms", t0.elapsed().as_millis());

            let env = EnvMackeyGlass::new(
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
                "img/mackey_glass_esn.png",
                (3840, 1080),
            );
        }
        2 => {
            todo!("NG-RC not implemented for mackey-glass")
        }
        3 => {
            todo!("ESN-Firefly is not implemented for mackey-glass")
        }
        4 => {
            todo!("ESN-RandomSearch is not implemented for mackey-glass")
        }
        5 => {
            todo!("EuSN-Firefly is not implemented for mackey-glass")
        }
        6 => {
            todo!("EuSN-RandomSearch is not implemented for mackey-glass")
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
