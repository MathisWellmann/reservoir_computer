#[macro_use]
extern crate log;

use arrayfire::{Array, Dim4, Seq, Window};
use next_generation_rcs_arrayfire::{Activation, NGRCArrayfire, Params};
use rc_plot::plot;
use time_series_generator::generate_sine_wave;

const TRAIN_LEN: usize = 600;
const SEED: Option<u64> = Some(0);

type Series = Vec<(f64, f64)>;

pub(crate) fn main() {
    pretty_env_logger::init();

    let mut values: Vec<f64> = generate_sine_wave(100);
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    values.append(&mut values.clone());
    info!("got {} datapoints", values.len());

    let params = Params {
        num_time_delay_taps: 20,
        num_samples_to_skip: 5,
        output_activation: Activation::Identity,
        regularization_coeff: 990.0,
    };
    let mut rc = NGRCArrayfire::new(params);

    let inputs: Vec<f32> = values.iter().take(TRAIN_LEN - 1).map(|x| *x as f32).collect();
    let targets: Vec<f32> = values.iter().skip(1).take(TRAIN_LEN - 1).map(|x| *x as f32).collect();
    rc.train(&inputs, &targets);

    let mut plot_targets: Series = Vec::with_capacity(1_000_000);
    let mut train_predictions: Series = Vec::with_capacity(TRAIN_LEN);
    let mut test_predictions: Series = Vec::with_capacity(1_000_000);

    let n_vals = values.len();
    for i in 1..n_vals {
        plot_targets.push((i as f64, values[i]));

        if i < rc.params().warmup_steps() {
            rc.update_state(values[i - 1] as f32);
            continue;
        }

        let predicted_out = rc.readout();

        // To begin forecasting, replace target input with it's own prediction
        let input = if i > TRAIN_LEN {
            test_predictions.push((i as f64, predicted_out as f64));
            predicted_out
        } else {
            train_predictions.push((i as f64, predicted_out as f64));
            values[i - 1] as f32
        };

        rc.update_state(input);
    }

    let filename = "img/sine.png";
    plot(&plot_targets, &train_predictions, &test_predictions, filename, (2560, 1440));
}
