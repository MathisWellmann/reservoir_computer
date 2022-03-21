use std::time::Instant;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use crate::{
    activation::Activation,
    esn::Inputs,
    euler_state_network::{EuSNParams, EulerStateNetwork},
    experiments::mackey_glass::mackey_glass_series,
    plot::plot,
    Series,
};

pub(crate) fn start() {
    let train_len = 5000;
    let test_len = 1000;
    let total_len = train_len + test_len;

    let seed = Some(0);
    let values = mackey_glass_series(total_len, 30, seed);
    info!("values: {:?}", values);

    let washout_frac = 0.1;
    let params = EuSNParams {
        input_sparsity: 0.1,
        input_weight_scaling: 0.5,
        reservoir_size: 500,
        reservoir_weight_scaling: 0.7,
        reservoir_bias_scaling: 0.1,
        reservoir_activation: Activation::Relu,
        initial_state_value: values[0],
        seed,
        washout_frac,
        regularization_coeff: 0.1,
        epsilon: 0.008,
        gamma: 0.05,
    };
    let mut rc = EulerStateNetwork::new(params);

    let train_inputs = Matrix::from_vec_generic(
        Dim::from_usize(train_len),
        Dim::from_usize(1),
        values.iter().take(train_len).cloned().collect::<Vec<f64>>(),
    );
    let train_targets = Matrix::from_vec_generic(
        Dim::from_usize(train_len),
        Dim::from_usize(1),
        values.iter().skip(1).take(train_len).cloned().collect::<Vec<f64>>(),
    );
    let t0 = Instant::now();
    rc.train(&train_inputs, &train_targets);
    info!("ESN training done in {}ms", t0.elapsed().as_millis());

    let mut plot_targets: Series = vec![];
    let mut train_predictions: Series = vec![];
    let mut test_predictions: Series = vec![];

    let inputs: Inputs =
        Matrix::from_vec_generic(Dim::from_usize(values.len()), Dim::from_usize(1), values);
    rc.reset_state();

    let mut train_rmse = 0.0;
    let washout_idx = (washout_frac * total_len as f64) as usize;
    for i in 0..total_len {
        plot_targets.push((i as f64, *inputs.row(i).get(0).unwrap()));

        let predicted_out = rc.readout();
        let mut last_prediction = *predicted_out.get(0).unwrap();
        if !last_prediction.is_finite() {
            last_prediction = 0.0;
        }

        // To begin forecasting, replace target input with it's own prediction
        let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |_, j| {
                *predicted_out.get(j).unwrap()
            });
        let input = if i > train_len {
            test_predictions.push((i as f64, last_prediction));
            m.row(0)
        } else {
            if i > washout_idx {
                train_rmse += (*inputs.row(i).get(0).unwrap() - last_prediction).powi(2);
            }
            train_predictions.push((i as f64, last_prediction));
            inputs.row(i)
        };

        rc.update_state(&input);
    }
    info!("train_rmse: {}", train_rmse.sqrt());

    plot(
        &plot_targets,
        &train_predictions,
        &test_predictions,
        "img/mackey_glass_eusn.png",
        (3840, 1080),
    );
}
