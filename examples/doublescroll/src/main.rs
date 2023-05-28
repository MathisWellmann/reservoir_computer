#[macro_use]
extern crate log;

use std::{fs::File, time::Instant};

use common::{Activation, ReservoirComputer};
use dialoguer::{theme::ColorfulTheme, Select};
use lin_reg::{LinReg, TikhonovRegularization};
use nalgebra::{Const, DMatrix, Dim, Dyn, Matrix, VecStorage};
use next_generation_rcs::{NGRCConstructor, NextGenerationRC, Params as NGRCParams};
use rc_plot::{plot, PlotGather};

const TRAIN_LEN: usize = 100;
const TEST_LEN: usize = 800;

pub(crate) fn main() {
    let file = File::open("doublescroll_soln.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut values: DMatrix<f64> = Matrix::from_element_generic(
        Dim::from_usize(3),
        Dim::from_usize(TRAIN_LEN + TEST_LEN - 1),
        0.0,
    );
    for (i, result) in rdr.records().enumerate() {
        let record = result.unwrap();

        let mut row: Vec<f64> = vec![];
        for r in record.iter().take(TRAIN_LEN + TEST_LEN) {
            let val: f64 = r.parse().unwrap();
            row.push(val);
        }
        let input_row: Matrix<f64, Const<1>, Dyn, VecStorage<f64, Const<1>, Dyn>> =
            Matrix::from_vec_generic(
                Dim::from_usize(1),
                Dim::from_usize(row.len() - 1),
                row.iter().take(row.len() - 1).cloned().collect::<Vec<f64>>(),
            );
        values.set_row(i, &input_row);
    }

    let rcs = vec!["ESN", "EuSN", "NG-RC"];

    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Reservoir Computer")
        .items(&rcs)
        .default(0)
        .interact()
        .unwrap();

    match e {
        0 => {
            todo!("ESN not yet available")
        }
        1 => {
            todo!("EuSN not yet available")
        }
        2 => {
            let num_time_delay_taps = 5;
            let num_samples_to_skip = 2;

            let params = NGRCParams {
                num_time_delay_taps,
                num_samples_to_skip,
                output_activation: Activation::Identity,
                reservoir_size: num_time_delay_taps * num_samples_to_skip,
            };
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.0001,
            };
            let ngrc_constructor = NGRCConstructor::new(num_time_delay_taps, num_samples_to_skip);
            let mut rc = NextGenerationRC::new(params, regressor, ngrc_constructor);

            let mut p = PlotGather::default();
            gather_plot_data(&values, &mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/doublescroll_ngrc.png",
                (2160, 2160),
            );
        }
        _ => panic!("invalid selection"),
    }
}

pub(crate) fn gather_plot_data<RC, R>(
    values: &DMatrix<f64>,
    rc: &mut RC,
    mut plot: Option<&mut PlotGather>,
) where
    RC: ReservoirComputer<R>,
    R: LinReg,
{
    let t0 = Instant::now();
    rc.train(&values.columns(0, TRAIN_LEN - 1), &values.columns(1, TRAIN_LEN));
    info!("NGRC training took {}ms", t0.elapsed().as_millis());

    for i in 1..values.nrows() {
        if let Some(plot) = plot.as_mut() {
            plot.push_target(i as f64, *values.column(i).get(0).unwrap());
        }

        let predicted_out = rc.readout();
        let last_pred = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: DMatrix<f64> =
            Matrix::from_fn_generic(Dim::from_usize(3), Dim::from_usize(1), |i, _| {
                *predicted_out.get(i).unwrap()
            });

        let input = if i > TRAIN_LEN {
            if let Some(plot) = plot.as_mut() {
                plot.push_test_pred(i as f64, last_pred);
            }
            m.row(0)
        } else {
            if let Some(plot) = plot.as_mut() {
                plot.push_train_pred(i as f64, last_pred);
            }
            values.row(i - 1)
        };

        rc.update_state(&input);
    }
}
