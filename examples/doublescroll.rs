#[macro_use]
extern crate log;

mod plot;

use std::{fs::File, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, VecStorage};
use plot::{plot, PlotGather, Series};
use reservoir_computer::{ngrc, Activation, LinReg, ReservoirComputer, TikhonovRegularization};

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
        let input_row: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
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
            let params = ngrc::Params {
                input_dim: 3,
                output_dim: 3,
                num_time_delay_taps: 5,
                num_samples_to_skip: 2,
                output_activation: Activation::Identity,
            };
            let regressor = TikhonovRegularization {
                regularization_coeff: 0.0001,
            };
            let mut rc = ngrc::NextGenerationRC::new(params, regressor);

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

pub(crate) fn gather_plot_data<RC, const N: usize, R>(
    values: &DMatrix<f64>,
    rc: &mut RC,
    mut plot: Option<&mut PlotGather>,
) where
    RC: ReservoirComputer<N, R>,
    R: LinReg,
{
    let t0 = Instant::now();
    rc.train(&values.columns(0, TRAIN_LEN - 1), &values.columns(1, TRAIN_LEN));
    info!("NGRC training took {}ms", t0.elapsed().as_millis());

    for j in 1..values.ncols() {
        if let Some(plot) = plot.as_mut() {
            plot.push_target(j as f64, *values.column(j).get(0).unwrap());
        }

        let predicted_out = rc.readout();
        let last_pred = *predicted_out.get(0).unwrap();

        // To begin forecasting, replace target input with it's own prediction
        let m: DMatrix<f64> =
            Matrix::from_fn_generic(Dim::from_usize(3), Dim::from_usize(1), |i, _| {
                *predicted_out.get(i).unwrap()
            });

        let input = if j > TRAIN_LEN {
            if let Some(plot) = plot.as_mut() {
                plot.push_test_pred(j as f64, last_pred);
            }
            m.column(0)
        } else {
            if let Some(plot) = plot.as_mut() {
                plot.push_train_pred(j as f64, last_pred);
            }
            values.column(j - 1)
        };

        rc.update_state(&input, &predicted_out);
    }
}