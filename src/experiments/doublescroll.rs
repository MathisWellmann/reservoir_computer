use std::{fs::File, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use crate::{
    activation::Activation, environments::PlotGather, plot::plot, reservoir_computers::ngrc,
    ReservoirComputer,
};

const TRAIN_LEN: usize = 100;
const TEST_LEN: usize = 800;

pub(crate) fn start() {
    let file = File::open("doublescroll_soln.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut values: Matrix<f64, Const<3>, Dynamic, VecStorage<f64, Const<3>, Dynamic>> =
        Matrix::from_element_generic(
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
                num_time_delay_taps: 5,
                num_samples_to_skip: 2,
                regularization_coeff: 0.0001,
                output_activation: Activation::Identity,
            };
            let mut rc = ngrc::NextGenerationRC::new(params);

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

pub(crate) fn gather_plot_data<R, const N: usize>(
    values: &Matrix<f64, Const<3>, Dynamic, VecStorage<f64, Const<3>, Dynamic>>,
    rc: &mut R,
    mut plot: Option<&mut PlotGather>,
) where
    R: ReservoirComputer<3, 3, N>,
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
        let m: Matrix<f64, Const<3>, Dynamic, VecStorage<f64, Const<3>, Dynamic>> =
            Matrix::from_fn_generic(Dim::from_usize(3), Dim::from_usize(1), |i, _| {
                *predicted_out.get(i).unwrap()
            });
        let target = *values.column(j).get(0).unwrap();

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
