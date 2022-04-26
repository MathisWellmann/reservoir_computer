use std::{fs::File, sync::Arc, time::Instant};

use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};

use crate::{
    activation::Activation,
    environments::{env_trades::EnvTrades, PlotGather},
    plot::plot,
    reservoir_computers::ngrc,
    ReservoirComputer,
};

const TRAIN_LEN: usize = 100;
const TEST_LEN: usize = 800;

pub(crate) fn start() {
    let file = File::open("doublescroll_soln.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut inputs: Matrix<f64, Const<3>, Dynamic, VecStorage<f64, Const<3>, Dynamic>> =
        Matrix::from_element_generic(
            Dim::from_usize(3),
            Dim::from_usize(TRAIN_LEN + TEST_LEN - 1),
            0.0,
        );
    let mut targets: Matrix<f64, Const<3>, Dynamic, VecStorage<f64, Const<3>, Dynamic>> =
        Matrix::from_element_generic(
            Dim::from_usize(3),
            Dim::from_usize(TRAIN_LEN + TEST_LEN - 1),
            0.0,
        );
    for (i, result) in rdr.records().enumerate() {
        let record = result.unwrap();
        println!("record: {:?}", record);

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
        let target_row: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
            Matrix::from_vec_generic(
                Dim::from_usize(1),
                Dim::from_usize(row.len() - 1),
                row.iter().skip(1).cloned().collect::<Vec<f64>>(),
            );
        inputs.set_row(i, &input_row);
        targets.set_row(i, &target_row);
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
            /*
            let params = ngrc::Params {
                num_time_delay_taps: 5,
                num_samples_to_skip: 2,
                regularization_coeff: 0.0001,
                output_activation: Activation::Identity,
            };
            let mut rc = ngrc::NextGenerationRC::new(params);
            let t0 = Instant::now();
            rc.train(&inputs.columns(0, TRAIN_LEN), &targets.columns(0, TRAIN_LEN));
            info!("NGRC training took {}ms", t0.elapsed().as_millis());

            let env = DoubleScrollEnv::new();
            let mut p = PlotGather::default();
            env.evaluate(&mut rc, Some(&mut p));

            plot(
                &p.plot_targets(),
                &p.train_predictions(),
                &p.test_predictions(),
                "img/trades_ngrc.png",
                (2160, 2160),
            );
            */
        }
        _ => panic!("invalid selection"),
    }
}
