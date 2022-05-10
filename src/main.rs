use dialoguer::{theme::ColorfulTheme, Select};

#[macro_use]
extern crate log;

mod activation;
mod environments;
mod experiments;
mod lin_reg;
mod load_sample_data;
mod optimizers;
pub(crate) mod plot;
mod reservoir_computers;
mod temporal_prediction_aggregator;
mod utils;

pub use environments::OptEnvironment;
pub use lin_reg::LinReg;
use nalgebra::{Const, Dynamic, Matrix, VecStorage};
pub use reservoir_computers::{RCParams, ReservoirComputer};

pub type Series = Vec<(f64, f64)>;

/// Used for single dimensional IO to and from the reservoir computers
pub type SingleDimIo = Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    let experiments = ["sine", "trades", "trades_sliding_window", "mackey_glass", "double_scroll"];

    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Experiment")
        .items(&experiments)
        .default(0)
        .interact()
        .unwrap();
    match e {
        0 => experiments::sine::start(),
        1 => experiments::trades::start(),
        2 => experiments::trades_sliding_window::start(),
        3 => experiments::mackey_glass::start(),
        4 => experiments::doublescroll::start(),
        _ => panic!("invalid experiment selected"),
    }
}
