use dialoguer::{theme::ColorfulTheme, Select};

#[macro_use]
extern crate log;

mod activation;
mod errors;
mod experiments;
mod firefly_optimizer;
mod load_sample_data;
pub(crate) mod plot;
mod reservoir_computers;
mod utils;

pub(crate) type Series = Vec<(f64, f64)>;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    let experiments = ["sine", "trades", "trades_sliding_window", "mackey_glass"];

    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Experiment")
        .items(&experiments)
        .default(0)
        .interact()
        .unwrap();
    match e {
        0 => experiments::sine::start(),
        1 => experiments::trades::trades::start(),
        2 => experiments::trades::trades_sliding_window::start(),
        3 => experiments::mackey_glass::start(),
        _ => panic!("invalid experiment selected"),
    }
}
