use dialoguer::{theme::ColorfulTheme, Select};

#[macro_use]
extern crate log;

mod activation;
mod errors;
mod esn;
mod euler_state_network;
mod experiments;
mod load_sample_data;
pub(crate) mod plot;
mod utils;
mod firefly_optimizer;

pub(crate) type Series = Vec<(f64, f64)>;

pub(crate) const INPUT_DIM: usize = 1;
pub(crate) const OUTPUT_DIM: usize = 1;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    let experiments = [
        "sine",
        "trades",
        "trades_eusn",
        "trades_sliding_window",
        "mackey_glass",
        "mackey_glass_eusn"
    ];

    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Experiment")
        .items(&experiments)
        .default(0)
        .interact()
        .unwrap();
    match e {
        0 => experiments::sine::start(),
        1 => experiments::trades::trades::start(),
        2 => experiments::trades::trades_eusn::start(),
        3 => experiments::trades::trades_sliding_window::start(),
        4 => experiments::mackey_glass::start(),
        5 => experiments::mackey_glass_eusn::start(),
        _ => panic!("invalid experiment selected"),
    }
}
