use dialoguer::{theme::ColorfulTheme, Select};

#[macro_use]
extern crate log;

mod activation;
mod errors;
mod esn;
mod experiments;
mod load_sample_data;
pub(crate) mod plot;
mod utils;

pub(crate) type Series = Vec<(f64, f64)>;

pub(crate) const INPUT_DIM: usize = 1;
pub(crate) const OUTPUT_DIM: usize = 1;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    let experiments = ["sine", "trades", "mackey_glass"];

    let e = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Experiment")
        .items(&experiments)
        .default(0)
        .interact()
        .unwrap();
    match e {
        0 => experiments::sine::start(),
        1 => experiments::trades::start(),
        2 => experiments::mackey_glass::start(),
        _ => panic!("invalid experiment selected"),
    }
}
