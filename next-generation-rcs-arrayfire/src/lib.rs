#[macro_use]
extern crate log;

mod activation;
mod ngrc;
mod params;

pub use activation::Activation;
pub use ngrc::NGRCArrayfire;
pub use params::Params;
