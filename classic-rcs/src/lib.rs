#[macro_use]
extern crate log;

use nalgebra::{Const, Dynamic, Matrix, VecStorage};

mod esn;
// mod eusn;
// mod rc_trait;
mod params;
mod utils;

pub use esn::ESN;
pub use params::Params;

pub type StateMatrix = Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>;
