#[macro_use]
extern crate log;

use nalgebra::{Const, Dynamic, Matrix, VecStorage};

mod esn;
// mod eusn;
// mod rc_trait;
mod params;

pub use esn::ESN;
pub use params::Params;

pub type StateMatrix = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;
