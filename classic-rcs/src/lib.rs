#[macro_use]
extern crate log;

use nalgebra::{Const, Dynamic, Matrix, VecStorage};

mod esn_constructor;
mod eusn_constructor;
mod params;
mod rc;
mod reservoir_constructor;

pub use esn_constructor::ESNConstructor;
pub use eusn_constructor::EUSNConstructor;
pub use params::Params;
pub use rc::RC;
pub use reservoir_constructor::ReservoirConstructor;

pub type StateMatrix = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;
