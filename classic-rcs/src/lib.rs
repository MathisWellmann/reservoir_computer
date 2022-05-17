#[macro_use]
extern crate log;

mod esn;
// mod eusn;
// mod rc_trait;
mod utils;

use nalgebra::{Const, Dynamic, Matrix, VecStorage};

pub type StateMatrix = Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>;
