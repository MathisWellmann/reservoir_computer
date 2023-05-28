//! This crate is for the classic variants of reservoir computers

#![deny(missing_docs)]
#![warn(clippy::all)]

use nalgebra::{Const, Dyn, Matrix, VecStorage};

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

/// The State matrix used for the classic reservoir computers
pub type StateMatrix = Matrix<f64, Dyn, Const<1>, VecStorage<f64, Dyn, Const<1>>>;
