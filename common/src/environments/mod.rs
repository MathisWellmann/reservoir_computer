//! Provides an `OptEnvironment`

use lin_reg::LinReg;
use rc_plot::PlotGather;

use crate::ReservoirComputer;

mod env_mackey_glass;

pub use env_mackey_glass::EnvMackeyGlass;

/// Optimization environment for validating parameters
/// R: ReservoirComputer
/// I: Input dimension
/// O: Output dimension
/// N: Dimensionality of parameter search space
pub trait OptEnvironment<RC: ReservoirComputer<R>, R: LinReg> {
    /// Evaluate the reservoir computer and return the rmse values
    fn evaluate(&self, rc: &mut RC, plot: Option<&mut PlotGather>) -> f64;
}
