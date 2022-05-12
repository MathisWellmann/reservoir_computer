use reservoir_computer::{LinReg, ReservoirComputer};

use crate::plot::PlotGather;

pub mod env_mackey_glass;
pub mod env_trades;

/// Optimization environment for validating parameters
/// R: ReservoirComputer
/// I: Input dimension
/// O: Output dimension
/// N: Dimensionality of parameter search space
pub trait OptEnvironment<RC: ReservoirComputer<N, R>, const N: usize, R: LinReg> {
    /// Evaluate the reservoir computer and return the rmse values
    fn evaluate(&self, rc: &mut RC, plot: Option<&mut PlotGather>) -> f64;
}
