use crate::ReservoirComputer;

use lin_reg::LinReg;

use rc_plot::PlotGather;

mod env_mackey_glass;
mod env_trades;

pub use env_mackey_glass::EnvMackeyGlass;
pub use env_trades::EnvTrades;

/// Optimization environment for validating parameters
/// R: ReservoirComputer
/// I: Input dimension
/// O: Output dimension
/// N: Dimensionality of parameter search space
pub trait OptEnvironment<RC: ReservoirComputer<R>, R: LinReg> {
    /// Evaluate the reservoir computer and return the rmse values
    fn evaluate(&self, rc: &mut RC, plot: Option<&mut PlotGather>) -> f64;
}
