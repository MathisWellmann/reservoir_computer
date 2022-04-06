use crate::ReservoirComputer;

pub mod env_mackey_glass;
pub mod env_trades;

/// Optimization environment for validating parameters
/// R: ReservoirComputer
/// I: Input dimension
/// O: Output dimension
/// N: Dimensionality of parameter search space
pub trait OptEnvironment<
    R: ReservoirComputer<I, O, N>,
    const I: usize,
    const O: usize,
    const N: usize,
>
{
    /// Evaluate the reservoir computer and return the rmse values
    fn evaluate(&self, rc: &mut R) -> f64;
}
