use crate::ReservoirComputer;

pub mod environment_trades;

/// Optimization environment for validating parameters
/// R: ReservoirComputer
pub trait OptEnvironment<
    R: ReservoirComputer<I, O, N>,
    const I: usize,
    const O: usize,
    const N: usize,
>
{
    /// Evaluate the parameters and return the fitness values
    fn evaluate(&self, rc: &mut R) -> f64;
}
