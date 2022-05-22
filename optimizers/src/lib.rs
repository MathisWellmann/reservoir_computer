use common::ReservoirComputer;
use lin_reg::LinReg;

pub mod firefly;
pub mod random_search;

pub trait OptEnvironment<RC, R>
where
    RC: ReservoirComputer<R>,
    R: LinReg,
{
    /// Evaluates the reservoir computers performance in the environment
    ///
    /// # Arguments:
    /// rc: mutable reference to a reservoir computer implementation
    ///
    /// # Returns:
    /// root mean square error
    fn evaluate(&self, rc: &mut RC) -> f64;
}
