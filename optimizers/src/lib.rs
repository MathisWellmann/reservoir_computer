//! Optimizers that can evaluate reservoir computers

#![deny(unused_imports, unused_crate_dependencies)]
#![warn(missing_docs)]

use common::ReservoirComputer;
use lin_reg::LinReg;

pub mod firefly;
pub mod random_search;

/// Any environment that wants to evaluate a reservoir computer must implement this
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
