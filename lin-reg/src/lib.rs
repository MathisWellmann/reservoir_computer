#[macro_use]
extern crate log;

use nalgebra::{Const, DMatrix, Dynamic, MatrixSlice};

mod tikhonov_regularization;

pub use tikhonov_regularization::TikhonovRegularization;

/// Generic way of performing linear regression and fitting the readout matrix
pub trait LinReg: Clone {
    /// Fit a readout matrix, mapping inputs to targets
    ///
    /// # Parameters
    /// design: Input data having, where the first column should be just 1s
    /// targets: Target data having O rows as the output dimensionality
    fn fit_readout<'a>(
        &self,
        design: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) -> DMatrix<f64>;
}
