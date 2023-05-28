//! A linear regression crate for fitting the readout layer

#![deny(unused_imports, unused_crate_dependencies)]
#![warn(missing_docs)]

use nalgebra::{Const, DMatrix, Dyn, MatrixSlice};

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
        design: &'a MatrixSlice<'a, f64, Dyn, Dyn, Const<1>, Dyn>,
        targets: &'a MatrixSlice<'a, f64, Dyn, Dyn, Const<1>, Dyn>,
    ) -> DMatrix<f64>;
}
