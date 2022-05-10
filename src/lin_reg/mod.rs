use nalgebra::{Const, Dynamic, Matrix, MatrixSlice, VecStorage};

mod tikhonov_regularization;

pub use tikhonov_regularization::TikhonovRegularization;

/// Generic way of performing linear regression and fitting the readout matrix
pub trait LinReg: Clone {
    /// Fit a readout matrix, mapping inputs to targets
    ///
    /// # Parameters
    /// design: Input data having I rows as the input dimensionality
    /// targets: Target data having O rows as the output dimensionality
    fn fit_readout<'a, const O: usize>(
        &self,
        design: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Const<O>, Dynamic, Const<1>, Const<O>>,
    ) -> Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>;
}
