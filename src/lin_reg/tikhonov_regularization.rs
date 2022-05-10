use nalgebra::{ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, VecStorage};

use super::LinReg;

/// Tikhonov regularization aka ridge regression
/// It is particularly useful to mitigate the problem of multicollinearity in
/// linear regression
#[derive(Debug, Clone)]
pub struct TikhonovRegularization {
    /// Ridge parameter
    pub regularization_coeff: f64,
}

impl LinReg for TikhonovRegularization {
    fn fit_readout<'a, const O: usize>(
        &self,
        design: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Const<O>, Dynamic, Const<1>, Const<O>>,
    ) -> Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>> {
        let reg_m: DMatrix<f64> = Matrix::from_diagonal_element_generic(
            Dim::from_usize(design.nrows()),
            Dim::from_usize(design.nrows()),
            self.regularization_coeff,
        );

        let p0 = targets * design.transpose();
        let p1 = design * design.transpose();
        let p2 = p1 + reg_m;

        p0 * p2.try_inverse().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tikhonov_regularization() {
        todo!()
    }
}
