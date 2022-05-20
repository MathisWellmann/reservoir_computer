use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice};

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
    fn fit_readout<'a>(
        &self,
        design: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) -> DMatrix<f64> {
        let reg_m: DMatrix<f64> = Matrix::from_diagonal_element_generic(
            Dim::from_usize(design.ncols()),
            Dim::from_usize(design.ncols()),
            self.regularization_coeff,
        );

        let p0 = design.transpose() * design;
        let p1 = (p0 + reg_m).try_inverse().unwrap();
        let p2 = design.transpose() * targets;

        p1 * p2
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::VecStorage;
    use round::round;

    use super::*;

    #[test]
    fn tikhonov_regularization() {
        if let Err(_) = pretty_env_logger::try_init() {}

        // Note the first column being just ones
        let design: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(4),
            Dim::from_usize(3),
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0],
        );
        let targets: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(4),
            Dim::from_usize(1),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        info!("design: {}, targets: {}", design, targets);

        let regressor = TikhonovRegularization {
            regularization_coeff: 0.0,
        };
        let mut readout_matrix = regressor
            .fit_readout(&design.columns(0, design.ncols()), &targets.columns(0, targets.ncols()));
        info!("readout_matrix: {}", readout_matrix);

        let goal_matrix: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_vec_generic(Dim::from_usize(3), Dim::from_usize(1), vec![1.0, 1.0, 0.0]);

        // round readout
        readout_matrix.iter_mut().for_each(|v| *v = round(*v, 1));

        assert_eq!(readout_matrix, goal_matrix,)
    }

    #[test]
    fn tikhonov_regularization_shifted() {
        if let Err(_) = pretty_env_logger::try_init() {}

        // Note the first column being just ones
        let design: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(4),
            Dim::from_usize(3),
            vec![100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 200.0, 300.0, 0.0, 0.0, 100.0, 200.0],
        );
        let targets: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(4),
            Dim::from_usize(1),
            vec![100.0, 200.0, 300.0, 400.0],
        );
        info!("design: {}, targets: {}", design, targets);

        let regressor = TikhonovRegularization {
            regularization_coeff: 0.0,
        };
        let mut readout_matrix = regressor
            .fit_readout(&design.columns(0, design.ncols()), &targets.columns(0, targets.ncols()));
        info!("readout_matrix: {}", readout_matrix);

        let goal_matrix: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_vec_generic(Dim::from_usize(3), Dim::from_usize(1), vec![1.0, 1.0, 0.0]);

        // round readout
        readout_matrix.iter_mut().for_each(|v| *v = round(*v, 1));

        assert_eq!(readout_matrix, goal_matrix,)
    }

    /// Tests how to extract the last observed state and perform a readout from it
    #[test]
    fn readout_from_state() {
        if let Err(_) = pretty_env_logger::try_init() {}

        // Note the first column being just ones
        let design: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(4),
            Dim::from_usize(3),
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0],
        );
        let targets: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(4),
            Dim::from_usize(1),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        info!("design: {}, targets: {}", design, targets);

        // Try to use the last row to predict the last target
        let state: DMatrix<f64> =
            Matrix::from_vec_generic(Dim::from_usize(1), Dim::from_usize(3), vec![1.0, 3.0, 2.0]);
        let target: DMatrix<f64> =
            Matrix::from_vec_generic(Dim::from_usize(1), Dim::from_usize(1), vec![4.0]);
        info!("state: {}, target: {}", state, target);

        let readout: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
            Matrix::from_vec_generic(Dim::from_usize(3), Dim::from_usize(1), vec![1.0, 1.0, 0.0]);

        let o = state * readout;
        info!("o: {}", o);

        assert_eq!(o, target);
    }
}
