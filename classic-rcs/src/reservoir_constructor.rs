use nalgebra::{Const, DMatrix, Dynamic, Matrix, VecStorage};

use crate::StateMatrix;

/// Provides the abstraction needed for custom implementations of the reservoir generation process
pub trait ReservoirConstructor {
    fn construct_reservoir_weights(&mut self) -> DMatrix<f64>;

    fn construct_reservoir_biases(&mut self) -> StateMatrix;

    fn construct_input_weight_matrix(
        &mut self,
    ) -> Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;
}
