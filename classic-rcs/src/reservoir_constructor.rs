use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};

use crate::StateMatrix;

/// Provides the abstraction needed for custom implementations of the reservoir
/// generation process
pub trait ReservoirConstructor {
    /// Construct the internal reservoir weight matrix
    fn construct_reservoir_weights(&mut self) -> DMatrix<f64>;

    /// Construct the biases of internal neurons
    fn construct_reservoir_biases(&mut self) -> StateMatrix;

    /// Construct the input connections into the reservoir
    fn construct_input_weight_matrix(
        &mut self,
    ) -> Matrix<f64, Dyn, Const<1>, VecStorage<f64, Dyn, Const<1>>>;
}
