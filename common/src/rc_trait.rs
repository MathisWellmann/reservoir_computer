use lin_reg::LinReg;
use nalgebra::{Const, DMatrix, Dyn, Matrix, MatrixView, VecStorage};

/// The ReservoirComputer trait
/// N: Number of values to map into Parameters
/// R: The linear regression method to use
pub trait ReservoirComputer<R: LinReg> {
    /// The reservoir parameters
    fn params(&self) -> &dyn RCParams;

    /// Train the readout layer using the given inputs and targets
    ///
    /// # Arguments:
    /// inputs: A Matrix where there are N rows corresponding to the datapoints
    fn train<'a>(
        &mut self,
        inputs: &'a MatrixView<'a, f64, Dyn, Dyn, Const<1>, Dyn>,
        targets: &'a MatrixView<'a, f64, Dyn, Dyn, Const<1>, Dyn>,
    );

    fn update_state<'a>(&mut self, input: &'a MatrixView<'a, f64, Const<1>, Dyn, Const<1>, Dyn>);

    /// Performs a readout of the current reservoir state
    fn readout(&self) -> Matrix<f64, Const<1>, Dyn, VecStorage<f64, Const<1>, Dyn>>;

    /// Sets the internal state matrix
    fn set_state(&mut self, state: Matrix<f64, Const<1>, Dyn, VecStorage<f64, Const<1>, Dyn>>);

    fn readout_matrix(&self) -> &DMatrix<f64>;
}

pub trait RCParams {
    fn initial_state_value(&self) -> f64;

    fn reservoir_size(&self) -> usize;
}
