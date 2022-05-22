use lin_reg::LinReg;
use nalgebra::{Const, DMatrix, Dynamic, Matrix, MatrixSlice, VecStorage};

/// The ReservoirComputer trait
/// N: Number of values to map into Parameters
/// R: The linear regression method to use
pub trait ReservoirComputer<R: LinReg> {
    /// The reservoir parameters
    fn params(&self) -> &dyn RCParams;

    /// Train the readout layer using the given inputs and targets
    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    );

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<1>, Dynamic, Const<1>, Dynamic>,
    );

    /// Performs a readout of the current reservoir state
    fn readout(&self) -> Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>;

    /// Sets the internal state matrix
    fn set_state(
        &mut self,
        state: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    );

    fn readout_matrix(&self) -> &DMatrix<f64>;
}

pub trait RCParams {
    fn initial_state_value(&self) -> f64;

    fn reservoir_size(&self) -> usize;
}
