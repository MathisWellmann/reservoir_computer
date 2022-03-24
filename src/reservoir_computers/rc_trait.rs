use nalgebra::{ArrayStorage, Const, Dynamic, Matrix, MatrixSlice, VecStorage};

pub(crate) type StateMatrix = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;

/// The ReservoirComputer trait is generic over it's required parameters
/// as well as the input dimension I
/// and the output dimension O
pub trait ReservoirComputer<P: RCParams, const I: usize, const O: usize> {
    /// Create a new ReservoirComputer instance with the given parameters
    /// and randomly initialized matrices
    fn new(params: P) -> Self;

    /// Train the readout layer using the given inputs and targets
    fn train(
        &mut self,
        inputs: &Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>>,
        targets: &Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    );

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<I>, Const<1>, Const<1>, Const<I>>,
        prev_pred: &Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>>,
    );

    /// Performs a readout of the current reservoir state
    fn readout(&self) -> Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>>;

    /// Sets the internal state matrix
    fn set_state(&mut self, state: StateMatrix);

    fn params(&self) -> &P;
}

pub trait RCParams {
    fn initial_state_value(&self) -> f64;

    fn reservoir_size(&self) -> usize;
}
