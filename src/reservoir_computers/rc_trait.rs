use nalgebra::{ArrayStorage, Const, Dynamic, Matrix, MatrixSlice, VecStorage};

pub(crate) type StateMatrix = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;

/// The ReservoirComputer trait
/// I: input dimension
/// O: output dimension
/// N: Number of values to map into Parameters
pub trait ReservoirComputer<const I: usize, const O: usize, const N: usize> {
    type ParamMapper: OptParamMapper<N>;

    /// Create a new ReservoirComputer instance with the given parameters
    /// and randomly initialized matrices
    fn new(
        params: <<Self as ReservoirComputer<I, O, N>>::ParamMapper as OptParamMapper<N>>::Params,
    ) -> Self;

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

    fn params(
        &self,
    ) -> &<<Self as ReservoirComputer<I, O, N>>::ParamMapper as OptParamMapper<N>>::Params;

    fn readout_matrix(&self)
        -> &Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>;
}

pub trait RCParams {
    fn initial_state_value(&self) -> f64;

    fn reservoir_size(&self) -> usize;
}

/// This is used for specifying desired parameter ranges
pub type Range = (f64, f64);

/// Maps the optimizer candidate parameters to concrete RC params
/// T is the type of Parameter output specific to the type of Reservoir Computer
/// N is the dimensionality of parameter space and is specific to the type of
/// Reservoir Computer
pub trait OptParamMapper<const N: usize> {
    type Params: RCParams;

    fn map(&self, params: &[f64; N]) -> Self::Params;
}
