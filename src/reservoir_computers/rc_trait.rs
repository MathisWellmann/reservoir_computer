use nalgebra::{Const, DMatrix, Dynamic, Matrix, MatrixSlice, VecStorage};

use crate::LinReg;

pub(crate) type StateMatrix = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;

/// The ReservoirComputer trait
/// N: Number of values to map into Parameters
/// R: The linear regression method to use
pub trait ReservoirComputer<const N: usize, R: LinReg> {
    type ParamMapper: OptParamMapper<N>;

    /// Create a new ReservoirComputer instance with the given parameters
    /// and randomly initialized matrices
    fn new(
        params: <<Self as ReservoirComputer<N, R>>::ParamMapper as OptParamMapper<N>>::Params,
        regressor: R,
    ) -> Self;

    /// Train the readout layer using the given inputs and targets
    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    );

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<1>, Dynamic, Const<1>, Dynamic>,
        prev_pred: &Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    );

    /// Performs a readout of the current reservoir state
    fn readout(&self) -> Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>;

    /// Sets the internal state matrix
    fn set_state(
        &mut self,
        state: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    );

    fn params(
        &self,
    ) -> &<<Self as ReservoirComputer<N, R>>::ParamMapper as OptParamMapper<N>>::Params;

    fn readout_matrix(&self) -> &DMatrix<f64>;
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
