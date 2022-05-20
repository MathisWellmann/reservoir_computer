use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, VecStorage};
use nanorand::{Rng, WyRand};

use crate::{Params, ReservoirConstructor, StateMatrix};
use common::{RCParams, ReservoirComputer};
use lin_reg::LinReg;

/// The Reseoir Computer, Leaky Echo State Network
#[derive(Debug)]
pub struct RC<R> {
    params: Params,
    input_weight_matrix: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    reservoir_weights: DMatrix<f64>,
    reservoir_biases: StateMatrix,
    readout_matrix: DMatrix<f64>,
    state: StateMatrix,
    rng: WyRand,
    regressor: R,
}

impl<R> RC<R> {
    /// Create a new reservoir, with random initiallization
    /// # Arguments
    pub fn new<C>(params: Params, regressor: R, mut reservoir_constructor: C) -> Self
    where
        C: ReservoirConstructor,
    {
        let mut rng = match params.seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };

        let reservoir_weights = reservoir_constructor.construct_reservoir_weights();
        let reservoir_biases = reservoir_constructor.construct_reservoir_biases();
        let input_weight_matrix = reservoir_constructor.construct_input_weight_matrix();

        let readout_matrix = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            |_, _| rng.generate::<f64>() * 2.0 - 1.0,
        );
        let state = Matrix::from_element_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            params.initial_state_value,
        );

        Self {
            params,
            reservoir_weights,
            input_weight_matrix,
            readout_matrix,
            state,
            reservoir_biases,
            rng,
            regressor,
        }
    }
}

impl<R> ReservoirComputer<R> for RC<R>
where
    R: LinReg,
{
    #[inline(always)]
    fn params(&self) -> &dyn RCParams {
        &self.params
    }

    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) {
        let washout_len = (inputs.ncols() as f64 * self.params.washout_pct) as usize;
        let harvest_len = inputs.nrows() - washout_len;

        let mut design: DMatrix<f64> = DMatrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(self.params.reservoir_size + 1), // +1 to account for column of 1s
            0.0,
        );

        for i in 0..inputs.nrows() {
            self.update_state(&inputs.row(i));

            // discard earlier values, as the state has to stabilize first
            if i >= washout_len {
                let d: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                    Matrix::from_fn_generic(
                        Dim::from_usize(1),
                        Dim::from_usize(self.params.reservoir_size + 1), // +1 to account for 1 at j == 0
                        |_, j| {
                            if j == 0 {
                                1.0
                            } else {
                                *self.state.get(j - 1).unwrap()
                            }
                        },
                    );
                design.set_row(i - washout_len, &d);
            }
        }

        self.readout_matrix = self
            .regressor
            .fit_readout(&design.rows(0, design.nrows()), &targets.rows(0, harvest_len));

        info!("readout dims: ({}, {})", self.readout_matrix.nrows(), self.readout_matrix.ncols());

        /*
        // TODO: move this qr based fit into its own file
        // TODO: put into its own lin_reg file with tests
        // Ridge regression regularization, I think
        let x: DMatrix<f64> = Matrix::from_fn_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(design_cols),
            |i, j| {
                if i == j {
                    *design_matrix.row(i).column(j).get(0).unwrap()
                        + self.params.regularization_coeff
                } else {
                    *design_matrix.row(i).column(j).get(0).unwrap()
                }
            },
        );
        let qr = x.qr();
        let a = qr.r().try_inverse().unwrap() * qr.q().transpose();
        let b = a * &target_matrix;
        self.readout_matrix = b.transpose();
        */
    }

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<1>, Dynamic, Const<1>, Dynamic>,
    ) {
        let noise: StateMatrix = Matrix::from_fn_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(1),
            |_, _| (self.rng.generate::<f64>() * 2.0 - 1.0) * self.params.state_update_noise_frac,
        );
        let lin_part = &self.input_weight_matrix * input.transpose();

        let mut state_delta: StateMatrix = lin_part
            + self.params.leaking_rate * (&self.reservoir_weights * &self.state)
            + &self.reservoir_biases
            + noise;
        self.params.reservoir_activation.activate(state_delta.as_mut_slice());

        // perform node-to-node update
        self.state = (1.0 - self.params.leaking_rate) * &self.state + state_delta;
    }

    /// Perform a readout operation
    #[inline]
    #[must_use]
    fn readout(&self) -> Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> {
        debug!(
            "readout: dims of state: ({}, {}), readout: ({}, {})",
            self.state.nrows(),
            self.state.ncols(),
            self.readout_matrix.nrows(),
            self.readout_matrix.ncols()
        );
        // prepend the 1 to state for proper readout
        let state: StateMatrix = Matrix::from_fn_generic(
            Dim::from_usize(self.params.reservoir_size + 1),
            Dim::from_usize(1),
            |i, _| {
                if i == 0 {
                    1.0
                } else {
                    *self.state.get(i - 1).unwrap()
                }
            },
        );
        let mut pred = &state.transpose() * &self.readout_matrix;
        self.params.output_activation.activate(pred.as_mut_slice());

        pred
    }

    /// Resets the state to it's initial values
    #[inline(always)]
    fn set_state(
        &mut self,
        state: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    ) {
        self.state = state.transpose();
    }

    #[inline(always)]
    fn readout_matrix(&self) -> &DMatrix<f64> {
        &self.readout_matrix
    }
}
