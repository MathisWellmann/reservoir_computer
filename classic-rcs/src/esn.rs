use nalgebra::{
    ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, SymmetricEigen, VecStorage,
};
use nanorand::{Rng, WyRand};

use crate::StateMatrix;
use common::Activation;
use lin_reg::LinReg;

/// The parameters of the Echo State Network
#[derive(Debug, Clone)]
pub struct Params {
    /// Probability of inputs connecting to state
    pub input_sparsity: f64,
    /// Activation function applied to result of input transformation
    pub input_activation: Activation,
    /// Scales the input weight matrix
    pub input_weight_scaling: f64,

    /// Number of nodes in the reservoir
    pub reservoir_size: usize,
    /// Scales the reservoir biases
    pub reservoir_bias_scaling: f64,
    /// Connection probability within the reservoir
    pub reservoir_sparsity: f64,
    /// Activation function of reservoir state transition
    pub reservoir_activation: Activation,

    /// Controls the retention of information from previous time steps.
    /// The spectral radius determines how fast the influence of an input
    /// dies out in a reservoir with time, and how stable the reservoir
    /// activations are. The spectral radius should be greater in tasks
    /// requiring longer memory of the input.
    pub spectral_radius: f64,
    /// Tunes the decay time of internal activity of the network
    /// The leaking rate a can be regarded as the speed of the reservoir
    /// update dynamics discretized in time. This can be adapted online
    /// to deal with time wrapping of the signals. Set the leaking rate
    /// to match the speed of the dynamics of input / target This sort
    /// of acts as a EMA smoothing filter, so other MAs could be used
    /// such as ALMA
    pub leaking_rate: f64,
    /// Ridge regression regulazation applied in training
    pub regularization_coeff: f64,
    /// Fraction of initial state transitions to disregard in training
    pub washout_pct: f64,
    /// Activation function of networks readout
    pub output_activation: Activation,
    /// Optional seed for Rng
    pub seed: Option<u64>,
    /// Fraction of noise to add to state update equation
    pub state_update_noise_frac: f64,
    /// Initial value of state
    pub initial_state_value: f64,

    /// If true, concatenates inputs and state to perform readout
    pub readout_from_input_as_well: bool,
}

/*
impl RCParams for Params {
    #[inline(always)]
    fn initial_state_value(&self) -> f64 {
        self.initial_state_value
    }

    #[inline(always)]
    fn reservoir_size(&self) -> usize {
        self.reservoir_size
    }
}
*/

/*
pub struct ParamMapper {
    /// Parameter ranges
    pub input_sparsity_range: Range,
    pub input_activation: Activation,
    pub input_weight_scaling_range: Range,
    pub reservoir_size_range: Range,
    pub reservoir_bias_scaling_range: Range,
    pub reservoir_sparsity_range: Range,
    pub reservoir_activation: Activation,
    pub spectral_radius: f64,
    pub leaking_rate_range: Range,
    pub regularization_coeff_range: Range,
    pub washout_pct: f64,
    pub output_activation: Activation,
    pub seed: Option<u64>,
    pub state_update_noise_frac: f64,
    pub initial_state_value: f64,
    pub readout_from_input_as_well: bool,
}
*/

/// Dimensionality of the parameter space
pub const PARAM_DIM: usize = 7;

/*
impl OptParamMapper<PARAM_DIM> for ParamMapper {
    type Params = Params;

    fn map(&self, params: &[f64; PARAM_DIM]) -> Params {
        Params {
            input_sparsity: scale(
                0.0,
                1.0,
                self.input_sparsity_range.0,
                self.input_sparsity_range.1,
                params[0],
            ),
            input_activation: self.input_activation,
            input_weight_scaling: scale(
                0.0,
                1.0,
                self.input_weight_scaling_range.0,
                self.input_weight_scaling_range.1,
                params[1],
            ),
            reservoir_size: scale(
                0.0,
                1.0,
                self.reservoir_size_range.0,
                self.reservoir_size_range.1,
                params[2],
            ) as usize,
            reservoir_bias_scaling: scale(
                0.0,
                1.0,
                self.reservoir_bias_scaling_range.0,
                self.reservoir_bias_scaling_range.1,
                params[3],
            ),
            reservoir_sparsity: scale(
                0.0,
                1.0,
                self.reservoir_sparsity_range.0,
                self.reservoir_sparsity_range.1,
                params[4],
            ),
            reservoir_activation: self.reservoir_activation,
            spectral_radius: self.spectral_radius,
            leaking_rate: scale(
                0.0,
                1.0,
                self.leaking_rate_range.0,
                self.leaking_rate_range.1,
                params[5],
            ),
            regularization_coeff: scale(
                0.0,
                1.0,
                self.regularization_coeff_range.0,
                self.regularization_coeff_range.1,
                params[6],
            ),
            washout_pct: self.washout_pct,
            output_activation: self.output_activation,
            seed: self.seed,
            state_update_noise_frac: self.state_update_noise_frac,
            initial_state_value: self.initial_state_value,
            readout_from_input_as_well: self.readout_from_input_as_well,
        }
    }
}
*/

/// The Reseoir Computer, Leaky Echo State Network
#[derive(Debug)]
pub struct ESN<R> {
    params: Params,
    input_weight_matrix: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    reservoir_matrix: DMatrix<f64>,
    reservoir_biases: StateMatrix,
    readout_matrix: DMatrix<f64>,
    state: StateMatrix,
    extended_state: StateMatrix,
    rng: WyRand,
    regressor: R,
}

impl<R> ESN<R>
where
    R: LinReg,
{
    // type ParamMapper = ParamMapper;

    /// Create a new reservoir, with random initiallization
    /// # Arguments
    fn new(params: Params, regressor: R) -> Self {
        let mut rng = match params.seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };
        let mut weights: Vec<Vec<f64>> =
            vec![vec![0.0; params.reservoir_size]; params.reservoir_size];
        for i in 0..weights.len() {
            for j in 0..weights.len() {
                if rng.generate::<f64>() < params.reservoir_sparsity {
                    weights[i][j] = rng.generate::<f64>() * 2.0 - 1.0;
                }
            }
        }
        let mut reservoir_matrix: DMatrix<f64> = DMatrix::from_vec_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(params.reservoir_size),
            weights.iter().cloned().flatten().collect(),
        );

        let eigen = SymmetricEigen::new(reservoir_matrix.clone());
        let spec_rad = eigen.eigenvalues.abs().max();
        reservoir_matrix *= (1.0 / spec_rad) * params.spectral_radius;

        let input_weight_matrix = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            |_, _| {
                if rng.generate::<f64>() < params.input_sparsity {
                    (rng.generate::<f64>() * 2.0 - 1.0) * params.input_weight_scaling
                } else {
                    0.0
                }
            },
        );
        let reservoir_biases = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            |_, _| (rng.generate::<f64>() * 2.0 - 1.0) * params.reservoir_bias_scaling,
        );

        let cols = if params.readout_from_input_as_well {
            1 + params.reservoir_size
        } else {
            params.reservoir_size
        };
        let readout_matrix =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(cols), |_, _| {
                rng.generate::<f64>() * 2.0 - 1.0
            });
        let state = Matrix::from_element_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            params.initial_state_value,
        );
        let extended_state = Matrix::from_element_generic(
            Dim::from_usize(1 + params.reservoir_size),
            Dim::from_usize(1),
            params.initial_state_value,
        );
        trace!(
            "input_matrix: {}\nreservoir: {}\nreadout_matrix: {}",
            input_weight_matrix,
            reservoir_matrix,
            readout_matrix
        );

        Self {
            params,
            reservoir_matrix,
            input_weight_matrix,
            readout_matrix,
            state,
            reservoir_biases,
            rng,
            extended_state,
            regressor,
        }
    }

    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) {
        let washout_len = (inputs.ncols() as f64 * self.params.washout_pct) as usize;
        let harvest_len = inputs.ncols() - washout_len;

        let design_cols = if self.params.readout_from_input_as_well {
            1 + self.params.reservoir_size
        } else {
            self.params.reservoir_size
        };
        let mut design_matrix: DMatrix<f64> = DMatrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(design_cols),
            0.0,
        );
        let mut target_matrix: DMatrix<f64> =
            Matrix::from_element_generic(Dim::from_usize(harvest_len), Dim::from_usize(1), 0.0);
        let mut curr_pred = self.readout();

        for i in 0..inputs.nrows() {
            self.update_state(&inputs.row(i));

            curr_pred = self.readout();

            // discard earlier values, as the state has to stabilize first
            if i >= washout_len {
                let design: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                    Matrix::from_fn_generic(
                        Dim::from_usize(1),
                        Dim::from_usize(design_cols),
                        |_, j| {
                            if self.params.readout_from_input_as_well {
                                *self.extended_state.get(j).unwrap()
                            } else {
                                *self.state.get(j).unwrap()
                            }
                        },
                    );
                design_matrix.set_row(i - washout_len, &design);

                let target_col = targets.column(i);
                let target: Matrix<f64, Const<1>, Const<1>, ArrayStorage<f64, 1, 1>> =
                    Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |_, j| {
                        *target_col.get(j).unwrap()
                    });
                target_matrix.set_row(i - washout_len, &target);
            }
        }

        self.readout_matrix = self.regressor.fit_readout(
            &design_matrix.rows(0, design_matrix.nrows()),
            &target_matrix.rows(0, target_matrix.nrows()),
        );

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
        // perform node-to-node update
        let noise: StateMatrix = Matrix::from_fn_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(1),
            |_, _| (self.rng.generate::<f64>() * 2.0 - 1.0) * self.params.state_update_noise_frac,
        );
        let mut state_delta: StateMatrix = &self.input_weight_matrix * input
            + self.params.leaking_rate * (&self.reservoir_matrix * &self.state)
            + &self.reservoir_biases
            + noise;
        self.params.reservoir_activation.activate(state_delta.as_mut_slice());

        self.state = (1.0 - self.params.leaking_rate) * &self.state + state_delta;
        self.extended_state = Matrix::from_fn_generic(
            Dim::from_usize(1 + self.params.reservoir_size),
            Dim::from_usize(1),
            |i, _| {
                if i == 0 {
                    *input.get(0).unwrap()
                } else {
                    *self.state.row(i - 1).get(0).unwrap()
                }
            },
        );
    }

    /// Perform a readout operation
    #[inline]
    #[must_use]
    fn readout(&self) -> Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> {
        let mut pred = if self.params.readout_from_input_as_well {
            &self.extended_state * &self.readout_matrix
        } else {
            &self.state * &self.readout_matrix
        };
        self.params.output_activation.activate(pred.as_mut_slice());

        pred
    }

    /// Resets the state to it's initial values
    #[inline(always)]
    fn set_state(&mut self, state: StateMatrix) {
        self.state = state;
        self.extended_state = Matrix::from_element_generic(
            Dim::from_usize(1 + self.params.reservoir_size),
            Dim::from_usize(1),
            self.params.initial_state_value,
        );
    }

    #[inline(always)]
    fn params(&self) -> &Params {
        &self.params
    }

    #[inline(always)]
    fn readout_matrix(&self) -> &DMatrix<f64> {
        &self.readout_matrix
    }
}
