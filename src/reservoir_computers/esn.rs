use nalgebra::{
    ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, SymmetricEigen, VecStorage,
};
use nanorand::{Rng, WyRand};

use super::{RCParams, ReservoirComputer, StateMatrix};
use crate::activation::Activation;

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

    /// Influence of feedback computation
    pub feedback_gain: f64,

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
}

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

/// The Reseoir Computer, Leaky Echo State Network
#[derive(Debug)]
pub struct ESN<const I: usize, const O: usize> {
    params: Params,
    input_weight_matrix: Matrix<f64, Dynamic, Const<I>, VecStorage<f64, Dynamic, Const<I>>>,
    reservoir_matrix: DMatrix<f64>,
    reservoir_biases: StateMatrix,
    readout_matrix: Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    feedback_matrix: Matrix<f64, Dynamic, Const<O>, VecStorage<f64, Dynamic, Const<O>>>,
    state: StateMatrix,
    extended_state: StateMatrix,
    rng: WyRand,
}

impl<const I: usize, const O: usize> ReservoirComputer<Params, I, O> for ESN<I, O> {
    /// Create a new reservoir, with random initiallization
    /// # Arguments
    fn new(params: Params) -> Self {
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
            Dim::from_usize(I),
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

        let readout_matrix = Matrix::from_fn_generic(
            Dim::from_usize(O),
            Dim::from_usize(1 + params.reservoir_size),
            |_, _| rng.generate::<f64>() * 2.0 - 1.0,
        );
        let feedback_matrix: Matrix<f64, Dynamic, Const<O>, VecStorage<f64, Dynamic, Const<O>>> =
            Matrix::from_fn_generic(
                Dim::from_usize(params.reservoir_size),
                Dim::from_usize(O),
                |_, _| {
                    // TODO: input_sparsity should maybe be feedback_sparsity
                    if rng.generate::<f64>() < params.input_sparsity {
                        rng.generate::<f64>() * params.feedback_gain
                    } else {
                        0.0
                    }
                },
            );
        let state = Matrix::from_element_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            params.initial_state_value,
        );
        let extended_state = Matrix::from_element_generic(
            Dim::from_usize(I + params.reservoir_size),
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
            feedback_matrix,
            reservoir_biases,
            rng,
            extended_state,
        }
    }

    fn train(
        &mut self,
        inputs: &Matrix<f64, Const<I>, Dynamic, VecStorage<f64, Const<I>, Dynamic>>,
        targets: &Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    ) {
        let washout_len = (inputs.ncols() as f64 * self.params.washout_pct) as usize;
        let harvest_len = inputs.ncols() - washout_len;

        let mut design_matrix: DMatrix<f64> = DMatrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(I + self.params.reservoir_size),
            0.0,
        );
        let mut target_matrix: Matrix<f64, Dynamic, Const<O>, VecStorage<f64, Dynamic, Const<O>>> =
            Matrix::from_element_generic(Dim::from_usize(harvest_len), Dim::from_usize(O), 0.0);
        let mut curr_pred = self.readout();

        for j in 0..inputs.ncols() {
            self.update_state(&inputs.column(j), &curr_pred);

            curr_pred = self.readout();

            // discard earlier values, as the state has to stabilize first
            if j >= washout_len {
                let design: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                    Matrix::from_fn_generic(
                        Dim::from_usize(1),
                        Dim::from_usize(I + self.params.reservoir_size),
                        |_, j| *self.extended_state.get(j).unwrap(),
                    );
                design_matrix.set_row(j - washout_len, &design);

                let target_col = targets.column(j);
                let target: Matrix<f64, Const<1>, Const<O>, ArrayStorage<f64, 1, O>> =
                    Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(O), |_, j| {
                        *target_col.get(j).unwrap()
                    });
                target_matrix.set_row(j - washout_len, &target);
            }
        }

        let reg_m: DMatrix<f64> = DMatrix::from_diagonal_element_generic(
            Dim::from_usize(I + self.params.reservoir_size),
            Dim::from_usize(I + self.params.reservoir_size),
            self.params.regularization_coeff,
        );

        let qr = design_matrix.qr();
        let a = qr.r().try_inverse().unwrap() * qr.q().transpose();
        let b = a * &target_matrix;
        self.readout_matrix = b.transpose();
    }

    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<I>, Const<1>, Const<1>, Const<I>>,
        prev_pred: &Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>>,
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
            + (&self.feedback_matrix * prev_pred)
            + noise;
        self.params.reservoir_activation.activate(state_delta.as_mut_slice());

        self.state = (1.0 - self.params.leaking_rate) * &self.state + state_delta;
        self.extended_state = Matrix::from_fn_generic(
            Dim::from_usize(I + self.params.reservoir_size),
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
    fn readout(&self) -> Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>> {
        let mut pred = &self.readout_matrix * &self.extended_state;
        self.params.output_activation.activate(pred.as_mut_slice());

        pred
    }

    /// Resets the state to it's initial values
    #[inline(always)]
    fn set_state(
        &mut self,
        state: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    ) {
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
    fn readout_matrix(
        &self,
    ) -> &Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>> {
        &self.readout_matrix
    }
}
