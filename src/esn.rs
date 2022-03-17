use nalgebra::{
    ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, SymmetricEigen, VecStorage,
};
use nanorand::{Rng, WyRand};

use crate::{activation::Activation, INPUT_DIM, OUTPUT_DIM};

pub(crate) type Inputs =
    Matrix<f64, Dynamic, Const<INPUT_DIM>, VecStorage<f64, Dynamic, Const<INPUT_DIM>>>;
pub(crate) type Targets =
    Matrix<f64, Dynamic, Const<OUTPUT_DIM>, VecStorage<f64, Dynamic, Const<INPUT_DIM>>>;
pub(crate) type StateMatrix = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;
pub(crate) type ReadoutMatrix =
    Matrix<f64, Dynamic, Const<OUTPUT_DIM>, VecStorage<f64, Dynamic, Const<OUTPUT_DIM>>>;
pub(crate) type Output = Matrix<f64, Const<1>, Const<1>, ArrayStorage<f64, 1, 1>>;

/// reservoir_size: number of nodes in the reservoir
/// fixed_in_degree_k: number of inputs per node
/// input_sparsity: the connection probability within the reservoir
/// input_scaling: multiplies the input weights
/// input_bias: adds a bias to inputs
///
/// leaking_rate:
/// The leaking rate a can be regarded as the speed of the reservoir
/// update dynamics discretized in time. This can be adapted online
/// to deal with time wrapping of the signals. Set the leaking rate
/// to match the speed of the dynamics of input / target This sort
/// of acts as a EMA smoothing filter, so other MAs could be used
/// such as ALMA
///
/// spectral_radius:
/// The spectral radius determines how fast the influence of an input
/// dies out in a reservoir with time, and how stable the reservoir
/// activations are. The spectral radius should be greater in tasks
/// requiring longer memory of the input.
///
/// regularization_coeff:
/// seed: optional RNG seed
#[derive(Debug, Clone)]
pub(crate) struct Params {
    pub(crate) input_sparsity: f64,
    pub(crate) input_activation: Activation,
    pub(crate) input_weight_scaling: f64,
    pub(crate) input_bias_scaling: f64,

    pub(crate) reservoir_size: usize,
    pub(crate) reservoir_fixed_in_degree_k: usize,
    pub(crate) reservoir_activation: Activation,

    pub(crate) feedback_gain: f64,
    pub(crate) spectral_radius: f64,
    pub(crate) leaking_rate: f64,
    pub(crate) regularization_coeff: f64,
    pub(crate) washout_pct: f64,
    pub(crate) output_tanh: bool,
    pub(crate) seed: Option<u64>,
    pub(crate) state_update_noise_frac: f64,
    pub(crate) initial_state_value: f64,
}

/// The Reseoir Computer, Leaky Echo State Network
pub(crate) struct ESN {
    params: Params,
    input_weight_matrix:
        Matrix<f64, Dynamic, Const<INPUT_DIM>, VecStorage<f64, Dynamic, Const<INPUT_DIM>>>,
    input_biases: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    reservoir_matrix: DMatrix<f64>,
    readout_matrix: ReadoutMatrix,
    feedback_matrix:
        Matrix<f64, Dynamic, Const<OUTPUT_DIM>, VecStorage<f64, Dynamic, Const<OUTPUT_DIM>>>,
    state: StateMatrix,
    rng: WyRand,
}

impl ESN {
    /// Create a new reservoir, with random initiallization
    /// # Arguments
    pub(crate) fn new(params: Params) -> Self {
        let mut rng = match params.seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };
        let mut weights: Vec<Vec<f64>> =
            vec![vec![0.0; params.reservoir_size]; params.reservoir_size];
        for j in 0..weights.len() {
            for _ in 0..params.reservoir_fixed_in_degree_k {
                // Choose random input node
                let i = rng.generate_range(0..params.reservoir_size);
                weights[i][j] = rng.generate::<f64>() * 2.0 - 1.0;
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

        let input_weight_matrix: Matrix<
            f64,
            Dynamic,
            Const<INPUT_DIM>,
            VecStorage<f64, Dynamic, Const<INPUT_DIM>>,
        > = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(INPUT_DIM),
            |_, _| {
                if rng.generate::<f64>() < params.input_sparsity {
                    (rng.generate::<f64>() * 2.0 - 1.0) * params.input_weight_scaling
                } else {
                    0.0
                }
            },
        );
        let input_biases: Matrix<
            f64,
            Dynamic,
            Const<INPUT_DIM>,
            VecStorage<f64, Dynamic, Const<1>>,
        > = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            |_, _| (rng.generate::<f64>() * 2.0 - 1.0) * params.input_bias_scaling,
        );

        let readout_matrix = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(OUTPUT_DIM),
            |_, _| rng.generate::<f64>() * 2.0 - 1.0,
        );
        let feedback_matrix: Matrix<
            f64,
            Dynamic,
            Const<OUTPUT_DIM>,
            VecStorage<f64, Dynamic, Const<OUTPUT_DIM>>,
        > = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(OUTPUT_DIM),
            |_, _| {
                // TODO: input_sparsity should maybe be feedback_sparsity
                if rng.generate::<f64>() < params.input_sparsity {
                    rng.generate::<f64>() * params.feedback_gain
                } else {
                    0.0
                }
            },
        );
        let state: StateMatrix = Matrix::from_element_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            params.initial_state_value,
        );
        info!(
            "input_matrix: {}\nreservoir: {}\nreadout_matrix: {}\nstate: {}",
            input_weight_matrix, reservoir_matrix, readout_matrix, state
        );

        Self {
            params,
            reservoir_matrix,
            input_weight_matrix,
            readout_matrix,
            state,
            feedback_matrix,
            input_biases,
            rng,
        }
    }

    pub(crate) fn train(&mut self, inputs: &Inputs, targets: &Targets) {
        let washout_len = (inputs.nrows() as f64 * self.params.washout_pct) as usize;
        let harvest_len = inputs.nrows() - washout_len;

        let mut design_matrix: DMatrix<f64> = DMatrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(1 + self.params.reservoir_size),
            0.0,
        );
        let mut target_matrix: Matrix<
            f64,
            Dynamic,
            Const<OUTPUT_DIM>,
            VecStorage<f64, Dynamic, Const<OUTPUT_DIM>>,
        > = Matrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(OUTPUT_DIM),
            0.0,
        );
        let mut curr_pred = self.readout();
        for i in 0..inputs.nrows() {
            self.update_state(&inputs.row(i), &curr_pred);

            curr_pred = self.readout();

            // discard earlier values, as the state has to stabilize first
            if i >= washout_len {
                let design: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                    Matrix::from_fn_generic(
                        Dim::from_usize(1),
                        Dim::from_usize(1 + self.params.reservoir_size),
                        |_, j| {
                            if j == 0 {
                                1.0
                            } else {
                                *self.state.get(j - 1).unwrap()
                            }
                        },
                    );
                design_matrix.set_row(i - washout_len, &design);
                let target_col = targets.row(i);
                let target: Matrix<
                    f64,
                    Const<OUTPUT_DIM>,
                    Const<1>,
                    ArrayStorage<f64, OUTPUT_DIM, 1>,
                > = Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |i, _j| {
                    *target_col.get(i).unwrap()
                });
                target_matrix.set_row(i - washout_len, &target);
            }
        }

        let k = design_matrix.transpose() * &design_matrix;
        let identity_m: DMatrix<f64> = DMatrix::from_diagonal_element_generic(
            Dim::from_usize(1 + self.params.reservoir_size),
            Dim::from_usize(1 + self.params.reservoir_size),
            1.0,
        );
        let p = (k + self.params.regularization_coeff * identity_m).try_inverse().unwrap();
        let xTy = design_matrix.transpose() * &target_matrix;
        let readout_matrix = p * xTy;
        self.readout_matrix = Matrix::from_fn_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(OUTPUT_DIM),
            |i, _| *readout_matrix.get(i + 1).unwrap(),
        );

        info!("trained readout_matrix: {}", self.readout_matrix);
    }

    pub(crate) fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<1>, Const<INPUT_DIM>, Const<1>, Dynamic>,
        prev_pred: &Output,
    ) {
        // perform node-to-node update
        let noise: StateMatrix = Matrix::from_fn_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(1),
            |_, _| (self.rng.generate::<f64>() * 2.0 - 1.0) * self.params.state_update_noise_frac,
        );
        let mut state_delta = &self.input_weight_matrix * input
            + self.params.leaking_rate * (&self.reservoir_matrix * &self.state)
            + &self.input_biases
            + (&self.feedback_matrix * prev_pred)
            + noise;
        self.params.reservoir_activation.activate(state_delta.as_mut_slice());

        self.state = (1.0 - self.params.leaking_rate) * &self.state + state_delta;
    }

    /// Perform a readout operation
    #[inline]
    #[must_use]
    pub(crate) fn readout(&self) -> Output {
        let mut pred = self.readout_matrix.transpose() * &self.state;
        if self.params.output_tanh {
            pred.iter_mut().for_each(|v| *v = v.tanh());
        }

        pred
    }

    /// Resets the state to it's initial values
    pub(crate) fn reset_state(&mut self) {
        self.state = Matrix::from_element_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(1),
            self.params.initial_state_value,
        );
    }

    #[inline(always)]
    pub(crate) fn state(&self) -> &StateMatrix {
        &self.state
    }
}
