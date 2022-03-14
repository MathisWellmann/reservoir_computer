use nalgebra::{ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, SymmetricEigen, VecStorage};
use nanorand::{Rng, WyRand};

const INPUT_DIM: usize = 1;
const OUTPUT_DIM: usize = 1;

pub(crate) type StateMatrix = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;
pub(crate) type ExtendedStateMatrix =
    Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;
pub(crate) type PredictionMatrix =
    Matrix<f64, Const<OUTPUT_DIM>, Const<OUTPUT_DIM>, ArrayStorage<f64, OUTPUT_DIM, OUTPUT_DIM>>;

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
    pub(crate) reservoir_size: usize,
    pub(crate) fixed_in_degree_k: usize,
    pub(crate) input_sparsity: f64,
    pub(crate) input_scaling: f64,
    pub(crate) input_bias: f64,
    pub(crate) feedback_gain: f64,
    pub(crate) spectral_radius: f64,
    pub(crate) leaking_rate: f64,
    pub(crate) regularization_coeff: f64,
    pub(crate) washout_pct: f64,
    pub(crate) seed: Option<u64>,
}

/// The Reseoir Computer, Leaky Echo State Network
#[derive(Debug)]
pub(crate) struct ESN {
    params: Params,
    input_matrix:
        Matrix<f64, Dynamic, Const<INPUT_DIM>, VecStorage<f64, Dynamic, Const<INPUT_DIM>>>,
    reservoir_matrix: DMatrix<f64>,
    readout_matrix:
        Matrix<f64, Const<OUTPUT_DIM>, Dynamic, VecStorage<f64, Const<OUTPUT_DIM>, Dynamic>>,
    feedback_matrix:
        Matrix<f64, Dynamic, Const<OUTPUT_DIM>, VecStorage<f64, Dynamic, Const<OUTPUT_DIM>>>,
    state: StateMatrix,
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
            for _ in 0..params.fixed_in_degree_k {
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

        let input_matrix: Matrix<
            f64,
            Dynamic,
            Const<INPUT_DIM>,
            VecStorage<f64, Dynamic, Const<INPUT_DIM>>,
        > = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(INPUT_DIM),
            |_, _| {
                if rng.generate::<f64>() < params.input_sparsity {
                    rng.generate::<f64>() * params.input_scaling
                } else {
                    0.0
                }
            },
        );
        let readout_matrix: Matrix<
            f64,
            Const<OUTPUT_DIM>,
            Dynamic,
            VecStorage<f64, Const<OUTPUT_DIM>, Dynamic>,
        > = Matrix::from_fn_generic(
            Dim::from_usize(OUTPUT_DIM),
            Dim::from_usize(params.reservoir_size),
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
                    rng.generate::<f64>() * params.input_scaling
                } else {
                    0.0
                }
            },
        );
        let state: StateMatrix = Matrix::from_element_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            0.0,
        );
        info!(
            "input_matrix: {}\nreservoir: {}\nreadout_matrix: {}\nstate: {}",
            input_matrix, reservoir_matrix, readout_matrix, state
        );

        Self {
            params,
            reservoir_matrix,
            input_matrix,
            readout_matrix,
            state,
            feedback_matrix,
        }
    }

    pub(crate) fn train(&mut self, values: &[f64]) {
        let washout_len = (values.len() as f64 * self.params.washout_pct) as usize;
        let harvest_len = values.len() - washout_len;

        let mut step_wise_state: DMatrix<f64> = DMatrix::from_element_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(harvest_len),
            0.0,
        );
        let mut step_wise_predictions: Matrix<
            f64,
            Const<1>,
            Dynamic,
            VecStorage<f64, Const<1>, Dynamic>,
        > = Matrix::from_element_generic(Dim::from_usize(1), Dim::from_usize(harvest_len), 0.0);
        let mut step_wise_target: Matrix<
            f64,
            Const<1>,
            Dynamic,
            VecStorage<f64, Const<1>, Dynamic>,
        > = Matrix::from_element_generic(Dim::from_usize(1), Dim::from_usize(harvest_len), 0.0);
        let mut curr_pred = self.readout();
        for (j, (val_0, val_1)) in values.iter().zip(values.iter().skip(1)).enumerate() {
            self.update_state(*val_0, &curr_pred);

            curr_pred = self.readout();

            // discard earlier values, as the state has to stabilize first
            if j > washout_len {
                step_wise_state.set_column(j - washout_len, &self.state);
                let target: Matrix<
                    f64,
                    Const<OUTPUT_DIM>,
                    Const<OUTPUT_DIM>,
                    ArrayStorage<f64, OUTPUT_DIM, OUTPUT_DIM>,
                > = Matrix::from_element_generic(Dim::from_usize(1), Dim::from_usize(1), *val_1);
                step_wise_target.set_column(j - washout_len, &target);
                step_wise_predictions.set_column(j - washout_len, &curr_pred);
            }
        }

        // compute optimal readout matrix
        let state_t = step_wise_state.transpose();
        // Use regularizaion whenever there is a danger of overfitting or feedback
        // instability
        let identity_m: DMatrix<f64> = DMatrix::from_diagonal_element_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(self.params.reservoir_size),
            1.0,
        );
        let b = step_wise_state * &state_t + self.params.regularization_coeff * identity_m;
        self.readout_matrix = step_wise_target * (&state_t * b.transpose());
        info!("trained readout_matrix: {}", self.readout_matrix);
    }

    pub(crate) fn update_state(&mut self, input: f64, prev_pred: &PredictionMatrix) {
        assert!(input.is_finite());

        // TODO: optionally add noise to state update

        let mut new_state = (1.0 - self.params.leaking_rate) * &self.input_matrix * input
            + self.params.leaking_rate * &self.reservoir_matrix * &self.state
            + self.params.feedback_gain * &self.feedback_matrix * prev_pred;
        new_state.iter_mut().for_each(|v| *v = v.tanh());
        self.state = new_state;
    }

    /// Perform a readout operation
    #[inline]
    #[must_use]
    pub(crate) fn readout(&self) -> PredictionMatrix {
        let mut pred = &self.readout_matrix * &self.state;
        pred.iter_mut().for_each(|v| *v = v.tanh());

        pred
    }

    /// Resets the state to it's initial values
    pub(crate) fn reset_state(&mut self) {
        self.state = Matrix::from_element_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(1),
            0.0,
        );
    }
}
