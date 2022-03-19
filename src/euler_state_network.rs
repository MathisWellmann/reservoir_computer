
use nalgebra::{ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, SymmetricEigen, VecStorage};
use nanorand::{Rng, WyRand};

use crate::{activation::Activation, esn::{Inputs, Output, ReadoutMatrix, StateMatrix, Targets}};

// TODO: value validation
#[derive(Debug, Clone)]
pub(crate) struct EuSNParams {
    pub(crate) input_sparsity: f64,
    pub(crate) input_weight_scaling: f64,

    pub(crate) reservoir_size: usize,
    pub(crate) reservoir_weight_scaling: f64,
    pub(crate) reservoir_bias_scaling: f64,
    pub(crate) reservoir_activation: Activation,

    pub(crate) initial_state_value: f64,

    pub(crate) seed: Option<u64>,
    pub(crate) washout_frac: f64,
    pub(crate) regularization_coeff: f64,

    /// step size of integration
    pub(crate) epsilon: f64,
    /// diffusion coeffient used for stabilizing the discrete forward propagation
    pub(crate) gamma: f64,
}

/// Euler State Network (EuSN)
/// from: https://arxiv.org/pdf/2203.09382.pdf
pub(crate) struct EulerStateNetwork {
    params: EuSNParams,
    state: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    input_weight_matrix:
        Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    reservoir_weight_matrix: DMatrix<f64>,
    reservoir_biases: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    readout_matrix: ReadoutMatrix,
}

impl EulerStateNetwork {
    /// Create a new untrained EuSN with the given parameters
    pub(crate) fn new(params: EuSNParams) -> Self {
        let state = Matrix::from_element_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            params.initial_state_value,
        );

        let mut rng = match params.seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };
        let mut weights: Vec<Vec<f64>> =
            vec![vec![0.0; params.reservoir_size]; params.reservoir_size];
        for i in 0..weights.len() {
            for j in 0..weights.len() {
                weights[i][j] = (rng.generate::<f64>() * 2.0 - 1.0) * params.reservoir_weight_scaling;
            }
        }
        let mut reservoir_matrix: DMatrix<f64> = DMatrix::from_vec_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(params.reservoir_size),
            weights.iter().cloned().flatten().collect(),
        );
        let identity_m: DMatrix<f64> =
            DMatrix::from_diagonal_element_generic(
                Dim::from_usize(params.reservoir_size),
                Dim::from_usize(params.reservoir_size),
                1.0,
            );
        // This satisfies the constraint of being anti-symmetric
        reservoir_matrix = &reservoir_matrix - reservoir_matrix.transpose() - params.gamma * identity_m;

        let reservoir_biases = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            |_, _| (rng.generate::<f64>() * 2.0 - 1.0) * params.reservoir_bias_scaling
        );

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

        let readout_matrix = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            |_, _| rng.generate::<f64>() * 2.0 - 1.0,
        );

        Self {
            state,
            params,
            input_weight_matrix,
            reservoir_weight_matrix: reservoir_matrix,
            reservoir_biases,
            readout_matrix,
        }
    }

    /// Train the EuSN with the given inputs and targets
    pub(crate) fn train(
        &mut self,
        inputs: &Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
        targets: &Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    ) {
        let washout_len = (inputs.nrows() as f64 * self.params.washout_frac) as usize;
        let harvest_len = inputs.nrows() - washout_len;

        let mut design_matrix: DMatrix<f64> = DMatrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(1 + self.params.reservoir_size),
            0.0,
        );
        let mut target_matrix: Matrix<
            f64,
            Dynamic,
            Const<1>,
            VecStorage<f64, Dynamic, Const<1>>,
        > = Matrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(1),
            0.0,
        );
        for i in 0..inputs.nrows() {
            self.update_state(&inputs.row(i));

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
                    Const<1>,
                    Const<1>,
                    ArrayStorage<f64, 1, 1>,
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
            Dim::from_usize(1),
            |i, _| *readout_matrix.get(i + 1).unwrap(),
        );

        info!("trained readout_matrix: {}", self.readout_matrix);

    }

    /// Propagate an input through the network and update its state
    pub(crate) fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<1>, Const<1>, Const<1>, Dynamic>,
    ) {
        let mut nonlinear = ((&self.reservoir_weight_matrix)
            * &self.state)
            + (&self.input_weight_matrix * input)
            + &self.reservoir_biases;
        self.params.reservoir_activation.activate(nonlinear.as_mut_slice());
        self.state = &self.state
            + self.params.epsilon * nonlinear
    }

    #[inline(always)]
    pub(crate) fn readout(&self) -> Output {
        self.readout_matrix.transpose() * &self.state
    }

    /// Resets the state to it's initial values
    #[inline(always)]
    pub(crate) fn reset_state(&mut self) {
        self.state = Matrix::from_element_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(1),
            self.params.initial_state_value,
        );
    }
}
