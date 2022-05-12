use nalgebra::{ArrayStorage, Const, DMatrix, Dim, Dynamic, Matrix, MatrixSlice, VecStorage};
use nanorand::{Rng, WyRand};

use super::{OptParamMapper, RCParams, Range, ReservoirComputer, StateMatrix};
use crate::{activation::Activation, lin_reg::LinReg, utils::scale};

// TODO: value validation
#[derive(Debug, Clone)]
pub struct Params {
    pub input_sparsity: f64,
    pub input_weight_scaling: f64,

    pub reservoir_size: usize,
    pub reservoir_weight_scaling: f64,
    pub reservoir_bias_scaling: f64,
    pub reservoir_activation: Activation,

    pub initial_state_value: f64,

    pub seed: Option<u64>,
    pub washout_frac: f64,
    pub regularization_coeff: f64,

    /// step size of integration
    pub epsilon: f64,
    /// diffusion coeffient used for stabilizing the discrete forward
    /// propagation
    pub gamma: f64,
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

pub struct ParamMapper {
    pub input_sparsity_range: Range,
    pub input_weight_scaling_range: Range,
    pub reservoir_size_range: Range,
    pub reservoir_weight_scaling_range: Range,
    pub reservoir_bias_scaling_range: Range,
    pub reservoir_activation: Activation,
    pub initial_state_value: f64,
    pub seed: Option<u64>,
    pub washout_frac: f64,
    pub regularization_coeff: f64,
    pub epsilon_range: Range,
    pub gamma_range: Range,
}

const PARAM_DIM: usize = 7;

impl OptParamMapper<PARAM_DIM> for ParamMapper {
    type Params = Params;

    fn map(&self, params: &[f64; PARAM_DIM]) -> Self::Params {
        Params {
            input_sparsity: scale(
                0.0,
                1.0,
                self.input_sparsity_range.0,
                self.input_sparsity_range.1,
                params[0],
            ),
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
            reservoir_weight_scaling: scale(
                0.0,
                1.0,
                self.reservoir_weight_scaling_range.0,
                self.reservoir_weight_scaling_range.1,
                params[3],
            ),
            reservoir_bias_scaling: scale(
                0.0,
                1.0,
                self.reservoir_bias_scaling_range.0,
                self.reservoir_bias_scaling_range.1,
                params[4],
            ),
            reservoir_activation: self.reservoir_activation,
            initial_state_value: self.initial_state_value,
            seed: self.seed,
            washout_frac: self.washout_frac,
            regularization_coeff: self.regularization_coeff,
            epsilon: scale(0.0, 1.0, self.epsilon_range.0, self.epsilon_range.1, params[5]),
            gamma: scale(0.0, 1.0, self.gamma_range.0, self.gamma_range.1, params[6]),
        }
    }
}

/// Euler State Network (EuSN)
/// from: https://arxiv.org/pdf/2203.09382.pdf
pub struct EulerStateNetwork<const I: usize, const O: usize, R> {
    params: Params,
    state: StateMatrix,
    input_weight_matrix: Matrix<f64, Dynamic, Const<I>, VecStorage<f64, Dynamic, Const<I>>>,
    reservoir_weight_matrix: DMatrix<f64>,
    reservoir_biases: StateMatrix,
    readout_matrix: Matrix<f64, Const<O>, Dynamic, VecStorage<f64, Const<O>, Dynamic>>,
    regressor: R,
}

impl<const I: usize, const O: usize, R> ReservoirComputer<I, O, PARAM_DIM, R>
    for EulerStateNetwork<I, O, R>
where
    R: LinReg,
{
    type ParamMapper = ParamMapper;

    /// Create a new untrained EuSN with the given parameters
    fn new(params: Params, regressor: R) -> Self {
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
                weights[i][j] =
                    (rng.generate::<f64>() * 2.0 - 1.0) * params.reservoir_weight_scaling;
            }
        }
        let mut reservoir_matrix: DMatrix<f64> = DMatrix::from_vec_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(params.reservoir_size),
            weights.iter().cloned().flatten().collect(),
        );
        let identity_m: DMatrix<f64> = DMatrix::from_diagonal_element_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(params.reservoir_size),
            1.0,
        );
        // This satisfies the constraint of being anti-symmetric
        reservoir_matrix =
            (&reservoir_matrix - reservoir_matrix.transpose()) - (params.gamma * identity_m);

        let reservoir_biases = Matrix::from_fn_generic(
            Dim::from_usize(params.reservoir_size),
            Dim::from_usize(1),
            |_, _| (rng.generate::<f64>() * 2.0 - 1.0) * params.reservoir_bias_scaling,
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
            Dim::from_usize(O),
            Dim::from_usize(params.reservoir_size),
            |_, _| rng.generate::<f64>() * 2.0 - 1.0,
        );

        Self {
            state,
            params,
            input_weight_matrix,
            reservoir_weight_matrix: reservoir_matrix,
            reservoir_biases,
            readout_matrix,
            regressor,
        }
    }

    /// Train the EuSN with the given inputs and targets
    fn train<'a>(
        &mut self,
        inputs: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
        targets: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) {
        let washout_len = (inputs.ncols() as f64 * self.params.washout_frac) as usize;
        let harvest_len = inputs.ncols() - washout_len;

        let mut design_matrix: DMatrix<f64> = DMatrix::from_element_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(self.params.reservoir_size),
            0.0,
        );
        let mut target_matrix: Matrix<f64, Dynamic, Const<O>, VecStorage<f64, Dynamic, Const<O>>> =
            Matrix::from_element_generic(Dim::from_usize(harvest_len), Dim::from_usize(1), 0.0);
        let curr_pred = self.readout();
        for j in 0..inputs.ncols() {
            self.update_state(&inputs.column(j), &curr_pred);

            // discard earlier values, as the state has to stabilize first
            if j >= washout_len {
                let design: Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> =
                    Matrix::from_fn_generic(
                        Dim::from_usize(1),
                        Dim::from_usize(self.params.reservoir_size),
                        |_, j| *self.state.get(j).unwrap(),
                    );
                design_matrix.set_row(j - washout_len, &design);
                let target_col = targets.column(j);
                let target: Matrix<f64, Const<1>, Const<O>, ArrayStorage<f64, 1, O>> =
                    Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(O), |i, _j| {
                        *target_col.get(i).unwrap()
                    });
                target_matrix.set_row(j - washout_len, &target);
            }
        }

        // Ridge regression regularization, I think
        let x: DMatrix<f64> = Matrix::from_fn_generic(
            Dim::from_usize(harvest_len),
            Dim::from_usize(self.params.reservoir_size),
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
    }

    /// Propagate an input through the network and update its state
    fn update_state<'a>(
        &mut self,
        input: &'a MatrixSlice<'a, f64, Const<I>, Const<1>, Const<1>, Const<I>>,
        _prev_pred: &Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>>,
    ) {
        let mut nonlinear = (&self.reservoir_weight_matrix * &self.state)
            + (&self.input_weight_matrix * input)
            + &self.reservoir_biases;
        self.params.reservoir_activation.activate(nonlinear.as_mut_slice());
        self.state = &self.state + self.params.epsilon * nonlinear
    }

    #[inline(always)]
    fn readout(&self) -> Matrix<f64, Const<O>, Const<1>, ArrayStorage<f64, O, 1>> {
        &self.readout_matrix * &self.state
    }

    /// Resets the state to it's initial values
    #[inline(always)]
    fn set_state(
        &mut self,
        state: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    ) {
        self.state = Matrix::from_fn_generic(
            Dim::from_usize(self.params.reservoir_size),
            Dim::from_usize(1),
            |i, _| *state.get(i).unwrap(),
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
