use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, VecStorage};
use nanorand::{Rng, WyRand};

use crate::ReservoirConstructor;

pub struct EUSNConstructor {
    rng: WyRand,
    reservoir_size: usize,
    reservoir_weight_scaling: f64,
    reservoir_bias_scaling: f64,
    input_sparsity: f64,
    input_weight_scaling: f64,

    /// diffusion coeffient used for stabilizing the discrete forward
    /// propagation
    gamma: f64,
}

impl EUSNConstructor {
    pub fn new(
        seed: Option<u64>,
        reservoir_size: usize,
        reservoir_weight_scaling: f64,
        reservoir_bias_scaling: f64,
        input_sparsity: f64,
        input_weight_scaling: f64,
        gamma: f64,
    ) -> Self {
        let rng = match seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };

        Self {
            rng,
            reservoir_size,
            reservoir_weight_scaling,
            reservoir_bias_scaling,
            input_sparsity,
            input_weight_scaling,
            gamma,
        }
    }
}

impl ReservoirConstructor for EUSNConstructor {
    fn construct_reservoir_weights(&mut self) -> DMatrix<f64> {
        let mut weights: Vec<Vec<f64>> = vec![vec![0.0; self.reservoir_size]; self.reservoir_size];
        for i in 0..weights.len() {
            for j in 0..weights.len() {
                weights[i][j] =
                    (self.rng.generate::<f64>() * 2.0 - 1.0) * self.reservoir_weight_scaling;
            }
        }
        let mut reservoir_matrix: DMatrix<f64> = DMatrix::from_vec_generic(
            Dim::from_usize(self.reservoir_size),
            Dim::from_usize(self.reservoir_size),
            weights.iter().cloned().flatten().collect(),
        );
        let identity_m: DMatrix<f64> = DMatrix::from_diagonal_element_generic(
            Dim::from_usize(self.reservoir_size),
            Dim::from_usize(self.reservoir_size),
            1.0,
        );
        // This satisfies the constraint of being anti-symmetric
        reservoir_matrix =
            (&reservoir_matrix - reservoir_matrix.transpose()) - (self.gamma * identity_m);

        reservoir_matrix
    }

    fn construct_reservoir_biases(&mut self) -> crate::StateMatrix {
        Matrix::from_fn_generic(Dim::from_usize(self.reservoir_size), Dim::from_usize(1), |_, _| {
            (self.rng.generate::<f64>() * 2.0 - 1.0) * self.reservoir_bias_scaling
        })
    }

    fn construct_input_weight_matrix(
        &mut self,
    ) -> Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> {
        Matrix::from_fn_generic(Dim::from_usize(self.reservoir_size), Dim::from_usize(1), |_, _| {
            if self.rng.generate::<f64>() < self.input_sparsity {
                (self.rng.generate::<f64>() * 2.0 - 1.0) * self.input_weight_scaling
            } else {
                0.0
            }
        })
    }
}
