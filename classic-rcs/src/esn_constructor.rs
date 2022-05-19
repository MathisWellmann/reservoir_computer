use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, SymmetricEigen, VecStorage};
use nanorand::{Rng, WyRand};

use crate::{ReservoirConstructor, StateMatrix};

/// Constructs the weights of a classic Echo State Network
pub struct ESNConstructor {
    /// Controls the retention of information from previous time steps.
    /// The spectral radius determines how fast the influence of an input
    /// dies out in a reservoir with time, and how stable the reservoir
    /// activations are. The spectral radius should be greater in tasks
    /// requiring longer memory of the input.
    spectral_radius: f64,

    /// The number of nodes in the reservoir
    reservoir_size: usize,

    /// How sparsly connected the reservoir will be
    reservoir_sparsity: f64,

    /// Scales the randomly generated biases
    reservoir_bias_scaling: f64,

    /// Probability of inputs connecting to state
    input_sparsity: f64,

    /// Scales the input weights
    input_weight_scaling: f64,

    rng: WyRand,
}

impl ESNConstructor {
    pub fn new(
        seed: Option<u64>,
        reservoir_size: usize,
        spectral_radius: f64,
        reservoir_sparsity: f64,
        reservoir_bias_scaling: f64,
        input_sparsity: f64,
        input_weight_scaling: f64,
    ) -> Self {
        let rng = match seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };

        Self {
            spectral_radius,
            reservoir_size,
            reservoir_sparsity,
            rng,
            reservoir_bias_scaling,
            input_sparsity,
            input_weight_scaling,
        }
    }
}

impl ReservoirConstructor for ESNConstructor {
    fn construct_reservoir_weights(&mut self) -> DMatrix<f64> {
        let mut weights: Vec<Vec<f64>> = vec![vec![0.0; self.reservoir_size]; self.reservoir_size];
        for i in 0..weights.len() {
            for j in 0..weights.len() {
                if self.rng.generate::<f64>() < self.reservoir_sparsity {
                    weights[i][j] = self.rng.generate::<f64>() * 2.0 - 1.0;
                }
            }
        }
        let mut reservoir_matrix: DMatrix<f64> = DMatrix::from_vec_generic(
            Dim::from_usize(self.reservoir_size),
            Dim::from_usize(self.reservoir_size),
            weights.iter().cloned().flatten().collect(),
        );

        let eigen = SymmetricEigen::new(reservoir_matrix.clone());
        let spec_rad = eigen.eigenvalues.abs().max();
        reservoir_matrix *= (1.0 / spec_rad) * self.spectral_radius;

        reservoir_matrix
    }

    fn construct_reservoir_biases(&mut self) -> StateMatrix {
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
