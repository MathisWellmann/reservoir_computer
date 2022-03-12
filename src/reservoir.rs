use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, SymmetricEigen, VecStorage};
use nanorand::{Rng, WyRand};

/// The Reseoir Computer
/// N is the number of reservoir nodes
/// M is the input and output dimension
#[derive(Debug)]
pub(crate) struct ReservoirComputer {
    reservoir_size: usize,
    leaking_rate: f32,
    input_bias: f32,
    input_matrix: Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>>,
    reservoir: DMatrix<f32>,
    readout_matrix: Matrix<f32, Const<1>, Dynamic, VecStorage<f32, Const<1>, Dynamic>>,
    state: Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>>,
}

impl ReservoirComputer {
    /// Create a new reservoir, with random initiallization
    /// # Arguments
    /// reservoir_size: number of nodes in the reservoir
    /// fixed_in_degree_k: number of inputs per node
    /// input_sparsity: the connection probability within the reservoir
    /// input_scaling: multiplies the input weights
    /// input_bias: adds a bias to inputs
    /// leaking_rate:
    ///     The leaking rate a can be regarded as the speed of the reservoir
    /// update     dynamics discretized in time. This can be adapted online
    /// to deal with     time wrapping of the signals. Set the leaking rate
    /// to match the speed of     the dynamics of input / target This sort
    /// of acts as a EMA smoothing     filter, so other MAs could be used
    /// such as ALMA spectral_radius:
    ///     The spectral radius determines how fast the influence of an input
    ///     dies out in a reservoir with time, and how stable the reservoir
    /// activations     are. The spectral radius should be greater in tasks
    /// requiring longer     memory of the input.
    /// seed: optional RNG seed
    pub(crate) fn new(
        reservoir_size: usize,
        fixed_in_degree_k: usize,
        input_sparsity: f32,
        input_scaling: f32,
        input_bias: f32,
        spectral_radius: f32,
        leaking_rate: f32,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = match seed {
            Some(seed) => WyRand::new_seed(seed),
            None => WyRand::new(),
        };
        let mut weights: Vec<Vec<f32>> = vec![vec![0.0; reservoir_size]; reservoir_size];
        for j in 0..weights.len() {
            for _ in 0..fixed_in_degree_k {
                // Choose random input node
                let i = rng.generate_range(0..reservoir_size);
                weights[i][j] = rng.generate::<f32>() * 2.0 - 1.0;
            }
        }
        let mut reservoir: DMatrix<f32> = DMatrix::from_vec_generic(
            Dim::from_usize(reservoir_size),
            Dim::from_usize(reservoir_size),
            weights.iter().cloned().flatten().collect(),
        );
        debug!("reservoir: {}", reservoir);

        let eigen = SymmetricEigen::new(reservoir.clone());
        let spec_rad = eigen.eigenvalues.abs().max();
        reservoir *= (1.0 / spec_rad) * spectral_radius;

        let input_matrix: Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(reservoir_size), Dim::from_usize(1), |_, _| {
                if rng.generate::<f32>() < input_sparsity {
                    rng.generate::<f32>() * input_scaling
                } else {
                    0.0
                }
            });
        let mut readout_matrix: Matrix<f32, Const<1>, Dynamic, VecStorage<f32, Const<1>, Dynamic>> =
            Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(reservoir_size), |_, _| {
                rng.generate::<f32>() * 2.0 - 1.0
            });
        let mut state: Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>> =
            Matrix::from_fn_generic(Dim::from_usize(reservoir_size), Dim::from_usize(1), |_, _| {
                rng.generate::<f32>() * 2.0 - 1.0
            });
        debug!(
            "input_matrix: {}\nreservoir: {}\nreadout_matrix: {}\nstate: {}",
            input_matrix, reservoir, readout_matrix, state
        );

        Self {
            reservoir,
            input_matrix,
            readout_matrix,
            state,
            reservoir_size,
            leaking_rate,
            input_bias,
        }
    }

    pub(crate) fn train(&mut self, values: &[f32]) {
        let mut step_wise_design: DMatrix<f32> = DMatrix::from_fn_generic(
            Dim::from_usize(self.reservoir_size),
            Dim::from_usize(values.len()),
            |_, _| 0.0,
        );
        let mut step_wise_predictions: Matrix<
            f32,
            Const<1>,
            Dynamic,
            VecStorage<f32, Const<1>, Dynamic>,
        > = Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(values.len()), |_, _| 0.0);
        let step_wise_target: Matrix<f32, Const<1>, Dynamic, VecStorage<f32, Const<1>, Dynamic>> =
            Matrix::from_vec_generic(
                Dim::from_usize(1),
                Dim::from_usize(values.len()),
                values.to_vec(),
            );
        for (j, val) in values.iter().enumerate() {
            let predicted_out = &self.readout_matrix * &self.state;
            step_wise_predictions.set_column(j, &predicted_out);

            step_wise_design.set_column(j, &self.state);

            let a = (1.0 - self.leaking_rate) * &self.state;
            let unit_vec: Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>> =
                Matrix::from_element_generic(
                    Dim::from_usize(self.reservoir_size),
                    Dim::from_usize(1),
                    1.0,
                );
            let mut b = &self.reservoir * &self.state
                + &self.input_matrix * *val
                + self.input_bias * unit_vec;
            b.iter_mut().for_each(|v| *v = v.tanh());
            self.state = a + self.leaking_rate * b;
        }

        // compute optimal readout matrix
        let design_t = step_wise_design.transpose();
        // Use regularizaion whenever there is a danger of overfitting or feedback
        // instability
        let regularization_coeff: f32 = 0.0;
        let identity_m: DMatrix<f32> = DMatrix::from_diagonal_element_generic(
            Dim::from_usize(self.reservoir_size),
            Dim::from_usize(self.reservoir_size),
            1.0,
        );
        let b = step_wise_design * &design_t + regularization_coeff * identity_m;
        let b = b.transpose();
        self.readout_matrix = step_wise_target * &design_t * b;
    }

    pub(crate) fn readout_matrix(
        &self,
    ) -> &Matrix<f32, Const<1>, Dynamic, VecStorage<f32, Const<1>, Dynamic>> {
        &self.readout_matrix
    }

    pub(crate) fn reservoir(&self) -> &DMatrix<f32> {
        &self.reservoir
    }

    pub(crate) fn input_matrix(
        &self,
    ) -> &Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>> {
        &self.input_matrix
    }

    pub(crate) fn state(
        &self,
    ) -> Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>> {
        self.state.clone()
    }
}
