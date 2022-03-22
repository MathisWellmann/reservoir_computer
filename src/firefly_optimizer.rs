use std::time::Instant;

use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use nanorand::{Rng, WyRand};

use crate::{
    activation::Activation,
    esn::{EsnParams, Inputs, Targets, ESN},
    utils::scale,
};

const NUM_CANDIDATE_PARAMS: usize = 4;

pub struct ParameterMapper {
    param_ranges: Vec<(f64, f64)>,
    input_activation: Activation,
    reservoir_size: usize,
    reservoir_activation: Activation,
    leaking_rate: f64,
    regularization_coeff: f64,
    seed: Option<u64>,
    state_update_noise_frac: f64,
    initial_state_value: f64,
}

impl ParameterMapper {
    pub fn new(
        param_ranges: Vec<(f64, f64)>,
        input_activation: Activation,
        reservoir_size: usize,
        reservoir_activation: Activation,
        leaking_rate: f64,
        regularization_coeff: f64,
        seed: Option<u64>,
        state_update_noise_frac: f64,
        initial_state_value: f64,
    ) -> Self {
        assert_eq!(param_ranges.len(), NUM_CANDIDATE_PARAMS);

        Self {
            param_ranges,
            input_activation,
            reservoir_size,
            reservoir_activation,
            leaking_rate,
            regularization_coeff,
            seed,
            state_update_noise_frac,
            initial_state_value,
        }
    }

    pub fn map(&self, params: &Vec<f64>) -> EsnParams {
        assert_eq!(params.len(), self.param_ranges.len());

        EsnParams {
            input_sparsity: scale(
                0.0,
                1.0,
                self.param_ranges[0].0,
                self.param_ranges[0].1,
                params[0],
            ),
            input_activation: self.input_activation,
            input_weight_scaling: scale(
                0.0,
                1.0,
                self.param_ranges[1].0,
                self.param_ranges[1].1,
                params[1],
            ),
            reservoir_size: self.reservoir_size,
            reservoir_bias_scaling: scale(
                0.0,
                1.0,
                self.param_ranges[2].0,
                self.param_ranges[2].1,
                params[2],
            ),
            reservoir_fixed_in_degree_k: scale(
                0.0,
                1.0,
                self.param_ranges[3].0,
                self.param_ranges[3].1,
                params[3],
            ) as usize,
            reservoir_activation: self.reservoir_activation,
            feedback_gain: 0.0,
            spectral_radius: 0.9,
            leaking_rate: self.leaking_rate,
            regularization_coeff: self.regularization_coeff,
            washout_pct: 0.1,
            output_tanh: false,
            seed: self.seed,
            state_update_noise_frac: self.state_update_noise_frac,
            initial_state_value: self.initial_state_value,
        }
    }
}

pub struct FireflyParams {
    /// Influces the clustering behaviour. In range [0, 1]
    pub gamma: f64,
    /// Amount of random influence in parameter update
    pub alpha: f64,
    pub num_candidates: usize,
    pub param_mapping: ParameterMapper,
}

pub struct FireflyOptimizer {
    params: FireflyParams,
    candidates: Vec<Vec<f64>>,
    fits: Vec<f64>,
    elite_idx: usize,
    rng: WyRand,
}

impl FireflyOptimizer {
    #[inline(always)]
    pub fn new(params: FireflyParams) -> Self {
        // TODO: optional poisson-disk sampling
        let mut rng = WyRand::new();
        let candidates = (0..params.num_candidates)
            .map(|_| (0..NUM_CANDIDATE_PARAMS).map(|_| rng.generate::<f64>()).collect())
            .collect();
        let fits = vec![0.0; params.num_candidates];

        Self {
            params,
            candidates,
            fits,
            elite_idx: 0,
            rng,
        }
    }

    pub fn step(
        &mut self,
        train_inputs: &Inputs,
        train_targets: &Targets,
        inputs: &Inputs,
        targets: &Targets,
    ) {
        self.update_candidates();

        for (i, c) in self.candidates.iter().enumerate() {
            let params = self.params.param_mapping.map(&c);
            let mut rc = ESN::new(params);

            rc.train(train_inputs, train_targets);
            rc.reset_state();

            let f = self.evaluate(&mut rc, inputs, targets);
            self.fits[i] = f;
        }
        let mut max_idx = 0;
        let mut max_fit = self.fits[0];
        for (i, fit) in self.fits.iter().enumerate() {
            if *fit > max_fit {
                max_fit = *fit;
                max_idx = i;
            }
        }
        self.elite_idx = max_idx;
    }

    fn update_candidates(&mut self) {
        for i in 0..self.fits.len() {
            for j in 0..self.fits.len() {
                if self.fits[i] < self.fits[j] {
                    continue;
                }

                // I is more fit than J

                let mut dist: f64 = 0.0;
                for p in 0..self.candidates[i].len() {
                    dist += (self.candidates[i][p] - self.candidates[j][p]).powi(2);
                }
                dist = dist.sqrt();

                let beta: f64 = -self.params.gamma * dist.powi(2);
                let r = self.params.alpha * self.rng.generate::<f64>();

                for p in 0..self.candidates[i].len() {
                    let new =
                        beta * self.candidates[j][p] + (1.0 - beta) * self.candidates[i][p] + r;
                    self.candidates[j][p] = new;
                }
            }
        }
    }

    /// Evaluate the performance of the ESN
    fn evaluate(&self, rc: &mut ESN, inputs: &Inputs, targets: &Targets) -> f64 {
        let t0 = Instant::now();
        let mut test_rmse = 0.0;
        for i in 0..inputs.nrows() {
            let predicted_out = rc.readout();
            let last_prediction = *predicted_out.get(0).unwrap();

            // To begin forecasting, replace target input with it's own prediction
            let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |_, j| {
                    *predicted_out.get(j).unwrap()
                });
            let target = *targets.row(i).get(0).unwrap();
            test_rmse += (last_prediction - target).powi(2);
            let input = m.row(0);

            rc.update_state(&input, &predicted_out);
        }
        info!("evaluation took {}ms, rmse: {}", t0.elapsed().as_millis(), test_rmse);

        test_rmse
    }

    #[inline(always)]
    pub fn elite(&self) -> ESN {
        let c = &self.candidates[self.elite_idx];
        let params = self.params.param_mapping.map(c);

        ESN::new(params)
    }

    #[inline(always)]
    pub fn fits(&self) -> &Vec<f64> {
        &self.fits
    }

    pub fn candidates(&self) -> &Vec<Vec<f64>> {
        &self.candidates
    }
}
