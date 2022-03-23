use std::{sync::Arc, time::Instant};

use crossbeam::channel::unbounded;
use nalgebra::{Const, Dim, Dynamic, Matrix, VecStorage};
use nanorand::{Rng, WyRand};
use threadpool::ThreadPool;

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
            output_tanh: true,
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
    /// Step size
    pub step_size: f64,
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
        train_inputs: Arc<Inputs>,
        train_targets: Arc<Targets>,
        inputs: Arc<Inputs>,
        targets: Arc<Targets>,
    ) {
        self.update_candidates();

        let pool = ThreadPool::new(num_cpus::get());

        let (ch_fit_s, ch_fit_r) = unbounded();
        for (i, c) in self.candidates.iter().enumerate() {
            let ch_fit_s = ch_fit_s.clone();
            let train_inputs = train_inputs.clone();
            let train_targets = train_targets.clone();
            let inputs = inputs.clone();
            let targets = targets.clone();
            let params = self.params.param_mapping.map(&c);
            pool.execute(move || {
                let mut rc = ESN::new(params);

                rc.train(&train_inputs, &train_targets);
                rc.reset_state();

                let f = Self::evaluate(&mut rc, &inputs, &targets);
                ch_fit_s.send((i, f)).unwrap();
            });
        }
        drop(ch_fit_s);
        while let Ok((i, fit)) = ch_fit_r.recv() {
            self.fits[i] = fit;
        }

        let mut min_idx = 0;
        let mut min_rmse = self.fits[0];
        for (i, fit) in self.fits.iter().enumerate() {
            // minizing rmse
            if *fit < min_rmse {
                min_rmse = *fit;
                min_idx = i;
            }
        }
        self.elite_idx = min_idx;
    }

    fn update_candidates(&mut self) {
        for i in 0..self.fits.len() {
            for j in 0..self.fits.len() {
                if self.fits[i] > self.fits[j] {
                    continue;
                }

                // I is more fit than J, by having lower rmse

                let mut dist: f64 = 0.0;
                for p in 0..self.candidates[i].len() {
                    dist += (self.candidates[i][p] - self.candidates[j][p]).powi(2);
                }
                let attractiveness = self.fits[i] * (-self.params.gamma * dist).exp();

                let r = self.params.alpha * (self.rng.generate::<f64>() * 2.0 - 1.0);

                for p in 0..self.candidates[i].len() {
                    let old = self.candidates[j][p];
                    let new = old
                        + self.params.step_size
                            * attractiveness
                            * (self.candidates[j][p] - self.candidates[i][p])
                        + r;
                    self.candidates[j][p] = Self::bounds_checked(old, new);
                }
            }
        }
    }

    /// Evaluate the performance of the ESN
    fn evaluate(rc: &mut ESN, inputs: &Inputs, targets: &Targets) -> f64 {
        let t0 = Instant::now();
        let mut rmse = 0.0;
        let n = inputs.nrows();
        for i in 0..n {
            let predicted_out = rc.readout();
            let last_prediction = *predicted_out.get(0).unwrap();

            // To begin forecasting, replace target input with it's own prediction
            let m: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                Matrix::from_fn_generic(Dim::from_usize(1), Dim::from_usize(1), |_, j| {
                    *predicted_out.get(j).unwrap()
                });
            let target = *targets.row(i).get(0).unwrap();
            if i > n / 4 {
                rmse += (last_prediction - target).powi(2);
            }
            let input = m.row(0);

            rc.update_state(&input, &predicted_out);
        }
        info!("evaluation took {}ms, rmse: {}", t0.elapsed().as_millis(), rmse);

        rmse
    }

    // ensure the parameter bounds of the problem
    fn bounds_checked(old: f64, new: f64) -> f64 {
        return if new > 1.0 {
            old
        } else if new < 0.0 {
            old
        } else {
            new
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_firefly() {
        let params = FireflyParams {
            gamma: 0.1,
            alpha: 0.05,
            num_candidates: 10,
            param_mapping: ParameterMapper::new(
                vec![(0.01, 0.2), (0.5, 1.0), (0.0, 0.5), (2.0, 10.0)],
                Activation::Identity,
                500,
                Activation::Tanh,
                0.02,
                0.1,
                None,
                0.0005,
                0.0,
            ),
            step_size: 0.01,
        };
        let ff = FireflyOptimizer::new(params);
        println!("candidates: {:?}", ff.candidates());
    }
}
