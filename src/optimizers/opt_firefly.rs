use std::{cmp::max, sync::Arc};

use crossbeam::channel::unbounded;
use nanorand::{Rng, WyRand};
use threadpool::ThreadPool;

use super::OptEnvironment;

// TODO: rename to Params
#[derive(Debug, Clone)]
pub struct FireflyParams {
    /// Influces the clustering behaviour. In range [0, 1]
    pub gamma: f64,
    /// Amount of random influence in parameter update
    pub alpha: f64,
    /// Step size
    pub step_size: f64,
    pub num_candidates: usize,
}

pub struct FireflyOptimizer<const N: usize> {
    params: FireflyParams,
    candidates: Vec<[f64; N]>,
    fits: Vec<f64>,
    elite_idx: usize,
    rng: WyRand,
}

impl<const N: usize> FireflyOptimizer<N> {
    pub fn new(params: FireflyParams) -> Self {
        // TODO: optional poisson-disk sampling
        let mut rng = WyRand::new();
        let candidates = (0..params.num_candidates)
            .map(|_| {
                let mut params = [0.0; N];
                for p in params.iter_mut() {
                    *p = rng.generate::<f64>();
                }
                params
            })
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

    pub fn step(&mut self, env: Arc<dyn OptEnvironment<N> + Send + Sync>) {
        self.update_candidates();

        let pool = ThreadPool::new(max(num_cpus::get() - 1, 1));

        let (ch_fit_s, ch_fit_r) = unbounded();
        for (i, c) in self.candidates.iter().enumerate() {
            let ch_fit_s = ch_fit_s.clone();
            let params = c.clone();
            let e = env.clone();
            pool.execute(move || {
                let f = e.evaluate(&params);
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
            // minimizing rmse
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
    pub fn elite_params(&self) -> &[f64; N] {
        &self.candidates[self.elite_idx]
    }

    #[inline(always)]
    pub fn fits(&self) -> &Vec<f64> {
        &self.fits
    }

    #[inline(always)]
    pub fn candidates(&self) -> &Vec<[f64; N]> {
        &self.candidates
    }
}
