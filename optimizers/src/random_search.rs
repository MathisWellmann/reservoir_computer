//! The random search algorithm

use std::{cmp::max, sync::Arc};

use common::ReservoirComputer;
use crossbeam::channel::unbounded;
use lin_reg::LinReg;
use nanorand::{Rng, WyRand};
use threadpool::ThreadPool;

use crate::OptEnvironment;

/// Optimization using random search algorithm
pub struct RandomSearch<const N: usize> {
    /// Minimization objective
    best_error: f64,
    best_params: [f64; N],
    rng: WyRand,
    num_candidates: usize,
    candidates: Vec<[f64; N]>,
    rmses: Vec<f64>,
}

impl<const N: usize> RandomSearch<N> {
    /// Create a new random search based optimizer
    pub fn new(seed: Option<u64>, num_candidates: usize) -> Self {
        let rng = if let Some(seed) = seed {
            WyRand::new_seed(seed)
        } else {
            WyRand::new()
        };

        Self {
            best_error: f64::MAX,
            best_params: [0.0; N],
            rng,
            num_candidates,
            candidates: vec![[0.0; N]; num_candidates],
            rmses: vec![f64::MAX; num_candidates],
        }
    }

    /// Perform a single optimization step
    pub fn step<RC, R, F, E>(&mut self, env: Arc<E>, rc_gen: F)
    where
        RC: ReservoirComputer<R> + Send + Sync + 'static,
        R: LinReg + Send + Sync + 'static,
        F: Fn(&[f64; N]) -> RC,
        E: OptEnvironment<RC, R> + Send + Sync + 'static,
    {
        let pool = ThreadPool::new(max(num_cpus::get() - 2, 1));

        self.candidates = self.gen_candidates(self.num_candidates);

        let (ch_fit_s, ch_fit_r) = unbounded();
        for (i, c) in self.candidates.iter().enumerate() {
            let ch_fit_s = ch_fit_s.clone();
            let e = env.clone();
            let mut rc = rc_gen(c);
            pool.execute(move || {
                let f = e.evaluate(&mut rc);
                ch_fit_s.send((i, f)).unwrap();
            });
        }
        drop(ch_fit_s);
        while let Ok((i, error)) = ch_fit_r.recv() {
            self.rmses[i] = error;
        }

        for (i, e) in self.rmses.iter().enumerate() {
            if *e < self.best_error {
                self.best_error = *e;
                self.best_params = self.candidates[i];
            }
        }
    }

    /// The best evaluated parameters
    #[inline(always)]
    pub fn elite_params(&self) -> &[f64; N] {
        &self.best_params
    }

    /// The best RMSE of the `elite_params`
    #[inline(always)]
    pub fn best_rmse(&self) -> f64 {
        self.best_error
    }

    /// All the candidates to be evaluated
    #[inline(always)]
    pub fn candidates(&self) -> &Vec<[f64; N]> {
        &self.candidates
    }

    /// All the RMSE's of candidates that have been evaluated
    #[inline(always)]
    pub fn rmses(&self) -> &Vec<f64> {
        &self.rmses
    }

    /// Generate random candidates
    fn gen_candidates(&mut self, n: usize) -> Vec<[f64; N]> {
        (0..n)
            .map(|_| {
                let mut params = [0.0; N];
                for p in params.iter_mut() {
                    *p = self.rng.generate::<f64>();
                }
                params
            })
            .collect()
    }
}
