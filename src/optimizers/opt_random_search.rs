use std::{cmp::max, sync::Arc};

use crossbeam::channel::unbounded;
use nanorand::{Rng, WyRand};
use threadpool::ThreadPool;

use super::OptEnvironment;

pub struct RandomSearch<const N: usize> {
    /// Minimization objective
    best_error: f64,
    best_params: [f64; N],
    rng: WyRand,
    num_candidates: usize,
    candidates: Vec<[f64; N]>,
    errors: Vec<f64>,
}

impl<const N: usize> RandomSearch<N> {
    #[inline]
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
            errors: vec![f64::MAX; num_candidates],
        }
    }

    pub fn step(&mut self, env: Arc<dyn OptEnvironment<N> + Send + Sync>) {
        let pool = ThreadPool::new(max(num_cpus::get() - 1, 1));

        self.candidates = self.gen_candidates(self.num_candidates);

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
        while let Ok((i, error)) = ch_fit_r.recv() {
            self.errors[i] = error;
        }

        for (i, e) in self.errors.iter().enumerate() {
            if *e < self.best_error {
                self.best_error = *e;
                self.best_params = self.candidates[i];
            }
        }
    }

    #[inline(always)]
    pub fn elite_params(&self) -> &[f64; N] {
        &self.best_params
    }

    #[inline(always)]
    pub fn candidates(&self) -> &Vec<[f64; N]> {
        &self.candidates
    }

    #[inline(always)]
    pub fn errors(&self) -> &Vec<f64> {
        &self.errors
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
