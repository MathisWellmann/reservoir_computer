/// Temporal Prediction Aggregation
/// H: max buffer length, which should be larger than the prediction_horizon
pub struct TPA<const H: usize> {
    idx_first: usize,
    buf_avg: [f64; H],
    buf_weighted_avg: [f64; H],
    buf_min: [f64; H],
    buf_max: [f64; H],
    buf_stddev: [f64; H],
}

impl<const H: usize> TPA<H> {
    pub fn new() -> Self {
        Self {
            idx_first: 0,
            buf_avg: [0.0; H],
            buf_weighted_avg: [0.0; H],
            buf_min: [0.0; H],
            buf_max: [0.0; H],
            buf_stddev: [0.0; H],
        }
    }

    pub fn update<const N: usize>(&mut self, predictions: &[[f64; N]]) {
        self.idx_first += 1;

        if self.idx_first >= H / 2 {
            // shift all the data to lower indices to make space for newer ones
            for i in 0..self.idx_first {
                let old_idx = self.idx_first + i;
                self.buf_avg[i] = self.buf_avg[old_idx];
                self.buf_weighted_avg[i] = self.buf_weighted_avg[old_idx];
                self.buf_min[i] = self.buf_min[old_idx];
                self.buf_max[i] = self.buf_max[old_idx];
                self.buf_stddev[i] = self.buf_stddev[old_idx];
            }
            self.idx_first = 0;
        }

        for (i, p) in predictions.iter().enumerate() {
            let idx = self.idx_first + i;

            // update average
            let avg = p.iter().sum::<f64>() / p.len() as f64;
            let old_avg = self.buf_avg[idx];
            self.buf_avg[idx] = (avg + old_avg) / 2.0;

            // update minimum and maximum
            let mut min: f64 = p[0];
            let mut max: f64 = p[0];
            for val in p.iter() {
                if *val < min {
                    min = *val;
                }
                if *val > max {
                    max = *val;
                }
            }
            if min < self.buf_min[idx] {
                self.buf_min[idx] = min;
            }
            if max > self.buf_max[idx] {
                self.buf_max[idx] = max;
            }

            // TODO: update std_dev
        }

        todo!()
    }

    #[inline(always)]
    pub fn prediction_avg_series(&self) -> &[f64] {
        &self.buf_avg[self.idx_first..self.idx_first + H]
    }

    #[inline(always)]
    pub fn prediction_weighted_avg_series(&self) -> &[f64] {
        &self.buf_weighted_avg[self.idx_first..self.idx_first + H]
    }

    #[inline(always)]
    pub fn prediction_max_series(&self) -> &[f64] {
        &self.buf_max[self.idx_first..self.idx_first + H]
    }

    #[inline(always)]
    pub fn prediction_min_series(&self) -> &[f64] {
        &self.buf_min[self.idx_first..self.idx_first + H]
    }

    #[inline(always)]
    pub fn prediction_stddev_series(&self) -> &[f64] {
        &self.buf_stddev[self.idx_first..self.idx_first + H]
    }
}
