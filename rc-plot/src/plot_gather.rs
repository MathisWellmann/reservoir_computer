use crate::Series;

#[derive(Debug, Clone, Default)]
pub struct PlotGather {
    plot_targets: Series,
    train_predictions: Series,
    test_predictions: Series,
}

impl PlotGather {
    #[inline(always)]
    pub fn push_target(&mut self, x: f64, y: f64) {
        self.plot_targets.push((x, y))
    }

    #[inline(always)]
    pub fn push_train_pred(&mut self, x: f64, y: f64) {
        self.train_predictions.push((x, y))
    }

    #[inline(always)]
    pub fn push_test_pred(&mut self, x: f64, y: f64) {
        self.test_predictions.push((x, y))
    }

    #[inline(always)]
    pub fn plot_targets(&self) -> &Series {
        &self.plot_targets
    }

    #[inline(always)]
    pub fn train_predictions(&self) -> &Series {
        &self.train_predictions
    }

    #[inline(always)]
    pub fn test_predictions(&self) -> &Series {
        &self.test_predictions
    }
}
