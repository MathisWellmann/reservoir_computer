use crate::Series;

/// Gathers plotting information
#[derive(Debug, Clone, Default)]
pub struct PlotGather {
    plot_targets: Series,
    train_predictions: Series,
    test_predictions: Series,
}

impl PlotGather {
    /// Push a target value
    #[inline(always)]
    pub fn push_target(&mut self, x: f64, y: f64) {
        self.plot_targets.push((x, y))
    }

    /// Push a training prediction
    #[inline(always)]
    pub fn push_train_pred(&mut self, x: f64, y: f64) {
        self.train_predictions.push((x, y))
    }

    /// Push a test prediction
    #[inline(always)]
    pub fn push_test_pred(&mut self, x: f64, y: f64) {
        self.test_predictions.push((x, y))
    }

    /// The targets
    #[inline(always)]
    pub fn target_series(&self) -> &Series {
        &self.plot_targets
    }

    /// The prediction series from training data
    #[inline(always)]
    pub fn train_predictions(&self) -> &Series {
        &self.train_predictions
    }

    /// The prediction series from test data
    #[inline(always)]
    pub fn test_predictions(&self) -> &Series {
        &self.test_predictions
    }
}
