use crate::{ReservoirComputer, Series};

pub mod env_mackey_glass;
pub mod env_trades;

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

/// Optimization environment for validating parameters
/// R: ReservoirComputer
/// I: Input dimension
/// O: Output dimension
/// N: Dimensionality of parameter search space
pub trait OptEnvironment<
    R: ReservoirComputer<I, O, N>,
    const I: usize,
    const O: usize,
    const N: usize,
>
{
    /// Evaluate the reservoir computer and return the rmse values
    fn evaluate(&self, rc: &mut R, plot: Option<&mut PlotGather>) -> f64;
}
