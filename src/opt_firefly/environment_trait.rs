/// Firefly environment for validating parameters
/// N: constant number of parameters
pub trait FFEnv<const N: usize> {
    /// Evaluate the parameters and return the fitness values
    /// # Arguments
    /// params: parameters in range [0.0, 1.0]
    ///
    /// # Returns
    /// fitness: higher is better
    fn evaluate(&self, params: &[f64; N]) -> f64;
}
