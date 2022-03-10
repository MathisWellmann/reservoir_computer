use nalgebra::{Const, Dim, OMatrix, SymmetricEigen};
use nanorand::{Rng, WyRand};

/// The Reseoir Computer
/// N is the number of reservoir nodes
/// M is the input and output dimension
#[derive(Debug)]
pub(crate) struct Reservoir<const N: usize, const M: usize> {
    reservoir: OMatrix<f32, Const<N>, Const<N>>,
}

impl<const N: usize, const M :usize> Reservoir<N, M> {
    /// Create a new reservoir, with random initiallization
    /// # Arguments
    /// sparsity: the connection probability within the reservoir
    /// spectral_radius:
    pub(crate) fn new(
        sparsity: f32,
        spectral_radius: f32,
        leaking_rate: f32,
    ) -> Self {

        Self {
            reservoir,
        }
    }

    /*
    pub(crate) fn forward(&self, input: SVector<f32, M>) -> SVector<f32, M> {
        todo!()
    }
    */
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir() {
        if let Err(_) = pretty_env_logger::try_init() {}

        let res = Reservoir::<10, 1>::new(0.1, 0.9, 0.1);
        info!("res: {:?}", res);
    }
}
