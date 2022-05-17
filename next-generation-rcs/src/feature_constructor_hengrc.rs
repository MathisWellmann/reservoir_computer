use nalgebra::{Const, DMatrix, Dynamic, MatrixSlice};

use super::{params::Params, FullFeatureConstructor};

/// The high-efficiency next-generation reservoir computer constructor
/// from: https://arxiv.org/abs/2110.13614
pub struct HENGRCConstructor {}

impl FullFeatureConstructor for HENGRCConstructor {
    fn construct_full_features<'a>(
        params: &Params,
        lin_part: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) -> DMatrix<f64> {
        todo!()
    }
}
