#[macro_use]
extern crate log;

use nalgebra::{Const, DMatrix, Dynamic, MatrixSlice};

pub use feature_constructor_hengrc::HENGRCConstructor;
pub use feature_constructor_ngrc::NGRCConstructor;
pub use ngrc::NextGenerationRC;
pub use params::Params;

mod feature_constructor_hengrc;
mod feature_constructor_ngrc;
mod ngrc;
mod params;

pub trait FullFeatureConstructor {
    /// Construct the full feature space from the linear part
    fn construct_full_features<'a>(
        params: &Params,
        lin_part: &'a MatrixSlice<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>,
    ) -> DMatrix<f64>;
}
