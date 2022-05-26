#[macro_use]
extern crate log;

pub use feature_constructor_hengrc::HENGRCConstructor;
pub use feature_constructor_ngrc::NGRCConstructor;
use nalgebra::DMatrix;
pub use ngrc::NextGenerationRC;
pub use params::Params;

mod feature_constructor_hengrc;
mod feature_constructor_ngrc;
mod ngrc;
mod params;

pub trait FullFeatureConstructor {
    /// Construct the full feature space from the linear part
    fn construct_full_features<'a>(&self, lin_part: &DMatrix<f64>) -> DMatrix<f64>;

    /// The total dimension of features
    fn d_total(&self) -> usize;
}
