use nalgebra::DMatrix;

use self::params::Params;

mod feature_constructor_hengrc;
mod feature_constructor_ngrc;
mod ngrc;
mod params;

pub trait FullFeatureConstructor {
    /// Construct the full feature space from the linear part
    fn construct_full_features<'a>(params: &Params, lin_part: &DMatrix<f64>) -> DMatrix<f64>;
}
