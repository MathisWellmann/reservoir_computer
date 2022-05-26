use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, VecStorage};

use super::{params::Params, FullFeatureConstructor};

/// The high-efficiency next-generation reservoir computer constructor
/// loosely based on: https://arxiv.org/abs/2110.13614
#[derive(Default, Clone, Copy)]
pub struct HENGRCConstructor {}

impl FullFeatureConstructor for HENGRCConstructor {
    fn construct_full_features<'a>(params: &Params, lin_part: &DMatrix<f64>) -> DMatrix<f64> {
        let d_lin = params.num_time_delay_taps;
        let d_nonlin = (2 * d_lin) - 1;
        let d_total = d_lin + d_nonlin;

        let mut full_features: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(lin_part.nrows()),
            Dim::from_usize(d_total),
            0.0,
        );
        for i in 0..lin_part.nrows() {
            full_features.set_row(i, &lin_part.row(i).resize_horizontally(d_total, 0.0));
        }

        let mut cnt: usize = 0;
        for i in 0..d_lin {
            for j in 0_i32..2_i32 {
                if i == 0 && j == 1 {
                    continue;
                }
                let column: Vec<f64> = lin_part
                    .column(i)
                    .iter()
                    .zip(lin_part.column((i as i32 - j) as usize).iter())
                    .map(|(middel, prev)| middel * prev)
                    .collect();

                let column: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                    Matrix::from_vec_generic(
                        Dim::from_usize(lin_part.nrows()),
                        Dim::from_usize(1),
                        column,
                    );
                full_features.set_column(d_lin + cnt, &column);
                cnt += 1;
            }
        }

        full_features
    }
}

#[cfg(test)]
mod tests {
    use common::Activation;
    use round::round;

    use super::*;

    #[test]
    fn feature_constructor_hengrc() {
        if let Err(_) = pretty_env_logger::try_init() {}

        let lin_part: DMatrix<f64> = Matrix::from_vec_generic(
            Dim::from_usize(1),
            Dim::from_usize(4),
            vec![0.1, 0.2, 0.3, 0.4],
        );
        info!("lin_part: {}", lin_part);

        let params = Params {
            input_dim: 1,
            output_dim: 1,
            num_time_delay_taps: 4,
            num_samples_to_skip: 1,
            output_activation: Activation::Identity,
        };
        let mut full_features =
            <HENGRCConstructor as FullFeatureConstructor>::construct_full_features(
                &params, &lin_part,
            );

        full_features.iter_mut().for_each(|v| *v = round(*v, 4));

        info!("full_features: {}", full_features);
    }
}
