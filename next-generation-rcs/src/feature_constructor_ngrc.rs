use nalgebra::{Const, DMatrix, Dim, Dynamic, Matrix, VecStorage};

use super::{params::Params, FullFeatureConstructor};

/// The classic next-generation reservoir computer constructor
#[derive(Clone)]
pub struct NGRCConstructor {
    num_time_delay_taps: usize,
    num_samples_to_skip: usize,
}

impl NGRCConstructor {
    /// Create a new constructor for the next generation reservoir computer
    ///
    /// # Arguments:
    /// num_time_delay_taps: The number of values to sample from input sequence
    /// num_samples_to_skip: Take samples from inputs every n steps
    pub fn new(num_time_delay_taps: usize, num_samples_to_skip: usize) -> Self {
        Self {
            num_time_delay_taps,
            num_samples_to_skip,
        }
    }
}

impl FullFeatureConstructor for NGRCConstructor {
    /// Construct the nonlinear part of feature matrix from linear part
    ///
    /// # Arguments
    /// inputs: Number of rows are the observed datapoints and number of columns
    /// represent the features at each timestep
    fn construct_full_features<'a>(&self, lin_part: &DMatrix<f64>) -> DMatrix<f64> {
        let d_lin = self.num_time_delay_taps;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        let d_total = d_lin + d_nonlin;

        let warmup = self.num_time_delay_taps * self.num_samples_to_skip;

        // manually copy over elements while skipping the warmup columns
        let mut full_features: DMatrix<f64> = Matrix::from_element_generic(
            Dim::from_usize(lin_part.nrows() - warmup),
            Dim::from_usize(d_total),
            0.0,
        );
        for i in warmup..lin_part.nrows() {
            full_features.set_row(i - warmup, &lin_part.row(i).resize_horizontally(d_total, 0.0));
        }

        let mut cnt: usize = 0;
        for i in 0..d_lin {
            for j in i..d_lin {
                for span in j..d_lin {
                    let column: Vec<f64> = lin_part
                        .column(i)
                        .iter()
                        .skip(warmup)
                        .zip(lin_part.column(j).iter().skip(warmup))
                        .zip(lin_part.column(span).iter().skip(warmup))
                        .map(|((v_i, v_j), v_s)| v_i * v_j * v_s)
                        .collect();
                    let column: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>> =
                        Matrix::from_vec_generic(
                            Dim::from_usize(lin_part.nrows() - warmup),
                            Dim::from_usize(1),
                            column,
                        );
                    full_features.set_column(d_lin + cnt, &column);
                    cnt += 1;
                }
            }
        }

        full_features
    }

    /// Total state dimension
    fn d_total(&self) -> usize {
        let d_lin = self.num_time_delay_taps;
        let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
        d_lin + d_nonlin
    }
}
