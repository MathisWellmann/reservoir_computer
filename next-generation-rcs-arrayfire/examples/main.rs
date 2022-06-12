use arrayfire::{af_print, info, set_device, Array, Dim4};
use common::Activation;

use next_generation_rcs_arrayfire::{NextGenerationRC, Params};

const NUM_VALS: usize = 9;

fn get_inputs() -> [f32; NUM_VALS] {
    [0.0, 0.55, 1.0, 0.45, 0.0, -0.55, -1.0, -0.45, 0.0]
}

fn main() {
    if let Err(_) = pretty_env_logger::try_init() {}

    set_device(0);
    info();

    let d_lin = 3;
    let d_nonlin = d_lin * (d_lin + 1) * (d_lin + 2) / 6;
    let reservoir_size = d_lin + d_nonlin;

    let params = Params {
        num_time_delay_taps: d_lin,
        num_samples_to_skip: 1,
        output_activation: Activation::Tanh,
        reservoir_size,
    };
    let mut rc = NextGenerationRC::new(params);

    let values: [f32; 3] = [1.0, 2.0, 3.0];
    let indices = Array::new(&values, Dim4::new(&[3, 1, 1, 1]));
    af_print!("indices", indices);

    /*
    let inputs = get_inputs();
    let lin_part = rc.construct_lin_part(&inputs);

    af_print!("lin_part", lin_part);
    */
}
