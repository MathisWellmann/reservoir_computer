#[macro_use]
extern crate log;

mod ngrc;
mod params;

pub use ngrc::NextGenerationRC;
pub use params::Params;

#[cfg(test)]
mod tests {
    use arrayfire::*;

    #[test]
    fn main() {
        set_device(0);
        print!("Info String: {}", info_string(true));
    }
}
