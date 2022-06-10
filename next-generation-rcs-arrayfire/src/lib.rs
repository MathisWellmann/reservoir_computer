#[cfg(test)]
mod tests {
    use arrayfire::*;

    #[test]
    fn main() {
        set_device(0);
        print!("Info String: {}", info_string(true));
    }
}
