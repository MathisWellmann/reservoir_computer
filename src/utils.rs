use num::Float;

#[inline(always)]
pub(crate) fn scale<F: Float>(from_min: F, from_max: F, to_min: F, to_max: F, value: F) -> F {
    to_min + ((value - from_min) * (to_max - to_min)) / (from_max - from_min)
}
