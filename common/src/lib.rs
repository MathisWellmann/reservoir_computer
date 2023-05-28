//! This crate provides common funcionality

#![deny(unused_imports, unused_crate_dependencies)]
#![warn(missing_docs)]

mod activation;
mod rc_trait;

pub use activation::Activation;
pub use rc_trait::{RCParams, ReservoirComputer};

#[cfg(feature = "environments")]
pub mod environments;
