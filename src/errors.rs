#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {}

pub(crate) type Result<T> = std::result::Result<T, Error>;
