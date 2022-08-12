# Reservoir Computers
This crate provides a few [reservoir computer](https://en.wikipedia.org/wiki/Reservoir_computing) implementations in Rust.
They all run on CPU and rely on the nalgebra crate for matrix multiplications.

This repository contains three types of reservoir computers:
- Echo State Network (ESN)
- Euler State Network (EuSN)
- Next Generation Reservoir Computer (NGRC)

## TODOS:
- benchmark chaotic dynamical systems
- implement lyapunov time for analysis purposes
- Make error measure generic and add MAE as well as RMSE, which already exists
- Add all fields to all Cargo.toml files
