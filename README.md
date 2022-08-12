# Reservoir Computers
This crate provides a few [reservoir computer](https://en.wikipedia.org/wiki/Reservoir_computing) implementations in Rust.
They all run on CPU and rely on the nalgebra crate for matrix multiplications.
It is intended to be a flexible, research oriented library that allows anyone to implement their own spin of RC.
It achieves this by making many things generic, so you can just plug and play your own extensions

There are two distinct approaches to reservoir computing,
one which I'll call the [classic approach](./classic-rcs) and the [next-generation RCs](./next-generation-rcs).

This repository contains three concrete reservoir computers:
- Echo State Network (ESN)
- Euler State Network (EuSN)
- Next Generation Reservoir Computer (NGRC)

## TODOS:
- benchmark chaotic dynamical systems
- implement lyapunov time for analysis purposes
- Make error measure generic and add MAE as well as RMSE, which already exists
- Add all fields to all Cargo.toml files
