# Reservoir Computers
[Reservoir Computers](https://en.wikipedia.org/wiki/Reservoir_computing) 
are a type of recurrent neural network with a fixed internal connection matrix.
The only thing being "trained" in this paradigm is the readout layer, by fitting a linear regression.
This makes it extremely efficient and powerful for predictive tasks of (chaotic) dynamical systems.

This crate provides a few reservoir computer implementations in Rust.
They all run on CPU and rely on the nalgebra crate for matrix multiplications.
It is intended to be a flexible, research oriented library that allows anyone to implement their own spin of RC.
It achieves this by making many things generic, so you can just plug and play your own extensions.
Currently I belive my networks only work for 1 dimensional input signals, which is my use case,
so I they may not work for multi dimensional IO.

There are two distinct approaches to reservoir computing,
one which I'll call the [classic approach](./classic-rcs) and the [next-generation RCs](./next-generation-rcs).
They both require two distinct APIs and the classic approach has significantly more parameters (7 vs 3).

This repository contains three concrete reservoir computers:
- Echo State Network (ESN)
- Euler State Network (EuSN)
- Next Generation Reservoir Computer (NGRC)

## TODOS:
- benchmark chaotic dynamical systems
- implement lyapunov time for analysis purposes
- Make error measure generic and add MAE as well as RMSE, which already exists
- Add all fields to all Cargo.toml files
