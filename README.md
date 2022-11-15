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

This repository contains these concrete reservoir computers:
- Echo State Network (ESN)
- Euler State Network (EuSN)

## TODOS:
- benchmark chaotic dynamical systems
- implement lyapunov time for analysis purposes
- Add all fields to all Cargo.toml files
