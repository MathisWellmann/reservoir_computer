# Reservoir Computers
[Reservoir Computers](https://en.wikipedia.org/wiki/Reservoir_computing) 
are a type of recurrent neural network with a fixed internal connection matrix.
The only thing being "trained" in this paradigm is the readout layer, by fitting a linear regression including regularization.
This makes it extremely efficient and powerful for predictive tasks of (chaotic) dynamical systems.

This crate provides a few reservoir computer implementations in Rust mainly for educational purposes.
They all run on CPU and rely on the (`nalgebra`)[https://github.com/dimforge/nalgebra] crate for matrix multiplications.
It is intended to be a flexible, research oriented library that allows anyone to implement their own spin of RC.
It achieves this by making many things generic, so you can just plug and play your own extensions.
Currently It only works for 1 dimensional input signals. 
If you are capable of generalizing this to arbitrary IO dimension, I encourage you to implement is and create a PR.

This repository contains three concrete reservoir computers:
- Echo State Network (ESN)
- Euler State Network (EuSN)
- Next Generation Reservoir Computer (NGRC)

:warning: The implementation aims to be as close as possible to the theoretical groundwork layed out in many academic papers, however the correctness cannot be guaranteed.
Its obvious that reservoir computers benefit massively from GPU acceleration due to the heavy use of matmul operations. A GPU variant of each network can be derived fairly easily from the knowledge in this repo. I use [arrayfire-rs](https://github.com/arrayfire/arrayfire-rust) for this acceleration (in separate repo and private for now)

### TODOS:
- benchmark chaotic dynamical systems
- implement lyapunov time for analysis purposes
- Make error measure generic and add MAE as well as RMSE, which already exists
- Add all fields to all Cargo.toml files
- Add prediction plot to this README

### Contributions
If you find a bug or would like to help out, feel free to create a pull-request.

### Donations :moneybag: :money_with_wings:
I you would like to support the development of this crate, feel free to send over a donation:

Monero (XMR) address:
```plain
47xMvxNKsCKMt2owkDuN1Bci2KMiqGrAFCQFSLijWLs49ua67222Wu3LZryyopDVPYgYmAnYkSZSz9ZW2buaDwdyKTWGwwb
```

![monero](img_readme/monero_donations_qrcode.png)

### License
Copyright (C) 2020  <Mathis Wellmann wellmannmathis@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

![GNU AGPLv3](img_readme/agplv3.png)
