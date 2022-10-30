# Reservoir Computers
[Reservoir Computers](https://en.wikipedia.org/wiki/Reservoir_computing) 
are a type of recurrent neural network with a fixed internal connection matrix.
The only thing being "trained" in this paradigm is the readout layer, by fitting a linear regression includingt regularization.
This makes it extremely efficient and powerful for predictive tasks of (chaotic) dynamical systems.

This crate provides a few reservoir computer implementations in Rust.
They all run on CPU and rely on the nalgebra crate for matrix multiplications.
It is intended to be a flexible, research oriented library that allows anyone to implement their own spin of RC.
It achieves this by making many things generic, so you can just plug and play your own extensions.
Currently I belive my networks only work for 1 dimensional input signals, which is my use case,
so I they may not work for multi dimensional IO. If you are capable of generalizing this to arbitrary IO dimension, I encourage you to implement is and create a PR.

There are two distinct approaches to reservoir computing,
one which I'll call the [classic approach](./classic-rcs) and the [next-generation RCs](./next-generation-rcs).
They both require two distinct APIs and the classic approach has significantly more parameters (7 vs 3).

This repository contains three concrete reservoir computers:
- Echo State Network (ESN)
- Euler State Network (EuSN)
- Next Generation Reservoir Computer (NGRC)

:warning: The implementation aims to be as close as possible to the theoretical groundwork layed out in many academic papers, however the correctness cannot be guaranteed.

### TODOS:
- benchmark chaotic dynamical systems
- implement lyapunov time for analysis purposes
- Make error measure generic and add MAE as well as RMSE, which already exists
- Add all fields to all Cargo.toml files

### Contributions
If you find a bug or would like to help out, feel free to create a pull-request.

### Donations :moneybag: :money_with_wings:
I you would like to support the development of this crate, feel free to send over a donation:

Monero (XMR) address:
```plain
47xMvxNKsCKMt2owkDuN1Bci2KMiqGrAFCQFSLijWLs49ua67222Wu3LZryyopDVPYgYmAnYkSZSz9ZW2buaDwdyKTWGwwb
```

![monero](img/monero_donations_qrcode.png)

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

![GNU AGPLv3](img/agplv3.png)
