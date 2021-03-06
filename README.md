# MNIST
A crate for parsing the [MNIST](http://yann.lecun.com/exdb/mnist/) and [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) data set into vectors to be
used by Rust programs.

* [Crate](https://crates.io/crates/mnist)
* [Documentation](https://docs.rs/mnist)

## Example
```rust
extern crate mnist;
extern crate rulinalg;

use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

fn main() {
    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Call `normalize()` on the return value of `finalize()` to get
    // Vec<f32> normalized values for the pixels instead of grayscale (bytes):
    // let NormalizedMnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
    //     .label_format_digit()
    //     .training_set_length(trn_size)
    //     .validation_set_length(10_000)
    //     .test_set_length(10_000)
    //     .finalize()
    //     .normalize();

    // Get the label of the first digit.
    let first_label = trn_lbl[0];
    println!("The first digit is a {}.", first_label);

    // Convert the flattened training images vector to a matrix.
    let trn_img = Matrix::new((trn_size * rows) as usize, cols as usize, trn_img);

    // Get the image of the first digit.
    let row_indexes = (0..27).collect::<Vec<_>>();
    let first_image = trn_img.select_rows(&row_indexes);
    println!("The image looks like... \n{}", first_image);

    // Convert the training images to f32 values scaled between 0 and 1.
    let trn_img: Matrix<f32> = trn_img.try_into().unwrap() / 255.0;

    // Get the image of the first digit and round the values to the nearest tenth.
    let first_image = trn_img.select_rows(&row_indexes)
        .apply(&|p| (p * 10.0).round() / 10.0);
    println!("The image looks like... \n{}", first_image);
}
```

## Fashion MNIST
The Fasion MNIST [dataset](https://github.com/zalandoresearch/fashion-mnist) offers a similarly-formatted 
drop-in replacement dataset for the original MNIST set, but typically poses a more difficult classification challenge that handwritten numbers. 

An example of downloading this dataset may be found by running: 
```sh
$ cargo run --features download --example fashion_mnist
```
This example uses the [minifb](https://github.com/emoon/rust_minifb) library to display the parsed images,
and may require the installation of certain dependencies. On an Ubuntu-like system, this may be done via:
```sh
$ sudo apt install libxkbcommon-dev libwayland-cursor0 libwayland-dev
```
