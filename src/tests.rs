#![cfg(test)]

use super::*;
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

#[test]
fn test_example() {
    // Deconstruct the returnded Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .image_format_28x28()
        .label_format_1x1()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Get the label of the first digit.
    let first_label = trn_lbl[[0, 0]];
    println!("The first digit is a {}.", first_label);

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

#[test]
fn test_00() {
    let mnist = MnistBuilder::new().finalize();
    assert!(mnist.trn_img.rows() == 60_000);
    assert!(mnist.trn_img.cols() == 784);
    assert!(mnist.trn_lbl.rows() == 60_000);
    assert!(mnist.trn_lbl.cols() == 1);
    assert!(mnist.val_img.rows() == 0);
    assert!(mnist.val_img.cols() == 784);
    assert!(mnist.val_lbl.rows() == 0);
    assert!(mnist.val_lbl.cols() == 1);
    assert!(mnist.tst_img.rows() == 10_000);
    assert!(mnist.tst_img.cols() == 784);
    assert!(mnist.tst_lbl.rows() == 10_000);
    assert!(mnist.tst_lbl.cols() == 1);
    assert!(mnist.trn_lbl[[0, 0]] == 5);
    assert!(mnist.tst_lbl[[0, 0]] == 7);
}

#[test]
fn test_01() {
    let mnist = MnistBuilder::new()
        .image_format_28x28()
        .label_format_1x10()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    assert!(mnist.trn_img.rows() == 50_000 * 28);
    assert!(mnist.trn_img.cols() == 28);
    assert!(mnist.trn_lbl.rows() == 50_000);
    assert!(mnist.trn_lbl.cols() == 10);
    assert!(mnist.val_img.rows() == 10_000 * 28);
    assert!(mnist.val_img.cols() == 28);
    assert!(mnist.val_lbl.rows() == 10_000);
    assert!(mnist.val_lbl.cols() == 10);
    assert!(mnist.tst_img.rows() == 10_000 * 28);
    assert!(mnist.tst_img.cols() == 28);
    assert!(mnist.tst_lbl.rows() == 10_000);
    assert!(mnist.tst_lbl.cols() == 10);
    assert!(mnist.trn_lbl[[0, 0]] == 0);
    assert!(mnist.trn_lbl[[0, 9]] == 0);
    assert!(mnist.trn_lbl[[0, 5]] == 1);
    assert!(mnist.tst_lbl[[0, 0]] == 0);
    assert!(mnist.tst_lbl[[0, 9]] == 0);
    assert!(mnist.tst_lbl[[0, 7]] == 1);
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000).")]
fn test_02() {
    let _ = MnistBuilder::new()
        .training_set_length(50_001)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000).")]
fn test_03() {
    let _ = MnistBuilder::new()
        .training_set_length(50_000)
        .validation_set_length(10_001)
        .test_set_length(10_000)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000).")]
fn test_04() {
    let _ = MnistBuilder::new()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_001)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Total data set length (70300) greater than maximum possible length (70000).")]
fn test_05() {
    let _ = MnistBuilder::new()
        .training_set_length(50_100)
        .validation_set_length(10_100)
        .test_set_length(10_100)
        .finalize();
}

#[test]
fn test_06() {
    let mnist = MnistBuilder::new()
        .image_format_28x28()
        .label_format_1x10()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .base_path("data_sets/mnist")
        .training_images_filename("training_images")
        .training_labels_filename("training_labels")
        .test_images_filename("test_images")
        .test_labels_filename("test_labels")
        .finalize();
    assert!(mnist.trn_img.rows() == 50_000 * 28);
    assert!(mnist.trn_img.cols() == 28);
    assert!(mnist.trn_lbl.rows() == 50_000);
    assert!(mnist.trn_lbl.cols() == 10);
    assert!(mnist.val_img.rows() == 10_000 * 28);
    assert!(mnist.val_img.cols() == 28);
    assert!(mnist.val_lbl.rows() == 10_000);
    assert!(mnist.val_lbl.cols() == 10);
    assert!(mnist.tst_img.rows() == 10_000 * 28);
    assert!(mnist.tst_img.cols() == 28);
    assert!(mnist.tst_lbl.rows() == 10_000);
    assert!(mnist.tst_lbl.cols() == 10);
    assert!(mnist.trn_lbl[[0, 5]] == 1);
    assert!(mnist.tst_lbl[[0, 7]] == 1);
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000).")]
fn test_07() {
    let _ = MnistBuilder::new()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_001)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Unable to find path to images at \"wrong/path/train-images.idx3-ubyte\".")]
fn test_08() {
    let _ = MnistBuilder::new()
        .base_path("wrong/path")
        .finalize();
}

#[test]
#[should_panic(
    expected = "Unable to find path to images at \"data/test\".")]
fn test_09() {
    let _ = MnistBuilder::new()
        .training_images_filename("test")
        .finalize();
}

#[test]
#[should_panic(
    expected = "Unable to find path to labels at \"data/test\".")]
fn test_10() {
    let _ = MnistBuilder::new()
        .training_labels_filename("test")
        .finalize();
}

#[test]
#[should_panic(
    expected = "Unable to find path to images at \"data/test\".")]
fn test_11() {
    let _ = MnistBuilder::new()
        .test_images_filename("test")
        .finalize();
}

#[test]
#[should_panic(
    expected = "Unable to find path to labels at \"data/test\".")]
fn test_12() {
    let _ = MnistBuilder::new()
        .test_labels_filename("test")
        .finalize();
}
