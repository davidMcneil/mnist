#![cfg(test)]

use super::*;
use ndarray::prelude::*;

#[test]
fn test_example() {

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);
    
}

#[test]
fn test_00() {
    let mnist = MnistBuilder::new().finalize();
    assert!(mnist.trn_img.len() == 60_000 * 28 * 28);
    assert!(mnist.trn_lbl.len() == 60_000);
    assert!(mnist.val_img.len() == 0);
    assert!(mnist.val_lbl.len() == 0);
    assert!(mnist.tst_img.len() == 10_000 * 28 * 28);
    assert!(mnist.tst_lbl.len() == 10_000);
    assert!(mnist.trn_lbl[0] == 5);
    assert!(mnist.tst_lbl[0] == 7);
}

#[test]
fn test_01() {
    let mnist = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    assert!(mnist.trn_img.len() == 50_000 * 28 * 28);
    assert!(mnist.trn_lbl.len() == 50_000 * 10);
    assert!(mnist.val_img.len() == 10_000 * 28 * 28);
    assert!(mnist.val_lbl.len() == 10_000 * 10);
    assert!(mnist.tst_img.len() == 10_000 * 28 * 28);
    assert!(mnist.tst_lbl.len() == 10_000 * 10);
    assert!(mnist.trn_lbl[0] == 0);
    assert!(mnist.trn_lbl[9] == 0);
    assert!(mnist.trn_lbl[5] == 1);
    assert!(mnist.tst_lbl[0] == 0);
    assert!(mnist.tst_lbl[9] == 0);
    assert!(mnist.tst_lbl[7] == 1);
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000)."
)]
fn test_02() {
    let _ = MnistBuilder::new()
        .training_set_length(50_001)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000)."
)]
fn test_03() {
    let _ = MnistBuilder::new()
        .training_set_length(50_000)
        .validation_set_length(10_001)
        .test_set_length(10_000)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000)."
)]
fn test_04() {
    let _ = MnistBuilder::new()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_001)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Total data set length (70300) greater than maximum possible length (70000)."
)]
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
        .label_format_one_hot()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .base_path("data_sets/mnist")
        .training_images_filename("training_images")
        .training_labels_filename("training_labels")
        .test_images_filename("test_images")
        .test_labels_filename("test_labels")
        .finalize();
    assert!(mnist.trn_img.len() == 50_000 * 28 * 28);
    assert!(mnist.trn_lbl.len() == 50_000 * 10);
    assert!(mnist.val_img.len() == 10_000 * 28 * 28);
    assert!(mnist.val_lbl.len() == 10_000 * 10);
    assert!(mnist.tst_img.len() == 10_000 * 28 * 28);
    assert!(mnist.tst_lbl.len() == 10_000 * 10);
    assert!(mnist.trn_lbl[5] == 1);
    assert!(mnist.tst_lbl[7] == 1);
}

#[test]
#[should_panic(
    expected = "Total data set length (70001) greater than maximum possible length (70000)."
)]
fn test_07() {
    let _ = MnistBuilder::new()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_001)
        .finalize();
}

#[test]
#[should_panic(
    expected = "Unable to find path to images at \"wrong/path/train-images-idx3-ubyte\"."
)]
fn test_08() {
    let _ = MnistBuilder::new().base_path("wrong/path").finalize();
}

#[test]
#[should_panic(expected = "Unable to find path to images at \"data/test\".")]
fn test_09() {
    let _ = MnistBuilder::new()
        .training_images_filename("test")
        .finalize();
}

#[test]
#[should_panic(expected = "Unable to find path to labels at \"data/test\".")]
fn test_10() {
    let _ = MnistBuilder::new()
        .training_labels_filename("test")
        .finalize();
}

#[test]
#[should_panic(expected = "Unable to find path to images at \"data/test\".")]
fn test_11() {
    let _ = MnistBuilder::new().test_images_filename("test").finalize();
}

#[test]
#[should_panic(expected = "Unable to find path to labels at \"data/test\".")]
fn test_12() {
    let _ = MnistBuilder::new().test_labels_filename("test").finalize();
}

#[test]
fn normalize_vector() {
    use super::normalize_vector;

    let v: Vec<u8> = vec![0, 1, 2, 127, 128, 129, 254, 255];
    let normalized_v: Vec<f32> = normalize_vector(&v);
    let expected: Vec<f32> = vec![
        0.0, 0.00392157, 0.00784314, 0.49803922, 0.50196078, 0.50588235, 0.99607843, 1.0,
    ];

    expected
        .iter()
        .zip(normalized_v.iter())
        .for_each(|(value, expected)| assert!((value - expected).abs() < 1.0e-6));
}
