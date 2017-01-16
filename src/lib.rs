//! A crate for parsing the [MNIST](http://yann.lecun.com/exdb/mnist/) data set into matricies to be
//!  used by Rust programs.
//!
//! ## About the MNIST Database
//! > The MNIST database (Mixed National Institute of Standards and Technology database) is a large
//! database of handwritten digits that is commonly used for training various image processing
//! systems. The database is also widely used for training and testing in the field of machine
//! learning. <sup><a href="https://en.wikipedia.org/wiki/MNIST_database">wikipedia</a></sup>
//!
//! The MNIST data set contains 70,000 images of handwritten digits and their
//! corresponding labels. The images are 28x28 with pixel values from 0 to 255. The labels are the
//! digits from 0 to 9. By default 60,000 of these images belong to a training set and 10,000 of
//! these images belong to a test set.
//!
//! ## Setup
//! The MNIST data set is a collection of four gzip files and can be found [here]
//! (http://yann.lecun.com/exdb/mnist/). There is one file for each of the following: the training
//! set images, the training set labels, the test set images, and the test set labels. Because of
//! space limitations, the files themselves could not be included in this crate. The four files must
//! be downloaded and extracted. By default, they will be looked for in a "data" directory at the
//! top of level of your crate.
//!
//! ## Usage
//! A [Mnist](struct.Mnist.html) struct is used to represent the various sets of data. In machine
//! learning, it is common to have three sets of data:
//!
//! * Training Set - Used to train a classifier.
//! * Validation Set - Used to regulate the training process (this set is not included in the
//! default MNIST data set partitioning).
//! * Test Set - Used after the training process to determine if the classifier has actually learned
//! something.
//!
//! Each set of data contains a matrix representing the image and a matrix representing the label.
//!
//! A [MnistBuilder](struct.MnistBuilder.html) struct is used to configure how to format the MNIST
//! data, retrieves the data, and returns the [Mnist](struct.Mnist.html) struct. Configuration
//! options include:
//!
//! * where to look for the MNIST data files.
//! * how to format the image matricies.
//! * how to format the label matricies.
//! * how to partition the data between the training, validation, and test sets.
//!
//! ## Examples
//! ```rust,no_run
//! extern crate mnist;
//! extern crate rulinalg;
//!
//! use mnist::{Mnist, MnistBuilder};
//! use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
//!
//! fn main() {
//!     // Deconstruct the returnded Mnist struct.
//!     let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
//!         .image_format_28x28()
//!         .label_format_1x1()
//!         .training_set_length(50_000)
//!         .validation_set_length(10_000)
//!         .test_set_length(10_000)
//!         .finalize();
//!
//!     // Get the label of the first digit.
//!     let first_label = trn_lbl[[0, 0]];
//!     println!("The first digit is a {}.", first_label);
//!
//!     // Get the image of the first digit.
//!     let row_indexes = (0..27).collect::<Vec<_>>();
//!     let first_image = trn_img.select_rows(&row_indexes);
//!     println!("The image looks like... \n{}", first_image);
//!
//!     // Convert the training images to f32 values scaled between 0 and 1.
//!     let trn_img: Matrix<f32> = trn_img.try_into().unwrap() / 255.0;
//!
//!     // Get the image of the first digit and round the values to the nearest tenth.
//!     let first_image = trn_img.select_rows(&row_indexes)
//!         .apply(&|p| (p * 10.0).round() / 10.0);
//!     println!("The image looks like... \n{}", first_image);
//! }
//! ```

#![doc(test(attr(allow(unused_variables), deny(warnings))))]

extern crate byteorder;
extern crate rulinalg;

mod tests;

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use rulinalg::matrix::Matrix;

static BASE_PATH: &'static str = "data/";
static TRN_IMG_FILENAME: &'static str = "train-images.idx3-ubyte";
static TRN_LBL_FILENAME: &'static str = "train-labels.idx1-ubyte";
static TST_IMG_FILENAME: &'static str = "t10k-images.idx3-ubyte";
static TST_LBL_FILENAME: &'static str = "t10k-labels.idx1-ubyte";
static IMG_MAGIC_NUMBER: u32 = 0x00000803;
static LBL_MAGIC_NUMBER: u32 = 0x00000801;
static TRN_LEN: u32 = 60000;
static TST_LEN: u32 = 10000;
static CLASSES: usize = 10;
static ROWS: usize = 28;
static COLS: usize = 28;

#[derive(Debug)]
/// Struct containing image and label matricies for the training, validation, and test sets.
pub struct Mnist {
    /// The training images matrix.
    pub trn_img: Matrix<u8>,
    /// The training labels matrix.
    pub trn_lbl: Matrix<u8>,
    /// The validation images matrix.
    pub val_img: Matrix<u8>,
    /// The validation labels matrix.
    pub val_lbl: Matrix<u8>,
    /// The test images matrix.
    pub tst_img: Matrix<u8>,
    /// The test labels matrix.
    pub tst_lbl: Matrix<u8>,
}

#[derive(Debug)]
/// Struct used for configuring how to load the MNIST data.
///
/// * img_format - Specify how to format the image matricies. Options include:
///     * 1x784 (default) - the 28x28 image is flattened to a 1x784 vector.
///     * 28x28 - the 28x28 image is left as a 28x28 matrix.
/// * lbl_format - Specify how to format the label matricies. Options include:
///     * 1x1 (default) - a single number from 0-9 representing the corresponding digit.
///     * 1x10 - a 1x10 one-hot vector of all 0's except for a 1 at the index of the digit.
///         * ex.) `3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
/// * trn_len - the length of the training set `(default = 60,000)`
/// * val_len - the length of the validation set `(default = 0)`
/// * tst_len - the length of the test set `(default = 10,000)`
/// * base_path - the path to the directory in which to look for the MNIST data files.
/// `(default = "data/")`
/// * trn_img_filename - the filename of the training images data file.
/// `(default = "train-images.idx3-ubyte")`
/// * trn_lbl_filename - the filename of the training labels data file.
/// `(default = "train-labels.idx1-ubyte")`
/// * tst_img_filename - the filename of the test images data file.
/// `(default = "10k-images.idx3-ubyte")`
/// * tst_lbl_filename - the filename of the test labels data file.
/// `(default = "t10k-labels.idx1-ubyte")`
pub struct MnistBuilder<'a> {
    img_format: ImageFormat,
    lbl_format: LabelFormat,
    trn_len: u32,
    val_len: u32,
    tst_len: u32,
    base_path: &'a str,
    trn_img_filename: &'a str,
    trn_lbl_filename: &'a str,
    tst_img_filename: &'a str,
    tst_lbl_filename: &'a str,
}

impl<'a> MnistBuilder<'a> {
    /// Create a new MnistBuilder with defaults set.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///    .finalize();
    /// ```
    pub fn new() -> MnistBuilder<'a> {
        MnistBuilder {
            img_format: ImageFormat::Dimensions1x784,
            lbl_format: LabelFormat::Dimensions1x1,
            trn_len: TRN_LEN,
            val_len: 0,
            tst_len: TST_LEN,
            base_path: BASE_PATH,
            trn_img_filename: TRN_IMG_FILENAME,
            trn_lbl_filename: TRN_LBL_FILENAME,
            tst_img_filename: TST_IMG_FILENAME,
            tst_lbl_filename: TST_LBL_FILENAME,
        }
    }

    /// Set the image format to 1x784.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .image_format_1x784()
    ///     .finalize();
    /// ```
    pub fn image_format_1x784(&mut self) -> &mut MnistBuilder<'a> {
        self.img_format = ImageFormat::Dimensions1x784;
        self
    }

    /// Set the image format to 28x28.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .image_format_28x28()
    ///     .finalize();
    /// ```
    pub fn image_format_28x28(&mut self) -> &mut MnistBuilder<'a> {
        self.img_format = ImageFormat::Dimensions28x28;
        self
    }

    /// Set the labels format to scalar.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .label_format_1x1()
    ///     .finalize();
    /// ```
    pub fn label_format_1x1(&mut self) -> &mut MnistBuilder<'a> {
        self.lbl_format = LabelFormat::Dimensions1x1;
        self
    }

    /// Set the labels format to vector.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .label_format_1x10()
    ///     .finalize();
    /// ```
    pub fn label_format_1x10(&mut self) -> &mut MnistBuilder<'a> {
        self.lbl_format = LabelFormat::Dimensions1x10;
        self
    }

    /// Set the training set length.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .training_set_length(40_000)
    ///     .finalize();
    /// ```
    pub fn training_set_length(&mut self, length: u32) -> &mut MnistBuilder<'a> {
        self.trn_len = length;
        self
    }

    /// Set the validation set length.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .validation_set_length(10_000)
    ///     .finalize();
    /// ```
    pub fn validation_set_length(&mut self, length: u32) -> &mut MnistBuilder<'a> {
        self.val_len = length;
        self
    }

    /// Set the test set length.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .test_set_length(10_000)
    ///     .finalize();
    /// ```
    pub fn test_set_length(&mut self, length: u32) -> &mut MnistBuilder<'a> {
        self.tst_len = length;
        self
    }

    /// Set the base path to look for the MNIST data files.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .base_path("data_sets/mnist")
    ///     .finalize();
    /// ```
    pub fn base_path(&mut self, base_path: &'a str) -> &mut MnistBuilder<'a> {
        self.base_path = base_path;
        self
    }

    /// Set the training images data set filename.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .training_images_filename("training_images")
    ///     .finalize();
    /// ```
    pub fn training_images_filename(&mut self, trn_img_filename: &'a str) -> &mut MnistBuilder<'a> {
        self.trn_img_filename = trn_img_filename;
        self
    }

    /// Set the training labels data set filename.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .training_labels_filename("training_labels")
    ///     .finalize();
    /// ```
    pub fn training_labels_filename(&mut self, trn_lbl_filename: &'a str) -> &mut MnistBuilder<'a> {
        self.trn_lbl_filename = trn_lbl_filename;
        self
    }

    /// Set the test images data set filename.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .test_images_filename("test_images")
    ///     .finalize();
    /// ```
    pub fn test_images_filename(&mut self, tst_img_filename: &'a str) -> &mut MnistBuilder<'a> {
        self.tst_img_filename = tst_img_filename;
        self
    }

    /// Set the test labels data set filename.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .test_labels_filename("test_labels")
    ///     .finalize();
    /// ```
    pub fn test_labels_filename(&mut self, tst_lbl_filename: &'a str) -> &mut MnistBuilder<'a> {
        self.tst_lbl_filename = tst_lbl_filename;
        self
    }

    /// Get the data according to the specified configuration.
    ///
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .finalize();
    /// ```
    ///
    /// # Panics
    /// If `trn_len + val_len + tst_len > 70,000`.
    pub fn finalize(&self) -> Mnist {
        let &MnistBuilder { trn_len, val_len, tst_len, .. } = self;
        let (trn_len, val_len, tst_len) = (trn_len as usize, val_len as usize, tst_len as usize);
        let total_length = trn_len + val_len + tst_len;
        let available_length = (TRN_LEN + TST_LEN) as usize;
        assert!(total_length <= available_length,
                format!("Total data set length ({}) greater than maximum possible length ({}).",
                        total_length,
                        available_length));
        let mut trn_img = images(&Path::new(self.base_path).join(self.trn_img_filename),
                                 TRN_LEN);
        let mut trn_lbl = labels(&Path::new(self.base_path).join(self.trn_lbl_filename),
                                 TRN_LEN);
        let mut tst_img = images(&Path::new(self.base_path).join(self.tst_img_filename),
                                 TST_LEN);
        let mut tst_lbl = labels(&Path::new(self.base_path).join(self.tst_lbl_filename),
                                 TST_LEN);
        trn_img.append(&mut tst_img);
        trn_lbl.append(&mut tst_lbl);
        let mut val_img = trn_img.split_off(trn_len * ROWS * COLS);
        let mut val_lbl = trn_lbl.split_off(trn_len);
        let mut tst_img = val_img.split_off(val_len * ROWS * COLS);
        let mut tst_lbl = val_lbl.split_off(val_len);
        tst_img.split_off(tst_len * ROWS * COLS);
        tst_lbl.split_off(tst_len);
        let (trn_img, val_img, tst_img) = match self.img_format {
            ImageFormat::Dimensions1x784 => {
                (Matrix::new(trn_len, ROWS * COLS, trn_img),
                 Matrix::new(val_len, ROWS * COLS, val_img),
                 Matrix::new(tst_len, ROWS * COLS, tst_img))
            }
            ImageFormat::Dimensions28x28 => {
                (Matrix::new(trn_len * ROWS, COLS, trn_img),
                 Matrix::new(val_len * ROWS, COLS, val_img),
                 Matrix::new(tst_len * ROWS, COLS, tst_img))
            }
        };
        let (trn_lbl, val_lbl, tst_lbl) = match self.lbl_format {
            LabelFormat::Dimensions1x1 => {
                (Matrix::new(trn_len, 1, trn_lbl),
                 Matrix::new(val_len, 1, val_lbl),
                 Matrix::new(tst_len, 1, tst_lbl))
            }
            LabelFormat::Dimensions1x10 => {
                fn scalar_vector2one_hot_vector(v: Vec<u8>) -> Vec<u8> {
                    v.iter()
                        .map(|&i| {
                            let mut v = vec![0; CLASSES as usize];
                            v[i as usize] = 1;
                            v
                        })
                        .flat_map(|e| e)
                        .collect()
                }
                let trn_lbl: Vec<_> = scalar_vector2one_hot_vector(trn_lbl);
                let val_lbl: Vec<_> = scalar_vector2one_hot_vector(val_lbl);
                let tst_lbl: Vec<_> = scalar_vector2one_hot_vector(tst_lbl);
                (Matrix::new(trn_len, CLASSES, trn_lbl),
                 Matrix::new(val_len, CLASSES, val_lbl),
                 Matrix::new(tst_len, CLASSES, tst_lbl))
            }
        };
        Mnist {
            trn_img: trn_img,
            trn_lbl: trn_lbl,
            val_img: val_img,
            val_lbl: val_lbl,
            tst_img: tst_img,
            tst_lbl: tst_lbl,
        }
    }
}

#[derive(Debug)]
enum ImageFormat {
    Dimensions1x784,
    Dimensions28x28,
}

#[derive(Debug)]
enum LabelFormat {
    Dimensions1x1,
    Dimensions1x10,
}

fn labels(path: &Path, expected_length: u32) -> Vec<u8> {
    let mut file = File::open(path)
        .expect(&format!("Unable to find path to labels at {:?}.", path));
    let magic_number = file.read_u32::<BigEndian>()
        .expect(&format!("Unable to read magic number from {:?}.", path));
    assert!(LBL_MAGIC_NUMBER == magic_number,
            format!("Expected magic number {} got {}.",
                    LBL_MAGIC_NUMBER,
                    magic_number));
    let length = file.read_u32::<BigEndian>()
        .expect(&format!("Unable to length from {:?}.", path));
    assert!(expected_length == length,
            format!("Expected data set length of {} got {}.",
                    expected_length,
                    length));
    file.bytes().map(|b| b.unwrap()).collect()
}

fn images(path: &Path, expected_length: u32) -> Vec<u8> {
    let mut file = File::open(path)
        .expect(&format!("Unable to find path to images at {:?}.", path));
    let magic_number = file.read_u32::<BigEndian>()
        .expect(&format!("Unable to read magic number from {:?}.", path));
    assert!(IMG_MAGIC_NUMBER == magic_number,
            format!("Expected magic number {} got {}.",
                    IMG_MAGIC_NUMBER,
                    magic_number));
    let length = file.read_u32::<BigEndian>()
        .expect(&format!("Unable to length from {:?}.", path));
    assert!(expected_length == length,
            format!("Expected data set length of {} got {}.",
                    expected_length,
                    length));
    let rows = file.read_u32::<BigEndian>()
        .expect(&format!("Unable to number of rows from {:?}.", path)) as usize;
    assert!(ROWS == rows,
            format!("Expected rows length of {} got {}.", ROWS, rows));
    let cols = file.read_u32::<BigEndian>()
        .expect(&format!("Unable to number of columns from {:?}.", path)) as usize;
    assert!(COLS == cols,
            format!("Expected cols length of {} got {}.", COLS, cols));
    file.bytes().map(|b| b.unwrap()).collect()
}
