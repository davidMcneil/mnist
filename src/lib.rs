//! A crate for parsing the [MNIST](http://yann.lecun.com/exdb/mnist/) data set into vectors to be
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
//! Each set of data contains a vector representing the image and a vector representing the label.
//! The vectors are always completely flattened. For example, the default image test set contains
//! 60,000 images. Therefore the vector size will be
//! 60,000 images x 28 rows x 28 cols = 47,040,000 elements in the vector.
//!
//! A [MnistBuilder](struct.MnistBuilder.html) struct is used to configure how to format the MNIST
//! data, retrieves the data, and returns the [Mnist](struct.Mnist.html) struct. Configuration
//! options include:
//!
//! * where to look for the MNIST data files.
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
//!     let (trn_size, rows, cols) = (50_000, 28, 28);
//!
//!     // Deconstruct the returned Mnist struct.
//!     let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
//!         .label_format_digit()
//!         .training_set_length(trn_size)
//!         .validation_set_length(10_000)
//!         .test_set_length(10_000)
//!         .finalize();
//!
//!     // Get the label of the first digit.
//!     let first_label = trn_lbl[0];
//!     println!("The first digit is a {}.", first_label);
//!
//!     // Convert the flattened training images vector to a matrix.
//!     let trn_img = Matrix::new((trn_size * rows) as usize, cols as usize, trn_img);
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

mod tests;

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

static BASE_PATH: &'static str = "data/";
static TRN_IMG_FILENAME: &'static str = "train-images-idx3-ubyte";
static TRN_LBL_FILENAME: &'static str = "train-labels-idx1-ubyte";
static TST_IMG_FILENAME: &'static str = "t10k-images-idx3-ubyte";
static TST_LBL_FILENAME: &'static str = "t10k-labels-idx1-ubyte";
static IMG_MAGIC_NUMBER: u32 = 0x00000803;
static LBL_MAGIC_NUMBER: u32 = 0x00000801;
static TRN_LEN: u32 = 60000;
static TST_LEN: u32 = 10000;
static CLASSES: usize = 10;
static ROWS: usize = 28;
static COLS: usize = 28;

#[derive(Debug)]
/// Struct containing image and label vectors for the training, validation, and test sets.
pub struct Mnist {
    /// The training images vector.
    pub trn_img: Vec<u8>,
    /// The training labels vector.
    pub trn_lbl: Vec<u8>,
    /// The validation images vector.
    pub val_img: Vec<u8>,
    /// The validation labels vector.
    pub val_lbl: Vec<u8>,
    /// The test images vector.
    pub tst_img: Vec<u8>,
    /// The test labels vector.
    pub tst_lbl: Vec<u8>,
}

#[derive(Debug)]
/// Struct used for configuring how to load the MNIST data.
///
/// * lbl_format - Specify how to format the label vectors. Options include:
///     * Digit (default) - a single number from 0-9 representing the corresponding digit.
///     * OneHotVector - a 1x10 one-hot vector of all 0's except for a 1 at the index of the digit.
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
            lbl_format: LabelFormat::Digit,
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

    /// Set the labels format to scalar.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .label_format_digit()
    ///     .finalize();
    /// ```
    pub fn label_format_digit(&mut self) -> &mut MnistBuilder<'a> {
        self.lbl_format = LabelFormat::Digit;
        self
    }

    /// Set the labels format to vector.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use mnist::MnistBuilder;
    /// let mnist = MnistBuilder::new()
    ///     .label_format_one_hot()
    ///     .finalize();
    /// ```
    pub fn label_format_one_hot(&mut self) -> &mut MnistBuilder<'a> {
        self.lbl_format = LabelFormat::OneHotVector;
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
        if self.lbl_format == LabelFormat::OneHotVector {
            fn digit2one_hot(v: Vec<u8>) -> Vec<u8> {
                v.iter()
                    .map(|&i| {
                        let mut v = vec![0; CLASSES as usize];
                        v[i as usize] = 1;
                        v
                    })
                    .flat_map(|e| e)
                    .collect()
            }
            trn_lbl = digit2one_hot(trn_lbl);
            val_lbl = digit2one_hot(val_lbl);
            tst_lbl = digit2one_hot(tst_lbl);
        }

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

#[derive(Debug, PartialEq)]
enum LabelFormat {
    Digit,
    OneHotVector,
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
    // Read whole file in memory
    let mut content: Vec<u8> = Vec::new();
    let mut file = {
        let mut fh = File::open(path)
            .expect(&format!("Unable to find path to images at {:?}.", path));
        let _ = fh.read_to_end(&mut content).expect(&format!("Unable to read whole file in memory ({})", path.display()));
        // The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
        // used with a `Vec` directly, it requires a slice.
        &content[..]
    };

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
    // Convert `file` from a Vec to a slice.
    file.to_vec()
}
