#![cfg(feature = "download")]

extern crate flate2;
extern crate reqwest;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{fs, io};

const BASE_URL: &str = "http://yann.lecun.com/exdb/mnist";
const FASHION_BASE_URL: &str = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com";
const ARCHIVE_TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
const ARCHIVE_TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
const ARCHIVE_TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
const ARCHIVE_TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";
const ARCHIVES_TO_DOWNLOAD: &[&str] = &[
    ARCHIVE_TRAIN_IMAGES,
    ARCHIVE_TRAIN_LABELS,
    ARCHIVE_TEST_IMAGES,
    ARCHIVE_TEST_LABELS,
];

pub(super) fn download_and_extract(base_path: &str, use_fashion_data: bool) -> Result<(), String> {
    let download_dir = PathBuf::from(base_path);
    if !download_dir.exists() {
        println!(
            "Download directory {} does not exists. Creating....",
            download_dir.display()
        );
        fs::create_dir_all(&download_dir).or_else(|e| {
            Err(format!(
                "Failed to create directory {:?}: {:?}",
                download_dir, e
            ))
        })?;
    }
    for archive in ARCHIVES_TO_DOWNLOAD {
        println!("Attempting to download and extract {}...", archive);
        download(&archive, &download_dir, use_fashion_data)?;
        extract(&archive, &download_dir)?;
    }
    Ok(())
}

fn download(archive: &str, download_dir: &Path, use_fashion_data: bool) -> Result<(), String> {
    let url = match use_fashion_data {
        false => format!("{}/{}", BASE_URL, archive),
        true => format!("{}/{}", FASHION_BASE_URL, archive),
    };

    let file_name = download_dir.join(&archive);
    if file_name.exists() {
        println!(
            "  File {:?} already exists, skipping downloading.",
            file_name
        );
    } else {
        println!("  Downloading {} to {:?}...", url, download_dir);
        let f = fs::File::create(&file_name)
            .or_else(|e| Err(format!("Failed to create file {:?}: {:?}", file_name, e)))?;
        let mut writer = io::BufWriter::new(f);
        let mut response =
            reqwest::blocking::get(&url).expect(format!("Failed to download {:?}", url).as_str());

        let _ = io::copy(&mut response, &mut writer).or_else(|e| {
            Err(format!(
                "Failed to to write to file {:?}: {:?}",
                file_name, e
            ))
        })?;
        println!("  Downloading or {} to {:?} done!", archive, download_dir);
    }
    Ok(())
}

fn extract(archive_name: &str, download_dir: &Path) -> Result<(), String> {
    let archive = download_dir.join(&archive_name);
    let extract_to = download_dir.join(&archive_name.replace(".gz", ""));
    if extract_to.exists() {
        println!(
            "  Extracted file {:?} already exists, skipping extraction.",
            extract_to
        );
    } else {
        println!("Extracting archive {:?} to {:?}...", archive, extract_to);
        let file_in = fs::File::open(&archive)
            .or_else(|e| Err(format!("Failed to open archive {:?}: {:?}", archive, e)))?;
        let file_in = io::BufReader::new(file_in);
        let file_out = fs::File::create(&extract_to).or_else(|e| {
            Err(format!(
                "  Failed to create extracted file {:?}: {:?}",
                archive, e
            ))
        })?;
        let mut file_out = io::BufWriter::new(file_out);
        let mut gz = flate2::bufread::GzDecoder::new(file_in);
        let mut v: Vec<u8> = Vec::with_capacity(10 * 1024 * 1024);
        gz.read_to_end(&mut v)
            .or_else(|e| Err(format!("Failed to extract archive {:?}: {:?}", archive, e)))?;
        file_out.write_all(&v).or_else(|e| {
            Err(format!(
                "Failed to write extracted data to {:?}: {:?}",
                archive, e
            ))
        })?;
    }
    Ok(())
}
