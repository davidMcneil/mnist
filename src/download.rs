#![cfg(feature = "download")]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{fs, io};

use curl::easy::Easy;
use std::fs::File;

use pbr::ProgressBar;
use std::convert::TryInto;
use std::thread;

use log::Level;

#[cfg(target_family = "unix")]
use std::os::unix::fs::MetadataExt;
#[cfg(target_family = "windows")]
use std::os::windows::fs::MetadataExt;

#[cfg(target_family = "unix")]
fn file_size(meta: &MetadataExt) -> usize {
    meta.size() as usize
}

#[cfg(target_family = "windows")]
fn file_size(meta: &MetadataExt) -> usize {
    meta.file_size() as usize
}

const ARCHIVE_TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
const ARCHIVE_TRAIN_IMAGES_SIZE: usize = 9912422;
const ARCHIVE_TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
const ARCHIVE_TRAIN_LABELS_SIZE: usize = 28881;
const ARCHIVE_TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
const ARCHIVE_TEST_IMAGES_SIZE: usize = 1648877;
const ARCHIVE_TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";
const ARCHIVE_TEST_LABELS_SIZE: usize = 4542;
const ARCHIVES_TO_DOWNLOAD: &[(&str, usize)] = &[
    (ARCHIVE_TRAIN_IMAGES, ARCHIVE_TRAIN_IMAGES_SIZE),
    (ARCHIVE_TRAIN_LABELS, ARCHIVE_TRAIN_LABELS_SIZE),
    (ARCHIVE_TEST_IMAGES, ARCHIVE_TEST_IMAGES_SIZE),
    (ARCHIVE_TEST_LABELS, ARCHIVE_TEST_LABELS_SIZE),
];

pub(super) fn download_and_extract(
    base_url: &str,
    base_path: &str,
    use_fashion_data: bool,
) -> Result<(), String> {
    let download_dir = PathBuf::from(base_path);
    if !download_dir.exists() {
        log::info!(
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

    for &(archive, size) in ARCHIVES_TO_DOWNLOAD {
        log::info!("Attempting to download and extract {}...", archive);
        download(base_url, archive, size, &download_dir, use_fashion_data)?;
        extract(archive, &download_dir)?;
    }
    Ok(())
}

fn download(
    base_url: &str,
    archive: &str,
    full_size: usize,
    download_dir: &Path,
    use_fashion_data: bool,
) -> Result<(), String> {
    let mut easy = Easy::new();
    let url = Path::new(base_url).join(archive);
    let file_name = download_dir.to_str().unwrap().to_owned() + archive; //.clone();
    if Path::new(&file_name).exists() {
            log::info!(
                "  File {:?} already exists, skipping downloading.",
                file_name
            );
    } else {
        log::info!(
            "- Downloading from file from {} and saving to file as: {}",
            url.to_str().unwrap(),
            file_name
        );

        let mut file = File::create(file_name.clone()).unwrap();
        let pb_thread = thread::spawn(move || {
            let mut pb = ProgressBar::new(full_size.try_into().unwrap());
            pb.format("╢=> ╟");

            let mut current_size = 0;
            while current_size < full_size {
                let meta = fs::metadata(file_name.clone())
                    .expect(&format!("Couldn't get metadata on {:?}", file_name));

                current_size = file_size(&meta);

                pb.set(current_size.try_into().unwrap());
                thread::sleep_ms(10);
            }
            pb.finish_println(" ");
        });

        easy.url(&url.to_str().unwrap()).unwrap();
        easy.write_function(move |data| {
            file.write_all(data).unwrap();
            Ok(data.len())
        })
        .unwrap();
        easy.perform().unwrap();
        pb_thread.join().unwrap();
    }

    Ok(())
}

fn extract(archive_name: &str, download_dir: &Path) -> Result<(), String> {
    let archive = download_dir.join(&archive_name);
    let extract_to = download_dir.join(&archive_name.replace(".gz", ""));
    if extract_to.exists() {
        log::info!(
            "  Extracted file {:?} already exists, skipping extraction.",
            extract_to
        );
    } else {
        log::info!("Extracting archive {:?} to {:?}...", archive, extract_to);
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
