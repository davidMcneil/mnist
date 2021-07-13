extern crate mnist;
use image::*;
use mnist::*;
use ndarray::prelude::*;
use show_image::{make_window_full, Event, WindowOptions};

fn main() {
    let (trn_size, _rows, _cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        // .use_fashion_data() // Comment out this and the changed `.base_path()` to run on the original MNIST
        //.base_url("<some_other_url>") // Since the base url is sometimes down due to high demand, you can replace is with another
        .base_path("data/") // Comment out this and `use_fashion_data()` to run on the original MNIST
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize();

    let item_num = 3;
    return_item_description_from_number(trn_lbl[item_num]);

    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .mapv(|x| x as f32 / 256.);

    let image = bw_ndarray2_to_rgb_image(train_data.slice(s![item_num, .., ..]).to_owned());
    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [100, 100],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    let window = make_window_full(window_options).unwrap();
    window.set_image(image, "test_result").unwrap();

    for event in window.events() {
        if let Event::KeyboardEvent(event) = event {
            if event.key == show_image::KeyCode::Escape {
                break;
            }
        }
    }

    show_image::stop().unwrap();
}

fn return_item_description_from_number(val: u8) {
    let description = match val {
        0 => "0",
        1 => "1",
        2 => "2",
        3 => "3",
        4 => "4",
        5 => "5",
        6 => "6",
        7 => "7",
        8 => "8",
        9 => "9",
        _ => panic!("An unrecognized label was used..."),
    };
    println!(
        "Based on the '{}' label, this image should be a: {} ",
        val, description
    );
    println!("Hit [ ESC ] to exit...");
}

fn bw_ndarray2_to_rgb_image(arr: Array2<f32>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (width, height) = (arr.ncols(), arr.ncols());
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let val = (arr[[y, x]] * 255.) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([val, val, val]))
        }
    }
    img
}
