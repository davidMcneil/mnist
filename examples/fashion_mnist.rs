extern crate minifb;
use minifb::{Key, ScaleMode, Window, WindowOptions};

extern crate mnist;
use mnist::*;

// $ cargo run --features download --example fashion_mnist
// minifb requires `$ sudo apt install libxkbcommon-dev libwayland-cursor0 libwayland-dev`

fn main() {
    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .base_path("data/fashion/") // Comment out this and `use_fashion_data()` to run on the original MNIST
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .download_and_extract()
        .use_fashion_data() // Commnent out this and the changed `.base_path()` to run on the original MNIST
        .finalize();

    let item_num = 0;
    return_item_description_from_number(trn_lbl[item_num]);
    display_img(trn_img[item_num * 784..item_num * 784 + 784].to_vec());
}

fn display_img(input: Vec<u8>) {
    // println!("img_vec: {:?}",img_vec);
    let mut buffer: Vec<u32> = Vec::with_capacity(28 * 28);
    for px in 0..784 {
        let temp: [u8; 4] = [input[px], input[px], input[px], 255u8];
        // println!("temp: {:?}",temp);
        buffer.push(u32::from_le_bytes(temp));
    }

    let (window_width, window_height) = (600, 600);
    let mut window = Window::new(
        "Test - ESC to exit",
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer, 28, 28).unwrap();
    }
}

fn return_item_description_from_number(val: u8) {
    let description = match val {
        0 => "T-shirt/top",
        1 => "Trouser",
        2 => "Pullover",
        3 => "Dress",
        4 => "Coat",
        5 => "Sandal",
        6 => "Shirt",
        7 => "Sneaker",
        8 => "Bag",
        9 => "Ankle boot",
        _ => panic!("An unrecognized label was used..."),
    };
    println!("Based on the '{}' label, this image should be a: {} ", val, description);
}
