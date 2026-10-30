use jhm17::{ColorType, Encoder, RdoOptions, SamplingFactor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: encode <input> [output.jpg] [quality]")?;
    let output = args.next().unwrap_or_else(|| "out.jpg".to_string());
    let quality: u8 = args.next().map(|q| q.parse()).transpose()?.unwrap_or(85);

    let img = image::open(&input)?.to_rgb8();
    let (width, height) = img.dimensions();

    let mut encoder = Encoder::new_file(&output, quality)?;
    encoder.set_sampling_factor(SamplingFactor::F_2_2);
    encoder.set_optimized_huffman_tables(true);
    encoder.set_rdo(Some(RdoOptions::default()));

    let start = Instant::now();
    encoder.encode(
        img.as_raw(),
        width.try_into()?,
        height.try_into()?,
        ColorType::Rgb,
    )?;
    println!(
        "{input} ({width}x{height}) -> {output} q{quality}: {} bytes in {:?}",
        std::fs::metadata(&output)?.len(),
        start.elapsed()
    );
    Ok(())
}
