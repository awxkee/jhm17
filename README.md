# jhm17

A JPEG encoder written in Rust, forked from
[vstroebel/jpeg-encoder](https://github.com/vstroebel/jpeg-encoder), with
rate-distortion optimization on top of the baseline writer:

- Baseline and progressive compression
- Progressive scan scripts chosen per component on exact coded size (successive approximation, EOB runs, per-scan Huffman tables)
- Chroma subsampling with box averaging by default
- Fractional RGB conversion retained through subsampling and quantization by default
- Optimized huffman tables
- Rate-distortion optimized quantization (trellis with Huffman prices, per-block adaptive λ, clip continuation on white)
- 1, 3 and 4 component colorspaces
- Restart interval
- Custom quantization tables
- SSE2, AVX2 and AArch64 NEON acceleration selected automatically

## Example
```rust
use jhm17::{Encoder, ColorType};

// An array with 4 pixels in RGB format.
let data = [
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 255,
];

// Create new encoder that writes to a file with maximum quality (100)
let mut encoder = Encoder::new_file("some.jpeg", 100)?;

// Encode the data with dimension 2x2
encoder.encode(&data, 2, 2, ColorType::Rgb)?;
```

## RGB sample precision

RGB, RGBA, BGR and BGRA inputs default to `SamplePrecision::All`, which retains
fractional luma and chroma through conversion, subsampling and the integer DCT.
The JPEG quantization tables and decoder requirements stay unchanged.

Use `encoder.set_sample_precision(SamplePrecision::Integer)` for the original
byte-rounded input path, or `SamplePrecision::Luma` to retain fractional luma
with byte-rounded chroma. Import `SamplePrecision` from `jhm17`.

Grayscale, direct YCbCr/CMYK/YCCK and existing custom `ImageBuffer`
implementations retain their original path unless the buffer explicitly
implements the precise-sample hooks.

## Platform support

The crate requires the Rust standard library. The `sse`, `avx2` and `neon` Cargo
features are all enabled by default and can be enabled independently:

- `sse`: SSE2 floating-point DCT on x86/x86_64, selected at runtime.
- `avx2`: AVX2 RGB conversion and integer DCT on x86/x86_64, selected at runtime.
- `neon`: NEON RGB conversion, fractional integer DCT and floating-point DCT on AArch64.

Disabled features, features for other architectures and unavailable CPU features
are skipped in favor of scalar implementations. Use `--no-default-features` for
scalar implementations throughout, or `--no-default-features --features avx2` to
enable only AVX2. RGB/RGBA/BGR/BGRA conversion supports every sample precision.

## Minimum Supported Version of Rust (MSRV)

This crate needs at least 1.89.0 or higher.

## License

Code written for jhm17 is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE))

at your option.

Code inherited from [vstroebel/jpeg-encoder](https://github.com/vstroebel/jpeg-encoder)
remains licensed under either of

- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE))
- MIT license (see [LICENSE](LICENSE-MIT))

at your option.

The forward DCT and progressive Huffman coding are derived from the Independent JPEG
Group's software, see [LICENSE-IJG](LICENSE-IJG).
