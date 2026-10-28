/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use jhm17::{
    ColorType, Encoder, ImageBuffer, JpegColorType, ProgressiveMode, RdoOptions, SamplePrecision,
    SamplingFactor, rgb_to_ycbcr,
};

fn source(width: u16, height: u16) -> Vec<u8> {
    (0..usize::from(width) * usize::from(height))
        .flat_map(|i| {
            let x = i % usize::from(width);
            let y = i / usize::from(width);
            [
                ((x * 17 + y * 3 + 23) % 256) as u8,
                ((x * 5 + y * 19 + 109) % 256) as u8,
                ((x * 11 + y * 7 + 207) % 256) as u8,
            ]
        })
        .collect()
}

fn encode(
    data: &[u8],
    color: ColorType,
    size: (u16, u16),
    precision: Option<SamplePrecision>,
    sampling: SamplingFactor,
    progression: Option<ProgressiveMode>,
    rdo: bool,
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = Encoder::new(&mut out, 93);
    if let Some(precision) = precision {
        encoder.set_sample_precision(precision);
    }
    encoder.set_sampling_factor(sampling);
    encoder.set_progressive_mode(progression);
    encoder.set_restart_interval(5);
    encoder.set_optimized_huffman_tables(true);
    if rdo {
        encoder.set_rdo(Some(RdoOptions::default()));
    }
    encoder.encode(data, size.0, size.1, color).unwrap();
    out
}

fn decoded(jpeg: &[u8], size: (u16, u16)) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg);
    let pixels = decoder.decode().unwrap();
    let info = decoder.info().unwrap();
    assert_eq!((info.width, info.height), size);
    assert!(!pixels.is_empty());
    pixels
}

/// Inspect transmitted DQT entries, independently of the encoder's internal scale.
fn quantization_tables(jpeg: &[u8]) -> Vec<(u8, Vec<u16>)> {
    assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
    let mut pos = 2;
    let mut tables = Vec::new();
    loop {
        assert_eq!(jpeg[pos], 0xff);
        while jpeg[pos] == 0xff {
            pos += 1;
        }
        let marker = jpeg[pos];
        pos += 1;
        if marker == 0xda || marker == 0xd9 {
            break;
        }
        let length = usize::from(u16::from_be_bytes([jpeg[pos], jpeg[pos + 1]]));
        assert!(length >= 2);
        let end = pos + length;
        pos += 2;
        if marker == 0xdb {
            while pos < end {
                let descriptor = jpeg[pos];
                pos += 1;
                let wide = descriptor >> 4;
                assert!(wide <= 1);
                let mut values = Vec::with_capacity(64);
                for _ in 0..64 {
                    let value = if wide == 0 {
                        let value = u16::from(jpeg[pos]);
                        pos += 1;
                        value
                    } else {
                        let value = u16::from_be_bytes([jpeg[pos], jpeg[pos + 1]]);
                        pos += 2;
                        value
                    };
                    values.push(value);
                }
                tables.push((descriptor & 15, values));
            }
        }
        pos = end;
    }
    tables.sort_by_key(|(id, _)| *id);
    assert!(!tables.is_empty());
    tables
}

#[test]
fn default_retains_fractional_rgb_samples() {
    let size = (35, 19);
    let rgb = source(size.0, size.1);
    for rdo in [false, true] {
        let default = encode(
            &rgb,
            ColorType::Rgb,
            size,
            None,
            SamplingFactor::F_2_2,
            None,
            rdo,
        );
        let all = encode(
            &rgb,
            ColorType::Rgb,
            size,
            Some(SamplePrecision::All),
            SamplingFactor::F_2_2,
            None,
            rdo,
        );
        let integer = encode(
            &rgb,
            ColorType::Rgb,
            size,
            Some(SamplePrecision::Integer),
            SamplingFactor::F_2_2,
            None,
            rdo,
        );
        assert_eq!(default, all, "the default must enable native precision");
        assert_ne!(
            decoded(&all, size),
            decoded(&integer, size),
            "fractional input must reach reconstruction"
        );
    }
}

#[test]
fn transmitted_tables_do_not_depend_on_precision_or_scan_layout() {
    let size = (17, 9);
    let rgb = source(size.0, size.1);
    let mut reference = None;
    for precision in [
        SamplePrecision::Integer,
        SamplePrecision::Luma,
        SamplePrecision::All,
    ] {
        for progression in [
            None,
            Some(ProgressiveMode::Standard),
            Some(ProgressiveMode::Optimized),
        ] {
            let jpeg = encode(
                &rgb,
                ColorType::Rgb,
                size,
                Some(precision),
                SamplingFactor::F_2_2,
                progression,
                true,
            );
            let tables = quantization_tables(&jpeg);
            if let Some(reference) = &reference {
                assert_eq!(&tables, reference);
            } else {
                reference = Some(tables);
            }
        }
    }
}

#[test]
fn odd_edges_and_restarts_decode_identically_across_scan_layouts() {
    let samplings = [
        SamplingFactor::F_1_1,
        SamplingFactor::F_2_2,
        SamplingFactor::F_2_1,
        SamplingFactor::F_1_2,
        SamplingFactor::F_4_1,
        SamplingFactor::F_4_2,
        SamplingFactor::F_1_4,
        SamplingFactor::F_2_4,
    ];
    for size in [(17, 9), (35, 19)] {
        let rgb = source(size.0, size.1);
        for sampling in samplings {
            for rdo in [false, true] {
                let sequential = encode(
                    &rgb,
                    ColorType::Rgb,
                    size,
                    Some(SamplePrecision::All),
                    sampling,
                    None,
                    rdo,
                );
                let expected = decoded(&sequential, size);
                for progression in [ProgressiveMode::Standard, ProgressiveMode::Optimized] {
                    let progressive = encode(
                        &rgb,
                        ColorType::Rgb,
                        size,
                        Some(SamplePrecision::All),
                        sampling,
                        Some(progression),
                        rdo,
                    );
                    assert_eq!(
                        decoded(&progressive, size),
                        expected,
                        "size={size:?}, sampling={sampling:?}, rdo={rdo}, progression={progression:?}"
                    );
                    assert_eq!(
                        quantization_tables(&progressive),
                        quantization_tables(&sequential)
                    );
                }
            }
        }
    }
}

#[test]
fn rgb_channel_orders_and_ignored_alpha_produce_the_same_native_jpeg() {
    for size in [(17, 9), (35, 19)] {
        let rgb = source(size.0, size.1);
        let rgba: Vec<_> = rgb
            .chunks_exact(3)
            .enumerate()
            .flat_map(|(i, p)| [p[0], p[1], p[2], i as u8])
            .collect();
        let bgr: Vec<_> = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        let bgra: Vec<_> = rgb
            .chunks_exact(3)
            .enumerate()
            .flat_map(|(i, p)| [p[2], p[1], p[0], !(i as u8)])
            .collect();
        for precision in [SamplePrecision::Luma, SamplePrecision::All] {
            let expected = encode(
                &rgb,
                ColorType::Rgb,
                size,
                Some(precision),
                SamplingFactor::F_2_2,
                None,
                true,
            );
            decoded(&expected, size);
            for (bytes, color) in [
                (&rgba, ColorType::Rgba),
                (&bgr, ColorType::Bgr),
                (&bgra, ColorType::Bgra),
            ] {
                assert_eq!(
                    encode(
                        bytes,
                        color,
                        size,
                        Some(precision),
                        SamplingFactor::F_2_2,
                        None,
                        true
                    ),
                    expected,
                    "color={color:?}, precision={precision:?}, size={size:?}"
                );
            }
        }
    }
}

#[test]
fn byte_component_inputs_keep_their_existing_reconstruction() {
    let size = (17, 9);
    for (color, channels) in [
        (ColorType::Luma, 1),
        (ColorType::Ycbcr, 3),
        (ColorType::Cmyk, 4),
        (ColorType::CmykAsYcck, 4),
        (ColorType::Ycck, 4),
    ] {
        let bytes: Vec<u8> = (0..usize::from(size.0) * usize::from(size.1) * channels)
            .map(|i| (i * 73 + i / 13) as u8)
            .collect();
        for rdo in [false, true] {
            for progression in [None, Some(ProgressiveMode::Standard)] {
                let integer = encode(
                    &bytes,
                    color,
                    size,
                    Some(SamplePrecision::Integer),
                    SamplingFactor::F_2_2,
                    progression,
                    rdo,
                );
                let all = encode(
                    &bytes,
                    color,
                    size,
                    Some(SamplePrecision::All),
                    SamplingFactor::F_2_2,
                    progression,
                    rdo,
                );
                assert_eq!(
                    all, integer,
                    "color={color:?}, rdo={rdo}, progression={progression:?}"
                );
                assert_eq!(decoded(&all, size), decoded(&integer, size));
            }
        }
    }
}

struct ByteRgb<'a> {
    pixels: &'a [u8],
    width: u16,
    height: u16,
}

impl ImageBuffer for ByteRgb<'_> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Ycbcr
    }
    fn width(&self) -> u16 {
        self.width
    }
    fn height(&self) -> u16 {
        self.height
    }
    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let row = usize::from(y) * usize::from(self.width) * 3;
        for pixel in self.pixels[row..row + usize::from(self.width) * 3]
            .as_chunks::<3>()
            .0
            .iter()
        {
            let (y, cb, cr) = rgb_to_ycbcr(pixel[0], pixel[1], pixel[2]);
            buffers[0].push(y);
            buffers[1].push(cb);
            buffers[2].push(cr);
        }
    }
}

#[test]
fn existing_custom_image_buffers_need_no_precision_implementation() {
    let size = (35, 19);
    let rgb = source(size.0, size.1);
    let image = ByteRgb {
        pixels: &rgb,
        width: size.0,
        height: size.1,
    };
    assert!(!image.supports_precise_samples());
    let mut untouched = core::array::from_fn(|c| vec![c as u16, 12345]);
    let before = untouched.clone();
    assert!(!image.fill_buffers_precise(0, &mut untouched, SamplePrecision::All));
    assert_eq!(untouched, before);
    for progression in [None, Some(ProgressiveMode::Standard)] {
        let mut reference = None;
        for precision in [SamplePrecision::Integer, SamplePrecision::All] {
            let mut jpeg = Vec::new();
            let mut encoder = Encoder::new(&mut jpeg, 93);
            encoder.set_sample_precision(precision);
            encoder.set_sampling_factor(SamplingFactor::F_2_2);
            encoder.set_progressive_mode(progression);
            encoder.set_restart_interval(5);
            encoder.set_rdo(Some(RdoOptions::default()));
            encoder
                .encode_image(ByteRgb {
                    pixels: &rgb,
                    width: size.0,
                    height: size.1,
                })
                .unwrap();
            decoded(&jpeg, size);
            if let Some(reference) = &reference {
                assert_eq!(&jpeg, reference);
            } else {
                reference = Some(jpeg);
            }
        }
    }
}
