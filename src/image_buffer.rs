#![allow(clippy::identity_op)]

use alloc::vec::Vec;

use crate::encoder::{JpegColorType, SamplePrecision};

/// Conversion from RGB to YCbCr
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    // To avoid floating point math this scales everything by 2^16 which gives
    // a precision of approx 4 digits.
    //
    // Non scaled conversion:
    // Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
    // Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128

    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    let y = 19595 * r + 38470 * g + 7471 * b;
    let cb = -11059 * r - 21709 * g + 32768 * b + (128 << 16);
    let cr = 32768 * r - 27439 * g - 5329 * b + (128 << 16);

    let y = (y + 0x7FFF) >> 16;
    let cb = (cb + 0x7FFF) >> 16;
    let cr = (cr + 0x7FFF) >> 16;

    (y as u8, cb as u8, cr as u8)
}

/// Conversion from CMYK to YCCK (YCbCrK)
#[inline]
pub fn cmyk_to_ycck(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8, u8) {
    let (y, cb, cr) = rgb_to_ycbcr(c, m, y);
    (y, cb, cr, 255 - k)
}

/// # Buffer used as input value for image encoding
///
/// Image encoding with [Encoder::encode_image](crate::Encoder::encode_image) needs an ImageBuffer
/// as input for the image data. For convenience the [Encoder::encode](crate::Encoder::encode)
/// function contains implementations for common byte based pixel formats.
/// Users that needs other pixel formats or don't have the data available as byte slices
/// can create their own buffer implementations.
///
/// ## Example: ImageBuffer implementation for RgbImage from the `image` crate
/// ```no_compile
/// use image::RgbImage;
/// use jhm17::{ImageBuffer, JpegColorType, rgb_to_ycbcr};
///
/// pub struct RgbImageBuffer {
///     image: RgbImage,
/// }
///
/// impl ImageBuffer for RgbImageBuffer {
///     fn get_jpeg_color_type(&self) -> JpegColorType {
///         // Rgb images are encoded as YCbCr in JFIF files
///         JpegColorType::Ycbcr
///     }
///
///     fn width(&self) -> u16 {
///         self.image.width() as u16
///     }
///
///     fn height(&self) -> u16 {
///         self.image.height() as u16
///     }
///
///     fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]){
///         for x in 0..self.width() {
///             let pixel = self.image.get_pixel(x as u32 ,y as u32);
///
///             let (y,cb,cr) = rgb_to_ycbcr(pixel[0], pixel[1], pixel[2]);
///
///             // For YCbCr the 4th buffer is not used
///             buffers[0].push(y);
///             buffers[1].push(cb);
///             buffers[2].push(cr);
///         }
///     }
/// }
///
/// ```
pub trait ImageBuffer {
    /// The color type used in the image encoding
    fn get_jpeg_color_type(&self) -> JpegColorType;

    /// Width of the image
    fn width(&self) -> u16;

    /// Height of the image
    fn height(&self) -> u16;

    /// Add color values for the row to color component buffers
    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]);

    /// Whether this buffer can provide fractional component samples.
    fn supports_precise_samples(&self) -> bool {
        false
    }

    /// Append a row of unsigned Q8 component samples (`sample * 256`).
    ///
    /// Return `true` after appending the row. Return `false` without changing
    /// any buffer when this input or precision mode is unsupported. The default
    /// preserves the byte-sample path for existing custom implementations.
    fn fill_buffers_precise(
        &self,
        _y: u16,
        _buffers: &mut [Vec<u16>; 4],
        _precision: SamplePrecision,
    ) -> bool {
        false
    }
}

pub(crate) struct GrayImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for GrayImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Luma
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let line = get_line(self.0, y, self.width(), 1);

        for &pixel in line {
            buffers[0].push(pixel);
        }
    }
}

#[inline(always)]
pub(crate) fn get_line(data: &[u8], y: u16, width: u16, num_colors: usize) -> &[u8] {
    let width = usize::from(width);
    let y = usize::from(y);

    let start = y * width * num_colors;
    let end = start + width * num_colors;

    &data[start..end]
}

#[inline(always)]
pub(crate) fn rgb_to_ycbcr_q8(r: u8, g: u8, b: u8, precise_chroma: bool) -> (u16, u16, u16) {
    let (r, g, b) = (i32::from(r), i32::from(g), i32::from(b));
    let y = 19595 * r + 38470 * g + 7471 * b;
    let cb = -11059 * r - 21709 * g + 32768 * b + (128 << 16);
    let cr = 32768 * r - 27439 * g - 5329 * b + (128 << 16);
    let y = ((y + 127) >> 8) as u16;
    let (cb, cr) = if precise_chroma {
        (((cb + 127) >> 8) as u16, ((cr + 127) >> 8) as u16)
    } else {
        // Match the existing byte conversion, including its half-step ties.
        (
            (((cb + 0x7fff) >> 16) as u16) << 8,
            (((cr + 0x7fff) >> 16) as u16) << 8,
        )
    };
    (y, cb, cr)
}

/// Scalar RGB-family conversion retaining fractional component samples.
#[inline]
pub(crate) fn fill_precise_rgb<const N: usize, const R: usize, const G: usize, const B: usize>(
    data: &[u8],
    y: u16,
    width: u16,
    buffers: &mut [Vec<u16>; 4],
    precision: SamplePrecision,
) -> bool {
    let precise_chroma = match precision {
        SamplePrecision::Integer => return false,
        SamplePrecision::Luma => false,
        SamplePrecision::All => true,
    };
    let line = get_line(data, y, width, N);
    let [ys, cbs, crs, _] = buffers;
    ys.reserve(usize::from(width));
    cbs.reserve(usize::from(width));
    crs.reserve(usize::from(width));
    const CHUNK: usize = 16;
    let mut y_chunk = [0u16; CHUNK];
    let mut cb_chunk = [0u16; CHUNK];
    let mut cr_chunk = [0u16; CHUNK];
    let mut chunks = line.chunks_exact(N * CHUNK);
    for chunk in &mut chunks {
        for i in 0..CHUNK {
            (y_chunk[i], cb_chunk[i], cr_chunk[i]) = rgb_to_ycbcr_q8(
                chunk[i * N + R],
                chunk[i * N + G],
                chunk[i * N + B],
                precise_chroma,
            );
        }
        ys.extend_from_slice(&y_chunk);
        cbs.extend_from_slice(&cb_chunk);
        crs.extend_from_slice(&cr_chunk);
    }
    for pixel in chunks.remainder().as_chunks::<N>().0.iter() {
        let (y, cb, cr) = rgb_to_ycbcr_q8(pixel[R], pixel[G], pixel[B], precise_chroma);
        ys.push(y);
        cbs.push(cb);
        crs.push(cr);
    }
    true
}

macro_rules! ycbcr_image {
    ($name:ident, $num_colors:expr, $o1:expr, $o2:expr, $o3:expr) => {
        pub struct $name<'a>(pub &'a [u8], pub u16, pub u16);

        impl<'a> ImageBuffer for $name<'a> {
            fn get_jpeg_color_type(&self) -> JpegColorType {
                JpegColorType::Ycbcr
            }

            fn width(&self) -> u16 {
                self.1
            }

            fn height(&self) -> u16 {
                self.2
            }

            fn supports_precise_samples(&self) -> bool {
                true
            }

            fn fill_buffers_precise(
                &self,
                y: u16,
                buffers: &mut [Vec<u16>; 4],
                precision: SamplePrecision,
            ) -> bool {
                fill_precise_rgb::<$num_colors, $o1, $o2, $o3>(
                    self.0,
                    y,
                    self.width(),
                    buffers,
                    precision,
                )
            }

            #[inline(always)]
            fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                let line = get_line(self.0, y, self.width(), $num_colors);

                // Doing the convertion in chunks allows the compiler to vectorize the code
                // A size of 16 seems optimal for SSE and AVX capable hardware
                const CHUNK_SIZE: usize = 16;

                let mut y_buffer = [0; CHUNK_SIZE];
                let mut cb_buffer = [0; CHUNK_SIZE];
                let mut cr_buffer = [0; CHUNK_SIZE];

                for chuck in line.chunks_exact($num_colors * CHUNK_SIZE) {
                    for i in (0..CHUNK_SIZE) {
                        let (y, cb, cr) = rgb_to_ycbcr(
                            chuck[i * $num_colors + $o1],
                            chuck[i * $num_colors + $o2],
                            chuck[i * $num_colors + $o3],
                        );

                        y_buffer[i] = y;
                        cb_buffer[i] = cb;
                        cr_buffer[i] = cr;
                    }

                    buffers[0].extend_from_slice(&y_buffer);
                    buffers[1].extend_from_slice(&cb_buffer);
                    buffers[2].extend_from_slice(&cr_buffer);
                }

                // Add the remaining pixels in case the number of
                // pixels is not a multiple of CHUNK_SIZE
                let pixel = line.len() / $num_colors;
                for i in pixel / CHUNK_SIZE * CHUNK_SIZE..pixel {
                    let (y, cb, cr) = rgb_to_ycbcr(
                        line[i * $num_colors + $o1],
                        line[i * $num_colors + $o2],
                        line[i * $num_colors + $o3],
                    );

                    buffers[0].push(y);
                    buffers[1].push(cb);
                    buffers[2].push(cr);
                }
            }
        }
    };
}

ycbcr_image!(RgbImage, 3, 0, 1, 2);
ycbcr_image!(RgbaImage, 4, 0, 1, 2);
ycbcr_image!(BgrImage, 3, 2, 1, 0);
ycbcr_image!(BgraImage, 4, 2, 1, 0);

pub(crate) struct YCbCrImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for YCbCrImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Ycbcr
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let line = get_line(self.0, y, self.width(), 3);

        for pixel in line.as_chunks::<3>().0 {
            buffers[0].push(pixel[0]);
            buffers[1].push(pixel[1]);
            buffers[2].push(pixel[2]);
        }
    }
}

pub(crate) struct CmykImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for CmykImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Cmyk
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let line = get_line(self.0, y, self.width(), 4);

        for pixel in line.as_chunks::<4>().0 {
            buffers[0].push(255 - pixel[0]);
            buffers[1].push(255 - pixel[1]);
            buffers[2].push(255 - pixel[2]);
            buffers[3].push(255 - pixel[3]);
        }
    }
}

pub(crate) struct CmykAsYcckImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for CmykAsYcckImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Ycck
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let line = get_line(self.0, y, self.width(), 4);

        for pixel in line.as_chunks::<4>().0 {
            let (y, cb, cr, k) = cmyk_to_ycck(pixel[0], pixel[1], pixel[2], pixel[3]);

            buffers[0].push(y);
            buffers[1].push(cb);
            buffers[2].push(cr);
            buffers[3].push(k);
        }
    }
}

pub(crate) struct YcckImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for YcckImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Ycck
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let line = get_line(self.0, y, self.width(), 4);

        for pixel in line.as_chunks::<4>().0 {
            buffers[0].push(pixel[0]);
            buffers[1].push(pixel[1]);
            buffers[2].push(pixel[2]);
            buffers[3].push(pixel[3]);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::rgb_to_ycbcr;

    #[cfg(any(
        all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    pub(crate) fn check_simd_rows(mut check: impl FnMut(&[u8], &[u8], u16, u16)) {
        for width in (0..=65).chain([127, 128, 129, 515, u16::MAX]) {
            let height = 2;
            let mut state = 0x9e3779b9u32;
            // A leading byte exercises unaligned input. Include endpoint and
            // half-step colors as well as independently varying random channels.
            let colors = (0..usize::from(width) * usize::from(height)).map(|i| {
                let edges = [
                    [0, 0, 0],
                    [255, 255, 255],
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 255, 0],
                    [0, 255, 255],
                    [255, 0, 255],
                    [59, 109, 6],
                    [0, 1, 0],
                    [255, 254, 255],
                ];
                if i % 23 < edges.len() {
                    edges[i % 23]
                } else {
                    core::array::from_fn(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        (state >> 24) as u8
                    })
                }
            });
            let mut rgb = alloc::vec![73];
            let mut rgba = alloc::vec![91];
            for [r, g, b] in colors {
                rgb.extend_from_slice(&[r, g, b]);
                rgba.extend_from_slice(&[r, g, b, (r ^ b).wrapping_add(g)]);
            }
            check(&rgb[1..], &rgba[1..], width, height);
        }
    }

    #[cfg(any(
        all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    pub(crate) fn check_simd_image(simd: &dyn ImageBuffer, scalar: &dyn ImageBuffer) {
        // Independent initial lengths and no spare capacity exercise reserve,
        // appending to existing data, and preservation of the fourth channel.
        let mut bytes = core::array::from_fn(|c| alloc::vec![c as u8; c + 1]);
        let mut expected = bytes.clone();
        for row in [1, 0] {
            simd.fill_buffers(row, &mut bytes);
            scalar.fill_buffers(row, &mut expected);
            assert_eq!(bytes, expected, "byte samples, width {}", simd.width());
        }
        for precision in [
            SamplePrecision::Integer,
            SamplePrecision::Luma,
            SamplePrecision::All,
        ] {
            let mut actual = core::array::from_fn(|c| alloc::vec![60000 + c as u16; c + 1]);
            let mut expected = actual.clone();
            for row in [1, 0] {
                assert_eq!(
                    simd.fill_buffers_precise(row, &mut actual, precision),
                    scalar.fill_buffers_precise(row, &mut expected, precision),
                );
                assert_eq!(actual, expected, "{precision:?}, width {}", simd.width());
            }
        }
    }

    #[test]
    fn precise_rgb_rows_preserve_channel_order_tails_and_append() {
        for width in [1u16, 7, 16, 17, 31, 32, 33] {
            let colors: Vec<[u8; 3]> = (0..usize::from(width) * 3)
                .map(|i| [(i * 37) as u8, (i * 83 + 17) as u8, (i * 13 + 251) as u8])
                .collect();
            let rgb: Vec<u8> = colors.iter().flatten().copied().collect();
            let rgba: Vec<u8> = colors.iter().flat_map(|c| [c[0], c[1], c[2], 99]).collect();
            let bgr: Vec<u8> = colors.iter().flat_map(|c| [c[2], c[1], c[0]]).collect();
            let bgra: Vec<u8> = colors.iter().flat_map(|c| [c[2], c[1], c[0], 11]).collect();
            let images: [&dyn ImageBuffer; 4] = [
                &RgbImage(&rgb, width, 3),
                &RgbaImage(&rgba, width, 3),
                &BgrImage(&bgr, width, 3),
                &BgraImage(&bgra, width, 3),
            ];
            for image in images {
                assert!(image.supports_precise_samples());
                for precision in [SamplePrecision::Luma, SamplePrecision::All] {
                    let mut out: [Vec<u16>; 4] = core::array::from_fn(|c| alloc::vec![c as u16]);
                    let mut expected = out.clone();
                    for row in [2u16, 0] {
                        assert!(image.fill_buffers_precise(row, &mut out, precision));
                        for &[r, g, b] in
                            &colors[usize::from(row * width)..usize::from((row + 1) * width)]
                        {
                            // Independent i64 matrix reference, including the channel offset.
                            let input = [i64::from(r), i64::from(g), i64::from(b), 1];
                            let matrix = [
                                [19595, 38470, 7471, 0],
                                [-11059, -21709, 32768, 128 << 16],
                                [32768, -27439, -5329, 128 << 16],
                            ];
                            for c in 0..3 {
                                let raw: i64 =
                                    input.iter().zip(matrix[c]).map(|(x, k)| x * k).sum();
                                let value = if c == 0 || matches!(precision, SamplePrecision::All) {
                                    ((raw + 127) >> 8) as u16
                                } else {
                                    (((raw + 0x7fff) >> 16) as u16) << 8
                                };
                                expected[c].push(value);
                            }
                        }
                    }
                    assert_eq!(out, expected);
                }
                let mut untouched = core::array::from_fn(|c| alloc::vec![c as u16, 60000]);
                let before = untouched.clone();
                assert!(!image.fill_buffers_precise(0, &mut untouched, SamplePrecision::Integer));
                assert_eq!(untouched, before);
            }
        }
    }

    #[test]
    fn precise_constant_colors_keep_fractional_values_and_endpoints() {
        for color in [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [59, 109, 6],
        ] {
            let pixels = color.repeat(19);
            let image = RgbImage(&pixels, 19, 1);
            let mut out: [Vec<u16>; 4] = Default::default();
            assert!(image.fill_buffers_precise(0, &mut out, SamplePrecision::All));
            let rgb: [f64; 3] = color.map(f64::from);
            let reference = [
                0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2],
                -0.16874 * rgb[0] - 0.33126 * rgb[1] + 0.5 * rgb[2] + 128.0,
                0.5 * rgb[0] - 0.41869 * rgb[1] - 0.08131 * rgb[2] + 128.0,
            ];
            for c in 0..3 {
                assert!(out[c].iter().all(|&v| v == out[c][0]));
                assert!((f64::from(out[c][0]) / 256.0 - reference[c]).abs() < 1.0 / 128.0);
            }
            let mut luma: [Vec<u16>; 4] = Default::default();
            assert!(image.fill_buffers_precise(0, &mut luma, SamplePrecision::Luma));
            let (_, cb, cr) = rgb_to_ycbcr(color[0], color[1], color[2]);
            assert_eq!(luma[0], out[0]);
            assert!(luma[1].iter().all(|&v| v == u16::from(cb) << 8));
            assert!(luma[2].iter().all(|&v| v == u16::from(cr) << 8));
            if color == [255, 0, 0] {
                assert_eq!(out[2][0], 65408);
            }
        }
        let mut out = core::array::from_fn(|c| alloc::vec![c as u16]);
        let before = out.clone();
        let gray = GrayImage(&[7], 1, 1);
        assert!(!gray.supports_precise_samples());
        assert!(!gray.fill_buffers_precise(0, &mut out, SamplePrecision::All));
        assert_eq!(out, before);
    }

    fn assert_rgb_to_ycbcr(rgb: [u8; 3], ycbcr: [u8; 3]) {
        let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
        assert_eq!([y, cb, cr], ycbcr);
    }

    #[test]
    fn test_rgb_to_ycbcr() {
        assert_rgb_to_ycbcr([0, 0, 0], [0, 128, 128]);
        assert_rgb_to_ycbcr([255, 255, 255], [255, 128, 128]);
        assert_rgb_to_ycbcr([255, 0, 0], [76, 85, 255]);
        assert_rgb_to_ycbcr([0, 255, 0], [150, 44, 21]);
        assert_rgb_to_ycbcr([0, 0, 255], [29, 255, 107]);

        // Values taken from libjpeg for a common image

        assert_rgb_to_ycbcr([59, 109, 6], [82, 85, 111]);
        assert_rgb_to_ycbcr([29, 60, 11], [45, 109, 116]);
        assert_rgb_to_ycbcr([57, 114, 26], [87, 94, 107]);
        assert_rgb_to_ycbcr([30, 60, 6], [45, 106, 117]);
        assert_rgb_to_ycbcr([41, 75, 11], [58, 102, 116]);
        assert_rgb_to_ycbcr([145, 184, 108], [164, 97, 115]);
        assert_rgb_to_ycbcr([33, 85, 7], [61, 98, 108]);
        assert_rgb_to_ycbcr([61, 90, 40], [76, 108, 118]);
        assert_rgb_to_ycbcr([75, 127, 45], [102, 96, 109]);
        assert_rgb_to_ycbcr([30, 56, 14], [43, 111, 118]);
        assert_rgb_to_ycbcr([106, 142, 81], [124, 104, 115]);
        assert_rgb_to_ycbcr([35, 59, 11], [46, 108, 120]);
        assert_rgb_to_ycbcr([170, 203, 123], [184, 94, 118]);
        assert_rgb_to_ycbcr([45, 87, 16], [66, 100, 113]);
        assert_rgb_to_ycbcr([59, 109, 21], [84, 92, 110]);
        assert_rgb_to_ycbcr([100, 167, 36], [132, 74, 105]);
        assert_rgb_to_ycbcr([17, 53, 5], [37, 110, 114]);
        assert_rgb_to_ycbcr([226, 244, 220], [236, 119, 121]);
        assert_rgb_to_ycbcr([192, 214, 120], [197, 85, 125]);
        assert_rgb_to_ycbcr([63, 107, 22], [84, 93, 113]);
        assert_rgb_to_ycbcr([44, 78, 19], [61, 104, 116]);
        assert_rgb_to_ycbcr([72, 106, 54], [90, 108, 115]);
        assert_rgb_to_ycbcr([99, 123, 73], [110, 107, 120]);
        assert_rgb_to_ycbcr([188, 216, 148], [200, 99, 120]);
        assert_rgb_to_ycbcr([19, 46, 7], [33, 113, 118]);
        assert_rgb_to_ycbcr([56, 95, 40], [77, 107, 113]);
        assert_rgb_to_ycbcr([81, 120, 56], [101, 103, 114]);
        assert_rgb_to_ycbcr([9, 30, 0], [20, 117, 120]);
        assert_rgb_to_ycbcr([90, 118, 46], [101, 97, 120]);
        assert_rgb_to_ycbcr([24, 52, 0], [38, 107, 118]);
        assert_rgb_to_ycbcr([32, 69, 9], [51, 104, 114]);
        assert_rgb_to_ycbcr([74, 134, 33], [105, 88, 106]);
        assert_rgb_to_ycbcr([37, 74, 7], [55, 101, 115]);
        assert_rgb_to_ycbcr([69, 119, 31], [94, 92, 110]);
        assert_rgb_to_ycbcr([63, 112, 21], [87, 91, 111]);
        assert_rgb_to_ycbcr([90, 148, 17], [116, 72, 110]);
        assert_rgb_to_ycbcr([50, 97, 30], [75, 102, 110]);
        assert_rgb_to_ycbcr([99, 129, 72], [114, 105, 118]);
        assert_rgb_to_ycbcr([161, 196, 57], [170, 64, 122]);
        assert_rgb_to_ycbcr([10, 26, 1], [18, 118, 122]);
        assert_rgb_to_ycbcr([87, 128, 68], [109, 105, 112]);
        assert_rgb_to_ycbcr([111, 155, 73], [132, 94, 113]);
        assert_rgb_to_ycbcr([33, 75, 11], [55, 103, 112]);
        assert_rgb_to_ycbcr([70, 122, 51], [98, 101, 108]);
        assert_rgb_to_ycbcr([22, 74, 3], [50, 101, 108]);
        assert_rgb_to_ycbcr([88, 142, 45], [115, 89, 109]);
        assert_rgb_to_ycbcr([66, 107, 40], [87, 101, 113]);
        assert_rgb_to_ycbcr([18, 45, 0], [32, 110, 118]);
        assert_rgb_to_ycbcr([163, 186, 88], [168, 83, 124]);
        assert_rgb_to_ycbcr([47, 104, 4], [76, 88, 108]);
        assert_rgb_to_ycbcr([147, 211, 114], [181, 90, 104]);
        assert_rgb_to_ycbcr([42, 77, 18], [60, 104, 115]);
        assert_rgb_to_ycbcr([37, 72, 6], [54, 101, 116]);
        assert_rgb_to_ycbcr([84, 140, 55], [114, 95, 107]);
        assert_rgb_to_ycbcr([46, 98, 25], [74, 100, 108]);
        assert_rgb_to_ycbcr([48, 97, 20], [74, 98, 110]);
        assert_rgb_to_ycbcr([189, 224, 156], [206, 100, 116]);
        assert_rgb_to_ycbcr([36, 83, 0], [59, 94, 111]);
        assert_rgb_to_ycbcr([159, 186, 114], [170, 97, 120]);
        assert_rgb_to_ycbcr([75, 118, 46], [97, 99, 112]);
        assert_rgb_to_ycbcr([193, 233, 158], [212, 97, 114]);
        assert_rgb_to_ycbcr([76, 116, 48], [96, 101, 114]);
        assert_rgb_to_ycbcr([108, 157, 79], [133, 97, 110]);
        assert_rgb_to_ycbcr([180, 208, 155], [194, 106, 118]);
        assert_rgb_to_ycbcr([74, 126, 53], [102, 100, 108]);
        assert_rgb_to_ycbcr([72, 123, 46], [99, 98, 109]);
        assert_rgb_to_ycbcr([71, 123, 34], [97, 92, 109]);
        assert_rgb_to_ycbcr([130, 184, 72], [155, 81, 110]);
        assert_rgb_to_ycbcr([30, 61, 17], [47, 111, 116]);
        assert_rgb_to_ycbcr([27, 71, 0], [50, 100, 112]);
        assert_rgb_to_ycbcr([45, 73, 24], [59, 108, 118]);
        assert_rgb_to_ycbcr([139, 175, 93], [155, 93, 117]);
        assert_rgb_to_ycbcr([11, 38, 0], [26, 114, 118]);
        assert_rgb_to_ycbcr([34, 87, 15], [63, 101, 107]);
        assert_rgb_to_ycbcr([43, 76, 35], [61, 113, 115]);
        assert_rgb_to_ycbcr([18, 35, 7], [27, 117, 122]);
        assert_rgb_to_ycbcr([69, 97, 48], [83, 108, 118]);
        assert_rgb_to_ycbcr([139, 176, 50], [151, 71, 120]);
        assert_rgb_to_ycbcr([21, 51, 7], [37, 111, 117]);
        assert_rgb_to_ycbcr([209, 249, 189], [230, 105, 113]);
        assert_rgb_to_ycbcr([32, 66, 14], [50, 108, 115]);
        assert_rgb_to_ycbcr([100, 143, 67], [121, 97, 113]);
        assert_rgb_to_ycbcr([40, 96, 14], [70, 96, 107]);
        assert_rgb_to_ycbcr([88, 130, 64], [110, 102, 112]);
        assert_rgb_to_ycbcr([52, 112, 14], [83, 89, 106]);
        assert_rgb_to_ycbcr([49, 72, 25], [60, 108, 120]);
        assert_rgb_to_ycbcr([144, 193, 75], [165, 77, 113]);
        assert_rgb_to_ycbcr([49, 94, 1], [70, 89, 113]);
    }
}
