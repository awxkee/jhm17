#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use alloc::vec::Vec;

use crate::image_buffer::{get_line, rgb_to_ycbcr_q8};
use crate::{ImageBuffer, JpegColorType, SamplePrecision, rgb_to_ycbcr};

#[inline]
#[target_feature(enable = "avx2")]
fn load_rgb<const N: usize, const R: usize, const G: usize, const B: usize>(
    data: *const u8,
) -> [__m256i; 3] {
    // SAFETY: eight complete pixels are available. The overlapping RGB loads
    // cover exactly bytes 0..24; RGBA uses exactly bytes 0..32.
    let packed = unsafe {
        if N == 3 {
            _mm256_set_m128i(
                _mm_loadu_si128(data.add(8).cast()),
                _mm_loadu_si128(data.cast()),
            )
        } else {
            _mm256_loadu_si256(data.cast())
        }
    };
    [R, G, B].map(|c| {
        let c = c as i8;
        let n = N as i8;
        let hi = if N == 3 { 4 } else { 0 } + c;
        let mask = _mm256_setr_epi8(
            c,
            c + n,
            c + 2 * n,
            c + 3 * n,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            hi,
            hi + n,
            hi + 2 * n,
            hi + 3 * n,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        );
        let shuffled = _mm256_shuffle_epi8(packed, mask);
        let bytes = _mm_unpacklo_epi32(
            _mm256_castsi256_si128(shuffled),
            _mm256_extracti128_si256::<1>(shuffled),
        );
        _mm256_cvtepu8_epi32(bytes)
    })
}

#[inline]
#[target_feature(enable = "avx2")]
fn convert8<const MODE: u8>([r, g, b]: [__m256i; 3]) -> [__m128i; 3] {
    // Exact green-factored matrix: four multiplies instead of nine.
    let rg = _mm256_sub_epi32(r, g);
    let bg = _mm256_sub_epi32(b, g);
    let y = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_slli_epi32::<16>(g),
            _mm256_mullo_epi32(rg, _mm256_set1_epi32(19595)),
        ),
        _mm256_mullo_epi32(bg, _mm256_set1_epi32(7471)),
    );
    let offset = _mm256_set1_epi32(128 << 16);
    let cb = _mm256_sub_epi32(
        _mm256_add_epi32(offset, _mm256_slli_epi32::<15>(bg)),
        _mm256_mullo_epi32(rg, _mm256_set1_epi32(11059)),
    );
    let cr = _mm256_sub_epi32(
        _mm256_add_epi32(offset, _mm256_slli_epi32::<15>(rg)),
        _mm256_mullo_epi32(bg, _mm256_set1_epi32(5329)),
    );
    let narrow = |value: __m256i, fractional: bool| {
        let value = _mm256_add_epi32(
            value,
            _mm256_set1_epi32(if fractional { 127 } else { 0x7fff }),
        );
        let value = if fractional {
            _mm256_srli_epi32::<8>(value)
        } else if MODE == 0 {
            _mm256_srli_epi32::<16>(value)
        } else {
            _mm256_slli_epi32::<8>(_mm256_srli_epi32::<16>(value))
        };
        _mm_packus_epi32(
            _mm256_castsi256_si128(value),
            _mm256_extracti128_si256::<1>(value),
        )
    };
    [
        narrow(y, MODE != 0),
        narrow(cb, MODE == 2),
        narrow(cr, MODE == 2),
    ]
}

#[target_feature(enable = "avx2")]
fn fill_bytes<const N: usize, const R: usize, const G: usize, const B: usize>(
    line: &[u8],
    buffers: &mut [Vec<u8>; 4],
) {
    let width = line.len() / N;
    let [ys, cbs, crs, _] = buffers;
    let channels = [ys, cbs, crs];
    let starts = channels.each_ref().map(|c| c.len());
    let mut out = channels.map(|c| {
        c.reserve(width);
        c
    });
    let mut chunks = line.chunks_exact(N * 8);
    // SAFETY: chunks_exact bounds all input loads to this row. reserve makes
    // width samples writable after each channel's independent current length.
    // The lengths are advanced only after all SIMD outputs are initialized.
    unsafe {
        let pointers = core::array::from_fn::<_, 3, _>(|c| out[c].as_mut_ptr().add(starts[c]));
        for (i, chunk) in (&mut chunks).enumerate() {
            let converted = convert8::<0>(load_rgb::<N, R, G, B>(chunk.as_ptr()));
            for c in 0..3 {
                _mm_storel_epi64(
                    pointers[c].add(i * 8).cast(),
                    _mm_packus_epi16(converted[c], _mm_setzero_si128()),
                );
            }
        }
        let written = width / 8 * 8;
        for (c, channel) in out.into_iter().enumerate() {
            channel.set_len(starts[c] + written);
        }
    }
    for pixel in chunks.remainder().as_chunks::<N>().0 {
        let (y, cb, cr) = rgb_to_ycbcr(pixel[R], pixel[G], pixel[B]);
        buffers[0].push(y);
        buffers[1].push(cb);
        buffers[2].push(cr);
    }
}

#[target_feature(enable = "avx2")]
fn fill_precise<const N: usize, const R: usize, const G: usize, const B: usize, const MODE: u8>(
    line: &[u8],
    buffers: &mut [Vec<u16>; 4],
) {
    let width = line.len() / N;
    let [ys, cbs, crs, _] = buffers;
    let channels = [ys, cbs, crs];
    let starts = channels.each_ref().map(|c| c.len());
    let mut out = channels.map(|c| {
        c.reserve(width);
        c
    });
    let mut chunks = line.chunks_exact(N * 8);
    // SAFETY: the same row-load and reserved-capacity invariants as fill_bytes.
    unsafe {
        let pointers = core::array::from_fn::<_, 3, _>(|c| out[c].as_mut_ptr().add(starts[c]));
        for (i, chunk) in (&mut chunks).enumerate() {
            let converted = convert8::<MODE>(load_rgb::<N, R, G, B>(chunk.as_ptr()));
            for c in 0..3 {
                _mm_storeu_si128(pointers[c].add(i * 8).cast(), converted[c]);
            }
        }
        let written = width / 8 * 8;
        for (c, channel) in out.into_iter().enumerate() {
            channel.set_len(starts[c] + written);
        }
    }
    for pixel in chunks.remainder().as_chunks::<N>().0 {
        let (y, cb, cr) = rgb_to_ycbcr_q8(pixel[R], pixel[G], pixel[B], MODE == 2);
        buffers[0].push(y);
        buffers[1].push(cb);
        buffers[2].push(cr);
    }
}

macro_rules! ycbcr_image_avx2 {
    ($name:ident, $n:expr, $r:expr, $g:expr, $b:expr) => {
        pub struct $name<'a>(pub &'a [u8], pub u16, pub u16);

        impl ImageBuffer for $name<'_> {
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

            fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                let line = get_line(self.0, y, self.1, $n);
                // SAFETY: Encoder dispatches here only after detecting AVX2.
                unsafe { fill_bytes::<$n, $r, $g, $b>(line, buffers) }
            }

            fn fill_buffers_precise(
                &self,
                y: u16,
                buffers: &mut [Vec<u16>; 4],
                precision: SamplePrecision,
            ) -> bool {
                if precision == SamplePrecision::Integer {
                    return false;
                }
                let line = get_line(self.0, y, self.1, $n);
                // SAFETY: Encoder dispatches here only after detecting AVX2.
                unsafe {
                    match precision {
                        SamplePrecision::All => fill_precise::<$n, $r, $g, $b, 2>(line, buffers),
                        SamplePrecision::Luma => fill_precise::<$n, $r, $g, $b, 1>(line, buffers),
                        SamplePrecision::Integer => unreachable!(),
                    }
                }
                true
            }
        }
    };
}

ycbcr_image_avx2!(RgbImageAVX2, 3, 0, 1, 2);
ycbcr_image_avx2!(RgbaImageAVX2, 4, 0, 1, 2);
ycbcr_image_avx2!(BgrImageAVX2, 3, 2, 1, 0);
ycbcr_image_avx2!(BgraImageAVX2, 4, 2, 1, 0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_buffer::*;

    #[test]
    fn rows_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            std::eprintln!("AVX2 unavailable; skipping runtime conversion checks");
            return;
        }
        crate::image_buffer::tests::check_simd_rows(|rgb, rgba, width, height| {
            let pairs: [(&dyn ImageBuffer, &dyn ImageBuffer); 4] = [
                (
                    &RgbImageAVX2(rgb, width, height),
                    &RgbImage(rgb, width, height),
                ),
                (
                    &RgbaImageAVX2(rgba, width, height),
                    &RgbaImage(rgba, width, height),
                ),
                (
                    &BgrImageAVX2(rgb, width, height),
                    &BgrImage(rgb, width, height),
                ),
                (
                    &BgraImageAVX2(rgba, width, height),
                    &BgraImage(rgba, width, height),
                ),
            ];
            for (simd, scalar) in pairs {
                crate::image_buffer::tests::check_simd_image(simd, scalar);
            }
        });
    }
}
