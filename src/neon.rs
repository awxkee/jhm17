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
//! Bit-exact RGB-family conversion on AArch64.

use alloc::vec::Vec;
use core::arch::aarch64::*;

use crate::image_buffer::{get_line, rgb_to_ycbcr_q8};
use crate::{ImageBuffer, JpegColorType, SamplePrecision, rgb_to_ycbcr};

#[inline]
#[target_feature(enable = "neon")]
fn load_rgb<const N: usize, const R: usize, const G: usize, const B: usize>(
    data: &[u8],
) -> [int16x8_t; 3] {
    assert!(data.len() >= if N == 3 { 24 } else { 32 });
    // SAFETY: the slice contains all bytes consumed by the packed load.
    let channels = unsafe {
        if N == 3 {
            let rgb = vld3_u8(data.as_ptr());
            [rgb.0, rgb.1, rgb.2, vdup_n_u8(0)]
        } else {
            let rgba = vld4_u8(data.as_ptr());
            [rgba.0, rgba.1, rgba.2, rgba.3]
        }
    };
    [R, G, B].map(|c| vreinterpretq_s16_u16(vmovl_u8(channels[c])))
}

#[inline]
#[target_feature(enable = "neon")]
fn convert4<const MODE: u8>(r: int16x4_t, g: int16x4_t, b: int16x4_t) -> [uint16x4_t; 3] {
    // The matrix rows sum to 65536, 0, 0 respectively. Factoring out
    // green reduces nine multiplies to four without changing any integer bits.
    let rg = vsub_s16(r, g);
    let bg = vsub_s16(b, g);
    let y = vmlal_n_s16(
        vmlal_n_s16(vshlq_n_s32::<16>(vmovl_s16(g)), rg, 19595),
        bg,
        7471,
    );
    let offset = vdupq_n_s32(128 << 16);
    let cb = vmlsl_n_s16(vaddq_s32(offset, vshll_n_s16::<15>(bg)), rg, 11059);
    let cr = vmlsl_n_s16(vaddq_s32(offset, vshll_n_s16::<15>(rg)), bg, 5329);
    let narrow = |value: int32x4_t, fractional: bool| {
        let rounding = vdupq_n_s32(if fractional { 127 } else { 0x7fff });
        let value = vreinterpretq_u32_s32(vaddq_s32(value, rounding));
        if fractional {
            vshrn_n_u32::<8>(value)
        } else if MODE == 0 {
            vshrn_n_u32::<16>(value)
        } else {
            vshl_n_u16::<8>(vshrn_n_u32::<16>(value))
        }
    };
    [
        narrow(y, MODE != 0),
        narrow(cb, MODE == 2),
        narrow(cr, MODE == 2),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn convert8<const MODE: u8>([r, g, b]: [int16x8_t; 3]) -> [uint16x8_t; 3] {
    let lo = convert4::<MODE>(vget_low_s16(r), vget_low_s16(g), vget_low_s16(b));
    let hi = convert4::<MODE>(vget_high_s16(r), vget_high_s16(g), vget_high_s16(b));
    core::array::from_fn(|c| vcombine_u16(lo[c], hi[c]))
}

#[target_feature(enable = "neon")]
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
    // SAFETY: chunks_exact supplies all eight pixels, and reserve guarantees
    // width writable samples after each independent channel's current length.
    // No input load extends past this row, including its final SIMD chunk.
    unsafe {
        let pointers = core::array::from_fn::<_, 3, _>(|c| out[c].as_mut_ptr().add(starts[c]));
        for (i, chunk) in (&mut chunks).enumerate() {
            let converted = convert8::<0>(load_rgb::<N, R, G, B>(chunk));
            for c in 0..3 {
                vst1_u8(pointers[c].add(i * 8), vmovn_u16(converted[c]));
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

#[target_feature(enable = "neon")]
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
    // SAFETY: the same row-load and reserved-capacity invariants as fill_bytes;
    // every stored u16 is initialized before the vector lengths are advanced.
    unsafe {
        let pointers = core::array::from_fn::<_, 3, _>(|c| out[c].as_mut_ptr().add(starts[c]));
        for (i, chunk) in (&mut chunks).enumerate() {
            let converted = convert8::<MODE>(load_rgb::<N, R, G, B>(chunk));
            for c in 0..3 {
                vst1q_u16(pointers[c].add(i * 8), converted[c]);
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

macro_rules! ycbcr_image_neon {
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
                // SAFETY: NEON is part of the AArch64 baseline target.
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
                // SAFETY: NEON is part of the AArch64 baseline target.
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

ycbcr_image_neon!(RgbImageNEON, 3, 0, 1, 2);
ycbcr_image_neon!(RgbaImageNEON, 4, 0, 1, 2);
ycbcr_image_neon!(BgrImageNEON, 3, 2, 1, 0);
ycbcr_image_neon!(BgraImageNEON, 4, 2, 1, 0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_buffer::*;

    #[test]
    fn rows_match_scalar() {
        crate::image_buffer::tests::check_simd_rows(|rgb, rgba, width, height| {
            let pairs: [(&dyn ImageBuffer, &dyn ImageBuffer); 4] = [
                (
                    &RgbImageNEON(rgb, width, height),
                    &RgbImage(rgb, width, height),
                ),
                (
                    &RgbaImageNEON(rgba, width, height),
                    &RgbaImage(rgba, width, height),
                ),
                (
                    &BgrImageNEON(rgb, width, height),
                    &BgrImage(rgb, width, height),
                ),
                (
                    &BgraImageNEON(rgba, width, height),
                    &BgraImage(rgba, width, height),
                ),
            ];
            for (simd, scalar) in pairs {
                crate::image_buffer::tests::check_simd_image(simd, scalar);
            }
        });
    }
}
