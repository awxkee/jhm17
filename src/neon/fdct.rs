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
 * //
 * // Portions of this file are derived from the Independent JPEG Group's software,
 * // see LICENSE-IJG.
 */
//! Four parallel LL&M transforms, retaining i32 Q2 coefficients.

use core::arch::aarch64::*;
use core::ops::{Add, Mul, Shl, Shr, Sub};

use crate::encoder::AlignedBlock;
use crate::fdct::integer_dct_kernel;

#[derive(Clone, Copy)]
struct I32x4(int32x4_t);

impl Add for I32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(vaddq_s32(self.0, rhs.0)) }
    }
}
impl Sub for I32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(vsubq_s32(self.0, rhs.0)) }
    }
}
impl Mul<i32> for I32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i32) -> Self {
        unsafe { Self(vmulq_n_s32(self.0, rhs)) }
    }
}
impl Add<i32> for I32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        unsafe { Self(vaddq_s32(self.0, vdupq_n_s32(rhs))) }
    }
}
impl Shl<i32> for I32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        unsafe { Self(vshlq_s32(self.0, vdupq_n_s32(rhs))) }
    }
}
impl Shr<i32> for I32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        unsafe { Self(vshlq_s32(self.0, vdupq_n_s32(-rhs))) }
    }
}

integer_dct_kernel!(kernel, I32x4);

#[inline(always)]
fn transpose([a, b, c, d]: [I32x4; 4]) -> [I32x4; 4] {
    // SAFETY: all operations are register-only baseline NEON instructions.
    unsafe {
        let ab0 = vreinterpretq_s64_s32(vtrn1q_s32(a.0, b.0));
        let ab1 = vreinterpretq_s64_s32(vtrn2q_s32(a.0, b.0));
        let cd0 = vreinterpretq_s64_s32(vtrn1q_s32(c.0, d.0));
        let cd1 = vreinterpretq_s64_s32(vtrn2q_s32(c.0, d.0));
        [
            I32x4(vreinterpretq_s32_s64(vzip1q_s64(ab0, cd0))),
            I32x4(vreinterpretq_s32_s64(vzip1q_s64(ab1, cd1))),
            I32x4(vreinterpretq_s32_s64(vzip2q_s64(ab0, cd0))),
            I32x4(vreinterpretq_s32_s64(vzip2q_s64(ab1, cd1))),
        ]
    }
}

// The caller checks the ten-bit input bound before selecting this kernel.
pub(crate) fn fdct_precise(data: &AlignedBlock) -> [i32; 64] {
    let rows: [[I32x4; 8]; 2] = core::array::from_fn(|group| {
        // SAFETY: each load is exactly one of the eight complete block rows.
        let input: [int16x8_t; 4] = core::array::from_fn(|y| unsafe {
            vld1q_s16(data.data.as_ptr().add((group * 4 + y) * 8))
        });
        let lo = transpose(input.map(|row| unsafe { I32x4(vmovl_s16(vget_low_s16(row))) }));
        let hi = transpose(input.map(|row| unsafe { I32x4(vmovl_high_s16(row)) }));
        kernel::<-2>([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    });
    let mut out = [0; 64];
    for x in [0, 4] {
        let top = transpose(core::array::from_fn(|i| rows[0][x + i]));
        let bottom = transpose(core::array::from_fn(|i| rows[1][x + i]));
        let columns = kernel::<2>([
            top[0], top[1], top[2], top[3], bottom[0], bottom[1], bottom[2], bottom[3],
        ]);
        for (y, row) in columns.into_iter().enumerate() {
            // SAFETY: x is 0 or 4, so all four output lanes are within row y.
            unsafe {
                vst1q_s32(out.as_mut_ptr().add(y * 8 + x), row.0);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precise_simd_matches_wide_scalar() {
        // The wide scalar oracle also detects overflow in SIMD intermediates.
        let check = |input: [i16; 64]| {
            let block = AlignedBlock::new(input);
            assert_eq!(
                fdct_precise(&block),
                crate::fdct::scalar_dct_i64(&block).map(|v| v as i32)
            );
        };
        for value in [-512, -511, -1, 0, 1, 510, 511, 512] {
            check([value; 64]);
        }
        for position in 0..64 {
            for value in [-512, 512] {
                let mut impulse = [0; 64];
                impulse[position] = value;
                check(impulse);
            }
        }
        for pattern in 0..8 {
            check(core::array::from_fn(|i| {
                if (i >> pattern) & 1 == 0 { -512 } else { 512 }
            }));
        }
        let mut seed = 0x31415926u32;
        for _ in 0..4096 {
            check(core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) % 1025) as i16 - 512
            }));
        }
    }
}
