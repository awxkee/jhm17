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
//! Eight independent butterfly transforms in AVX2 lanes.

use super::{Lane, kernel};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::ops::{Add, Mul, Sub};

// These private lanes are used only inside the AVX2 transform selected below.
// Keep multiply and add separate to retain the scalar transform's rounding.
#[derive(Clone, Copy)]
struct F32x8(__m256);
impl Lane for F32x8 {}

impl Add for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_add_ps(self.0, rhs.0)) }
    }
}
impl Sub for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) }
    }
}
impl Mul<f32> for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        unsafe { Self(_mm256_mul_ps(self.0, _mm256_set1_ps(rhs))) }
    }
}

#[inline(always)]
fn transpose(rows: [F32x8; 8]) -> [F32x8; 8] {
    // Each unpack/shuffle transposes inside the 128-bit halves; the final
    // permutes join the upper and lower four rows into complete columns.
    unsafe {
        let [a, b, c, d, e, f, g, h] = rows.map(|row| row.0);
        let ab0 = _mm256_unpacklo_ps(a, b);
        let ab1 = _mm256_unpackhi_ps(a, b);
        let cd0 = _mm256_unpacklo_ps(c, d);
        let cd1 = _mm256_unpackhi_ps(c, d);
        let ef0 = _mm256_unpacklo_ps(e, f);
        let ef1 = _mm256_unpackhi_ps(e, f);
        let gh0 = _mm256_unpacklo_ps(g, h);
        let gh1 = _mm256_unpackhi_ps(g, h);
        let top0 = _mm256_shuffle_ps::<0x44>(ab0, cd0);
        let top1 = _mm256_shuffle_ps::<0xee>(ab0, cd0);
        let top2 = _mm256_shuffle_ps::<0x44>(ab1, cd1);
        let top3 = _mm256_shuffle_ps::<0xee>(ab1, cd1);
        let bottom0 = _mm256_shuffle_ps::<0x44>(ef0, gh0);
        let bottom1 = _mm256_shuffle_ps::<0xee>(ef0, gh0);
        let bottom2 = _mm256_shuffle_ps::<0x44>(ef1, gh1);
        let bottom3 = _mm256_shuffle_ps::<0xee>(ef1, gh1);
        [
            F32x8(_mm256_permute2f128_ps::<0x20>(top0, bottom0)),
            F32x8(_mm256_permute2f128_ps::<0x20>(top1, bottom1)),
            F32x8(_mm256_permute2f128_ps::<0x20>(top2, bottom2)),
            F32x8(_mm256_permute2f128_ps::<0x20>(top3, bottom3)),
            F32x8(_mm256_permute2f128_ps::<0x31>(top0, bottom0)),
            F32x8(_mm256_permute2f128_ps::<0x31>(top1, bottom1)),
            F32x8(_mm256_permute2f128_ps::<0x31>(top2, bottom2)),
            F32x8(_mm256_permute2f128_ps::<0x31>(top3, bottom3)),
        ]
    }
}

/// # Safety
/// The caller must establish AVX2 support before entering this function.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn transform<const INVERSE: bool>(input: &[f32; 64], out: &mut [f32; 64]) {
    let rows = core::array::from_fn(|y| {
        // SAFETY: each load reads exactly one complete eight-float row.
        F32x8(unsafe { _mm256_loadu_ps(input.as_ptr().add(y * 8)) })
    });
    let horizontal = kernel::<_, INVERSE>(transpose(rows));
    let vertical = kernel::<_, INVERSE>(transpose(horizontal));
    for (row, value) in out.as_chunks_mut::<8>().0.iter_mut().zip(vertical) {
        // SAFETY: each store writes exactly one complete eight-float row;
        // loadu/storeu allow callers' arrays to have ordinary f32 alignment.
        unsafe { _mm256_storeu_ps(row.as_mut_ptr(), (value * 0.125).0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(input: &[f32; 64]) {
        let mut expected = [0.0; 64];
        let mut actual = [0.0; 64];
        for inverse in [false, true] {
            // SAFETY: the test establishes AVX2 support before calling check.
            unsafe {
                if inverse {
                    crate::dct_f32::scalar::<true>(input, &mut expected);
                    transform::<true>(input, &mut actual);
                } else {
                    crate::dct_f32::scalar::<false>(input, &mut expected);
                    transform::<false>(input, &mut actual);
                }
            }
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }

    #[test]
    fn avx2_matches_scalar_for_both_transforms() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for value in [-1e30, -8192.0, -0.0, 0.0, 8191.75, 1e30] {
            check(&[value; 64]);
        }
        for position in 0..64 {
            let mut input = [0.0; 64];
            input[position] = if position & 1 == 0 { -8192.0 } else { 8191.75 };
            check(&input);
        }
        for bit in 0..6 {
            check(&core::array::from_fn(|i| {
                if i & (1 << bit) == 0 {
                    -8192.0
                } else {
                    8191.75
                }
            }));
        }
        let mut seed = 0x243f6a88u32;
        for _ in 0..256 {
            check(&core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed as i32) as f32 * (1.0 / 262144.0)
            }));
        }
    }
}
