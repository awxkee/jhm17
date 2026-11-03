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
//! Four parallel LL&M transforms with native NEON rounding and multiply-adds.

use core::arch::aarch64::*;

use crate::encoder::AlignedBlock;

#[inline]
#[target_feature(enable = "neon")]
fn transpose([a, b, c, d]: [int32x4_t; 4]) -> [int32x4_t; 4] {
    let ab0 = vreinterpretq_s64_s32(vtrn1q_s32(a, b));
    let ab1 = vreinterpretq_s64_s32(vtrn2q_s32(a, b));
    let cd0 = vreinterpretq_s64_s32(vtrn1q_s32(c, d));
    let cd1 = vreinterpretq_s64_s32(vtrn2q_s32(c, d));
    [
        vreinterpretq_s32_s64(vzip1q_s64(ab0, cd0)),
        vreinterpretq_s32_s64(vzip1q_s64(ab1, cd1)),
        vreinterpretq_s32_s64(vzip2q_s64(ab0, cd0)),
        vreinterpretq_s32_s64(vzip2q_s64(ab1, cd1)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn kernel<const ROW: bool>(data: [int32x4_t; 8]) -> [int32x4_t; 8] {
    let tmp0 = vaddq_s32(data[0], data[7]);
    let tmp7 = vsubq_s32(data[0], data[7]);
    let tmp1 = vaddq_s32(data[1], data[6]);
    let tmp6 = vsubq_s32(data[1], data[6]);
    let tmp2 = vaddq_s32(data[2], data[5]);
    let tmp5 = vsubq_s32(data[2], data[5]);
    let tmp3 = vaddq_s32(data[3], data[4]);
    let tmp4 = vsubq_s32(data[3], data[4]);

    let tmp10 = vaddq_s32(tmp0, tmp3);
    let tmp13 = vsubq_s32(tmp0, tmp3);
    let tmp11 = vaddq_s32(tmp1, tmp2);
    let tmp12 = vsubq_s32(tmp1, tmp2);
    let even = |value| {
        if ROW {
            vshlq_n_s32::<2>(value)
        } else {
            vrshrq_n_s32::<2>(value)
        }
    };
    let descale = |value| {
        if ROW {
            vrshrq_n_s32::<11>(value)
        } else {
            vrshrq_n_s32::<15>(value)
        }
    };
    let out0 = even(vaddq_s32(tmp10, tmp11));
    let out4 = even(vsubq_s32(tmp10, tmp11));
    let z1 = vmulq_n_s32(vaddq_s32(tmp12, tmp13), 4433);
    let out2 = descale(vmlaq_n_s32(z1, tmp13, 6270));
    let out6 = descale(vmlaq_n_s32(z1, tmp12, -15137));

    let z1 = vmulq_n_s32(vaddq_s32(tmp4, tmp7), -7373);
    let z2 = vmulq_n_s32(vaddq_s32(tmp5, tmp6), -20995);
    let z3 = vaddq_s32(tmp4, tmp6);
    let z4 = vaddq_s32(tmp5, tmp7);
    let z5 = vmulq_n_s32(vaddq_s32(z3, z4), 9633);
    let z3 = vmlaq_n_s32(z5, z3, -16069);
    let z4 = vmlaq_n_s32(z5, z4, -3196);
    let out7 = descale(vmlaq_n_s32(vaddq_s32(z1, z3), tmp4, 2446));
    let out5 = descale(vmlaq_n_s32(vaddq_s32(z2, z4), tmp5, 16819));
    let out3 = descale(vmlaq_n_s32(vaddq_s32(z2, z3), tmp6, 25172));
    let out1 = descale(vmlaq_n_s32(vaddq_s32(z1, z4), tmp7, 12299));
    [out0, out1, out2, out3, out4, out5, out6, out7]
}

// The caller checks the ten-bit input bound before selecting this kernel.
#[target_feature(enable = "neon")]
pub(crate) fn fdct_precise(data: &AlignedBlock) -> [i32; 64] {
    let rows: [[int32x4_t; 8]; 2] = core::array::from_fn(|group| {
        // SAFETY: each load is exactly one of the eight complete block rows.
        let input: [int16x8_t; 4] = core::array::from_fn(|y| unsafe {
            vld1q_s16(data.data.as_ptr().add((group * 4 + y) * 8))
        });
        let lo = transpose(input.map(|row| vmovl_s16(vget_low_s16(row))));
        let hi = transpose(input.map(|row| vmovl_high_s16(row)));
        kernel::<true>([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    });
    let mut out = [0; 64];
    for x in [0, 4] {
        let top = transpose(core::array::from_fn(|i| rows[0][x + i]));
        let bottom = transpose(core::array::from_fn(|i| rows[1][x + i]));
        let columns = kernel::<false>([
            top[0], top[1], top[2], top[3], bottom[0], bottom[1], bottom[2], bottom[3],
        ]);
        for (y, row) in columns.into_iter().enumerate() {
            // SAFETY: x is 0 or 4, so all four output lanes are within row y.
            unsafe { vst1q_s32(out.as_mut_ptr().add(y * 8 + x), row) };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precise_simd_matches_wide_scalar() {
        crate::fdct::tests::check_precise_simd(|block| {
            // SAFETY: NEON is baseline on AArch64.
            unsafe { fdct_precise(block) }
        });
    }

    #[test]
    fn integer_simd_matches_scalar() {
        let mut seed = 0x27182818u32;
        for _ in 0..1024 {
            let input = core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as i16 - 128
            });
            let mut simd = AlignedBlock::new(input);
            let mut scalar = AlignedBlock::new(input);
            crate::fdct::fdct_neon(&mut simd);
            crate::fdct::fdct_scalar(&mut scalar);
            assert_eq!(simd.data, scalar.data);
        }
        // Integer clipping continuation uses the same wide fallback as Q2.
        for value in [i16::MIN, -513, 513, i16::MAX] {
            let mut simd = AlignedBlock::new([value; 64]);
            let expected = crate::fdct::scalar_dct_i64(&simd).map(|v| v as i16);
            crate::fdct::fdct_neon(&mut simd);
            assert_eq!(simd.data, expected);
        }
    }
}
