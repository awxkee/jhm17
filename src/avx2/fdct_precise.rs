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
//! Eight parallel LL&M transforms retaining the full i32 Q2 coefficients.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::encoder::AlignedBlock;

// Installed by EncoderFunctions only after detecting AVX2. Continuation can
// produce samples outside the safe i32 butterfly range; retain the wide fallback.
pub(crate) fn fdct_precise_avx2(data: &AlignedBlock) -> [i32; 64] {
    if data.data.iter().all(|&value| (-512..=512).contains(&value)) {
        // SAFETY: the encoder established AVX2 support before selecting us.
        unsafe { transform(data) }
    } else {
        crate::fdct::scalar_dct_i64(data).map(|value| value as i32)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose([a, b, c, d, e, f, g, h]: [__m256i; 8]) -> [__m256i; 8] {
    let ab0 = _mm256_unpacklo_epi32(a, b);
    let ab1 = _mm256_unpackhi_epi32(a, b);
    let cd0 = _mm256_unpacklo_epi32(c, d);
    let cd1 = _mm256_unpackhi_epi32(c, d);
    let ef0 = _mm256_unpacklo_epi32(e, f);
    let ef1 = _mm256_unpackhi_epi32(e, f);
    let gh0 = _mm256_unpacklo_epi32(g, h);
    let gh1 = _mm256_unpackhi_epi32(g, h);
    let top = [
        _mm256_unpacklo_epi64(ab0, cd0),
        _mm256_unpackhi_epi64(ab0, cd0),
        _mm256_unpacklo_epi64(ab1, cd1),
        _mm256_unpackhi_epi64(ab1, cd1),
    ];
    let bottom = [
        _mm256_unpacklo_epi64(ef0, gh0),
        _mm256_unpackhi_epi64(ef0, gh0),
        _mm256_unpacklo_epi64(ef1, gh1),
        _mm256_unpackhi_epi64(ef1, gh1),
    ];
    [
        _mm256_permute2x128_si256::<0x20>(top[0], bottom[0]),
        _mm256_permute2x128_si256::<0x20>(top[1], bottom[1]),
        _mm256_permute2x128_si256::<0x20>(top[2], bottom[2]),
        _mm256_permute2x128_si256::<0x20>(top[3], bottom[3]),
        _mm256_permute2x128_si256::<0x31>(top[0], bottom[0]),
        _mm256_permute2x128_si256::<0x31>(top[1], bottom[1]),
        _mm256_permute2x128_si256::<0x31>(top[2], bottom[2]),
        _mm256_permute2x128_si256::<0x31>(top[3], bottom[3]),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn kernel<const ROW: bool>(data: [__m256i; 8]) -> [__m256i; 8] {
    let tmp0 = _mm256_add_epi32(data[0], data[7]);
    let tmp7 = _mm256_sub_epi32(data[0], data[7]);
    let tmp1 = _mm256_add_epi32(data[1], data[6]);
    let tmp6 = _mm256_sub_epi32(data[1], data[6]);
    let tmp2 = _mm256_add_epi32(data[2], data[5]);
    let tmp5 = _mm256_sub_epi32(data[2], data[5]);
    let tmp3 = _mm256_add_epi32(data[3], data[4]);
    let tmp4 = _mm256_sub_epi32(data[3], data[4]);

    let tmp10 = _mm256_add_epi32(tmp0, tmp3);
    let tmp13 = _mm256_sub_epi32(tmp0, tmp3);
    let tmp11 = _mm256_add_epi32(tmp1, tmp2);
    let tmp12 = _mm256_sub_epi32(tmp1, tmp2);
    let even = |value| {
        if ROW {
            _mm256_slli_epi32::<2>(value)
        } else {
            _mm256_srai_epi32::<2>(_mm256_add_epi32(value, _mm256_set1_epi32(2)))
        }
    };
    let descale = |value| {
        if ROW {
            _mm256_srai_epi32::<11>(_mm256_add_epi32(value, _mm256_set1_epi32(1 << 10)))
        } else {
            _mm256_srai_epi32::<15>(_mm256_add_epi32(value, _mm256_set1_epi32(1 << 14)))
        }
    };
    let mul = |value, factor| _mm256_mullo_epi32(value, _mm256_set1_epi32(factor));
    let out0 = even(_mm256_add_epi32(tmp10, tmp11));
    let out4 = even(_mm256_sub_epi32(tmp10, tmp11));
    let z1 = mul(_mm256_add_epi32(tmp12, tmp13), 4433);
    let out2 = descale(_mm256_add_epi32(z1, mul(tmp13, 6270)));
    let out6 = descale(_mm256_add_epi32(z1, mul(tmp12, -15137)));

    let z1 = mul(_mm256_add_epi32(tmp4, tmp7), -7373);
    let z2 = mul(_mm256_add_epi32(tmp5, tmp6), -20995);
    let z3 = _mm256_add_epi32(tmp4, tmp6);
    let z4 = _mm256_add_epi32(tmp5, tmp7);
    let z5 = mul(_mm256_add_epi32(z3, z4), 9633);
    let z3 = _mm256_add_epi32(mul(z3, -16069), z5);
    let z4 = _mm256_add_epi32(mul(z4, -3196), z5);
    let out7 = descale(_mm256_add_epi32(mul(tmp4, 2446), _mm256_add_epi32(z1, z3)));
    let out5 = descale(_mm256_add_epi32(mul(tmp5, 16819), _mm256_add_epi32(z2, z4)));
    let out3 = descale(_mm256_add_epi32(mul(tmp6, 25172), _mm256_add_epi32(z2, z3)));
    let out1 = descale(_mm256_add_epi32(mul(tmp7, 12299), _mm256_add_epi32(z1, z4)));
    [out0, out1, out2, out3, out4, out5, out6, out7]
}

#[target_feature(enable = "avx2")]
fn transform(data: &AlignedBlock) -> [i32; 64] {
    // SAFETY: every load consumes exactly one complete eight-sample row.
    let input = core::array::from_fn(|y| unsafe {
        _mm256_cvtepi16_epi32(_mm_loadu_si128(data.data.as_ptr().add(y * 8).cast()))
    });
    let rows = kernel::<true>(transpose(input));
    let columns = kernel::<false>(transpose(rows));
    let mut out = [0; 64];
    for (y, row) in columns.into_iter().enumerate() {
        // SAFETY: each store initializes exactly eight elements of output row y.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().add(y * 8).cast(), row) };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precise_simd_matches_wide_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            std::eprintln!("AVX2 unavailable; skipping runtime FDCT checks");
            return;
        }
        crate::fdct::tests::check_precise_simd(fdct_precise_avx2);
    }

    #[test]
    fn continuation_uses_wide_fallback() {
        if !std::is_x86_feature_detected!("avx2") {
            std::eprintln!("AVX2 unavailable; skipping runtime FDCT checks");
            return;
        }
        let mut seed = 0x27182818u32;
        for pattern in 0..256 {
            let input = core::array::from_fn(|i| match pattern {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => {
                    if i % 2 == 0 {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                }
                _ => {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 16) as i16
                }
            });
            let block = AlignedBlock::new(input);
            assert_eq!(
                fdct_precise_avx2(&block),
                crate::fdct::scalar_dct_i64(&block).map(|v| v as i32)
            );
        }
    }
}
