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
//! libjpeg's `jpeg_idct_islow`: dequantization of zigzag levels and the
//! accurate integer inverse DCT, as libjpeg-turbo's decoder runs it.
//!
//! The SIMD versions evaluate the same i32 expression per sample (wrapping
//! like the scalar one in release builds) and skip libjpeg's all-zero AC
//! shortcuts, which give identical results: every version agrees bit for bit.

use crate::writer::ZIGZAG;

pub(crate) const CONST_BITS: i32 = 13;
pub(crate) const PASS1_BITS: i32 = 2;
/// Descaling of the column and row passes
const PASS1_SHIFT: i32 = CONST_BITS - PASS1_BITS;
const PASS2_SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;

const F0_298: i32 = 2446;
const F0_390: i32 = 3196;
const F0_541: i32 = 4433;
const F0_765: i32 = 6270;
const F0_899: i32 = 7373;
const F1_175: i32 = 9633;
const F1_501: i32 = 12299;
const F1_847: i32 = 15137;
const F1_961: i32 = 16069;
const F2_053: i32 = 16819;
const F2_562: i32 = 20995;
const F3_072: i32 = 25172;

/// Inverse transform of zigzag `levels` against a natural-order DQT into
/// row-major samples
pub(crate) type Islow = fn(&[i16; 64], &[u16; 64], &mut [u8; 64]);

/// The fastest available version, detected once by the caller.
pub(crate) fn islow() -> Islow {
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is baseline on AArch64.
        |levels, table, out| unsafe { neon::islow(levels, table, out) }
    }
    #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
    {
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: installed only after establishing AVX2 support.
            return |levels, table, out| unsafe { avx2::islow(levels, table, out) };
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("sse4.1") {
            // SAFETY: installed only after establishing SSE4.1 support.
            return |levels, table, out| unsafe { sse41::islow(levels, table, out) };
        }
        islow_scalar
    }
}

/// The zigzag scan as natural-order indices (`writer::ZIGZAG`, computable
/// at compile time)
#[cfg_attr(not(all(feature = "neon", target_arch = "aarch64")), allow(dead_code))]
const fn zigzag_order() -> [u8; 64] {
    let mut order = [0u8; 64];
    let mut k = 0;
    let mut sum = 0;
    while sum < 15 {
        // Even anti-diagonals run up-right, odd ones down-left.
        let mut i = if sum < 8 { sum } else { 7 };
        loop {
            let j = sum - i;
            if j > 7 {
                break;
            }
            let (row, column) = if sum % 2 == 1 { (j, i) } else { (i, j) };
            order[k] = (row * 8 + column) as u8;
            k += 1;
            if i == 0 {
                break;
            }
            i -= 1;
        }
        sum += 1;
    }
    order
}

/// Dequantized coefficients in natural order
#[inline(always)]
fn dequantize(levels: &[i16; 64], table: &[u16; 64]) -> [i32; 64] {
    let mut coef = [0i32; 64];
    for (&level, &z) in levels.iter().zip(&ZIGZAG) {
        let z = usize::from(z);
        coef[z] = i32::from(level).wrapping_mul(i32::from(table[z]));
    }
    coef
}

// The scalar version is the fallback and test oracle; AArch64 always runs NEON.
#[cfg_attr(all(feature = "neon", target_arch = "aarch64"), allow(dead_code))]
#[inline]
pub(crate) fn descale(x: i32, n: i32) -> i32 {
    x.wrapping_add(1 << (n - 1)) >> n
}

// The scalar version is the fallback and test oracle; AArch64 always runs NEON.
#[cfg_attr(all(feature = "neon", target_arch = "aarch64"), allow(dead_code))]
/// libjpeg's `jpeg_idct_islow` of zigzag levels against a natural-order DQT
pub(crate) fn islow_scalar(levels: &[i16; 64], table: &[u16; 64], out: &mut [u8; 64]) {
    let coef = dequantize(levels, table);
    let mut ws = [[0i32; 8]; 8];
    let rows = coef.as_chunks::<8>().0;
    for c in 0..8 {
        let column: [i32; 8] = core::array::from_fn(|r| rows[r][c]);
        let values = if column[1..].iter().all(|&v| v == 0) {
            [column[0] << PASS1_BITS; 8]
        } else {
            pass(column, PASS1_SHIFT)
        };
        for (row, v) in ws.iter_mut().zip(values) {
            row[c] = v;
        }
    }
    for (w, out) in ws.iter().zip(out.as_chunks_mut::<8>().0) {
        let values = if w[1..].iter().all(|&v| v == 0) {
            [descale(w[0], PASS1_BITS + 3); 8]
        } else {
            pass(*w, PASS2_SHIFT)
        };
        for (out, v) in out.iter_mut().zip(values) {
            *out = v.wrapping_add(128).clamp(0, 255) as u8;
        }
    }
}

// The scalar version is the fallback and test oracle; AArch64 always runs NEON.
#[cfg_attr(all(feature = "neon", target_arch = "aarch64"), allow(dead_code))]
/// One 1-D pass of the islow IDCT, descaled by `shift`
#[inline]
pub(crate) fn pass(i: [i32; 8], shift: i32) -> [i32; 8] {
    let (z2, z3) = (i[2], i[6]);
    let z1 = z2.wrapping_add(z3).wrapping_mul(F0_541);
    let tmp2 = z1.wrapping_sub(z3.wrapping_mul(F1_847));
    let tmp3 = z1.wrapping_add(z2.wrapping_mul(F0_765));
    let tmp0 = i[0].wrapping_add(i[4]) << CONST_BITS;
    let tmp1 = i[0].wrapping_sub(i[4]) << CONST_BITS;
    let (tmp10, tmp13) = (tmp0.wrapping_add(tmp3), tmp0.wrapping_sub(tmp3));
    let (tmp11, tmp12) = (tmp1.wrapping_add(tmp2), tmp1.wrapping_sub(tmp2));

    let (t0, t1, t2, t3) = (i[7], i[5], i[3], i[1]);
    let z1 = t0.wrapping_add(t3).wrapping_mul(-F0_899);
    let z2 = t1.wrapping_add(t2).wrapping_mul(-F2_562);
    let z5 = t0
        .wrapping_add(t2)
        .wrapping_add(t1.wrapping_add(t3))
        .wrapping_mul(F1_175);
    let z3 = t0.wrapping_add(t2).wrapping_mul(-F1_961).wrapping_add(z5);
    let z4 = t1.wrapping_add(t3).wrapping_mul(-F0_390).wrapping_add(z5);
    let t0 = t0.wrapping_mul(F0_298).wrapping_add(z1.wrapping_add(z3));
    let t1 = t1.wrapping_mul(F2_053).wrapping_add(z2.wrapping_add(z4));
    let t2 = t2.wrapping_mul(F3_072).wrapping_add(z2.wrapping_add(z3));
    let t3 = t3.wrapping_mul(F1_501).wrapping_add(z1.wrapping_add(z4));

    [
        descale(tmp10.wrapping_add(t3), shift),
        descale(tmp11.wrapping_add(t2), shift),
        descale(tmp12.wrapping_add(t1), shift),
        descale(tmp13.wrapping_add(t0), shift),
        descale(tmp13.wrapping_sub(t0), shift),
        descale(tmp12.wrapping_sub(t1), shift),
        descale(tmp11.wrapping_sub(t2), shift),
        descale(tmp10.wrapping_sub(t3), shift),
    ]
}

/// [`pass`] on eight lanes at once. Each ISA module defines `add`, `sub`,
/// `mul` (by a constant), `shl13` and `descale::<SHIFT>` for its row type
/// `R`, all wrapping, and invokes this inside its target-feature scope.
#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(
        any(feature = "avx2", feature = "sse"),
        any(target_arch = "x86", target_arch = "x86_64")
    )
))]
macro_rules! vector_pass {
    ($feature:literal) => {
        #[inline]
        #[target_feature(enable = $feature)]
        fn pass<const SHIFT: i32>(i: [R; 8]) -> [R; 8] {
            use super::{
                F0_298, F0_390, F0_541, F0_765, F0_899, F1_175, F1_501, F1_847, F1_961, F2_053,
                F2_562, F3_072,
            };
            let (z2, z3) = (i[2], i[6]);
            let z1 = mul(add(z2, z3), F0_541);
            let tmp2 = sub(z1, mul(z3, F1_847));
            let tmp3 = add(z1, mul(z2, F0_765));
            let tmp0 = shl13(add(i[0], i[4]));
            let tmp1 = shl13(sub(i[0], i[4]));
            let (tmp10, tmp13) = (add(tmp0, tmp3), sub(tmp0, tmp3));
            let (tmp11, tmp12) = (add(tmp1, tmp2), sub(tmp1, tmp2));

            let (t0, t1, t2, t3) = (i[7], i[5], i[3], i[1]);
            let z1 = mul(add(t0, t3), -F0_899);
            let z2 = mul(add(t1, t2), -F2_562);
            let z5 = mul(add(add(t0, t2), add(t1, t3)), F1_175);
            let z3 = add(mul(add(t0, t2), -F1_961), z5);
            let z4 = add(mul(add(t1, t3), -F0_390), z5);
            let t0 = add(mul(t0, F0_298), add(z1, z3));
            let t1 = add(mul(t1, F2_053), add(z2, z4));
            let t2 = add(mul(t2, F3_072), add(z2, z3));
            let t3 = add(mul(t3, F1_501), add(z1, z4));

            [
                descale::<SHIFT>(add(tmp10, t3)),
                descale::<SHIFT>(add(tmp11, t2)),
                descale::<SHIFT>(add(tmp12, t1)),
                descale::<SHIFT>(add(tmp13, t0)),
                descale::<SHIFT>(sub(tmp13, t0)),
                descale::<SHIFT>(sub(tmp12, t1)),
                descale::<SHIFT>(sub(tmp11, t2)),
                descale::<SHIFT>(sub(tmp10, t3)),
            ]
        }
    };
}

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon {
    use super::{PASS1_SHIFT, PASS2_SHIFT};
    use core::arch::aarch64::*;

    /// Byte indices gathering each natural-order row of i16 levels from the
    /// zigzag block: `[row][half]` reads bytes `0..64` (`half` 0) or
    /// `64..128` (`half` 1) of the block, 0xFF selecting zero elsewhere.
    static GATHER: [[[u8; 16]; 2]; 8] = {
        let order = super::zigzag_order();
        let mut zigzag_of = [0usize; 64];
        let mut k = 0;
        while k < 64 {
            zigzag_of[order[k] as usize] = k;
            k += 1;
        }
        let mut table = [[[0xFFu8; 16]; 2]; 8];
        let mut row = 0;
        while row < 8 {
            let mut j = 0;
            while j < 16 {
                let byte = 2 * zigzag_of[row * 8 + j / 2] + j % 2;
                if byte < 64 {
                    table[row][0][j] = byte as u8;
                } else {
                    table[row][1][j] = (byte - 64) as u8;
                }
                j += 1;
            }
            row += 1;
        }
        table
    };

    /// Eight i32 lanes: columns 0..4 and 4..8 of one row
    type R = [int32x4_t; 2];

    #[inline]
    #[target_feature(enable = "neon")]
    fn add(a: R, b: R) -> R {
        [vaddq_s32(a[0], b[0]), vaddq_s32(a[1], b[1])]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn sub(a: R, b: R) -> R {
        [vsubq_s32(a[0], b[0]), vsubq_s32(a[1], b[1])]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn mul(a: R, c: i32) -> R {
        [vmulq_n_s32(a[0], c), vmulq_n_s32(a[1], c)]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn shl13(a: R) -> R {
        [vshlq_n_s32::<13>(a[0]), vshlq_n_s32::<13>(a[1])]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn descale<const N: i32>(a: R) -> R {
        // Wrapping add then arithmetic shift, as the scalar `descale`.
        let round = vdupq_n_s32(1 << (N - 1));
        [
            vshrq_n_s32::<N>(vaddq_s32(a[0], round)),
            vshrq_n_s32::<N>(vaddq_s32(a[1], round)),
        ]
    }

    vector_pass!("neon");

    #[inline]
    #[target_feature(enable = "neon")]
    fn transpose4(a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t) -> [int32x4_t; 4] {
        let ab = vtrnq_s32(a, b);
        let cd = vtrnq_s32(c, d);
        [
            vcombine_s32(vget_low_s32(ab.0), vget_low_s32(cd.0)),
            vcombine_s32(vget_low_s32(ab.1), vget_low_s32(cd.1)),
            vcombine_s32(vget_high_s32(ab.0), vget_high_s32(cd.0)),
            vcombine_s32(vget_high_s32(ab.1), vget_high_s32(cd.1)),
        ]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn transpose(r: [R; 8]) -> [R; 8] {
        let tl = transpose4(r[0][0], r[1][0], r[2][0], r[3][0]);
        let bl = transpose4(r[4][0], r[5][0], r[6][0], r[7][0]);
        let tr = transpose4(r[0][1], r[1][1], r[2][1], r[3][1]);
        let br = transpose4(r[4][1], r[5][1], r[6][1], r[7][1]);
        [
            [tl[0], bl[0]],
            [tl[1], bl[1]],
            [tl[2], bl[2]],
            [tl[3], bl[3]],
            [tr[0], br[0]],
            [tr[1], br[1]],
            [tr[2], br[2]],
            [tr[3], br[3]],
        ]
    }

    #[target_feature(enable = "neon")]
    pub(super) fn islow(levels: &[i16; 64], table: &[u16; 64], out: &mut [u8; 64]) {
        // De-zigzag with table lookups, then dequantize in i32 lanes
        // (wrapping like the scalar product).
        // SAFETY: the block is 128 bytes; both loads read within it.
        let (lo, hi) = unsafe {
            let bytes = levels.as_ptr().cast::<u8>();
            (vld1q_u8_x4(bytes), vld1q_u8_x4(bytes.add(64)))
        };
        let steps = table.as_chunks::<8>().0;
        let mut r = [[vdupq_n_s32(0); 2]; 8];
        for (y, row) in r.iter_mut().enumerate() {
            // SAFETY: each index row holds 16 bytes and each step row 8 u16.
            let (natural, step) = unsafe {
                let gather = &GATHER[y];
                let bytes = vorrq_u8(
                    vqtbl4q_u8(lo, vld1q_u8(gather[0].as_ptr())),
                    vqtbl4q_u8(hi, vld1q_u8(gather[1].as_ptr())),
                );
                (vreinterpretq_s16_u8(bytes), vld1q_u16(steps[y].as_ptr()))
            };
            *row = [
                vmulq_s32(
                    vmovl_s16(vget_low_s16(natural)),
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(step))),
                ),
                vmulq_s32(
                    vmovl_high_s16(natural),
                    vreinterpretq_s32_u32(vmovl_high_u16(step)),
                ),
            ];
        }
        let columns = pass::<PASS1_SHIFT>(r);
        let rows = transpose(pass::<PASS2_SHIFT>(transpose(columns)));
        let bias = vdupq_n_s32(128);
        for (row, out) in rows.iter().zip(out.as_chunks_mut::<8>().0) {
            let narrow = vcombine_s16(
                vqmovn_s32(vaddq_s32(row[0], bias)),
                vqmovn_s32(vaddq_s32(row[1], bias)),
            );
            // SAFETY: each output chunk holds exactly eight bytes.
            unsafe { vst1_u8(out.as_mut_ptr(), vqmovun_s16(narrow)) };
        }
    }
}

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
mod avx2 {
    use super::{PASS1_SHIFT, PASS2_SHIFT, dequantize};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Eight i32 lanes: one row
    type R = __m256i;

    #[inline]
    #[target_feature(enable = "avx2")]
    fn add(a: R, b: R) -> R {
        _mm256_add_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn sub(a: R, b: R) -> R {
        _mm256_sub_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn mul(a: R, c: i32) -> R {
        _mm256_mullo_epi32(a, _mm256_set1_epi32(c))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn shl13(a: R) -> R {
        _mm256_slli_epi32::<13>(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn descale<const N: i32>(a: R) -> R {
        // Wrapping add then arithmetic shift, as the scalar `descale`.
        _mm256_sra_epi32(
            _mm256_add_epi32(a, _mm256_set1_epi32(1 << (N - 1))),
            _mm_cvtsi32_si128(N),
        )
    }

    vector_pass!("avx2");

    #[inline]
    #[target_feature(enable = "avx2")]
    fn transpose(r: [R; 8]) -> [R; 8] {
        let t: [R; 8] = [
            _mm256_unpacklo_epi32(r[0], r[1]),
            _mm256_unpackhi_epi32(r[0], r[1]),
            _mm256_unpacklo_epi32(r[2], r[3]),
            _mm256_unpackhi_epi32(r[2], r[3]),
            _mm256_unpacklo_epi32(r[4], r[5]),
            _mm256_unpackhi_epi32(r[4], r[5]),
            _mm256_unpacklo_epi32(r[6], r[7]),
            _mm256_unpackhi_epi32(r[6], r[7]),
        ];
        let u: [R; 8] = [
            _mm256_unpacklo_epi64(t[0], t[2]),
            _mm256_unpackhi_epi64(t[0], t[2]),
            _mm256_unpacklo_epi64(t[1], t[3]),
            _mm256_unpackhi_epi64(t[1], t[3]),
            _mm256_unpacklo_epi64(t[4], t[6]),
            _mm256_unpackhi_epi64(t[4], t[6]),
            _mm256_unpacklo_epi64(t[5], t[7]),
            _mm256_unpackhi_epi64(t[5], t[7]),
        ];
        [
            _mm256_permute2x128_si256::<0x20>(u[0], u[4]),
            _mm256_permute2x128_si256::<0x20>(u[1], u[5]),
            _mm256_permute2x128_si256::<0x20>(u[2], u[6]),
            _mm256_permute2x128_si256::<0x20>(u[3], u[7]),
            _mm256_permute2x128_si256::<0x31>(u[0], u[4]),
            _mm256_permute2x128_si256::<0x31>(u[1], u[5]),
            _mm256_permute2x128_si256::<0x31>(u[2], u[6]),
            _mm256_permute2x128_si256::<0x31>(u[3], u[7]),
        ]
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn islow(levels: &[i16; 64], table: &[u16; 64], out: &mut [u8; 64]) {
        let coef = dequantize(levels, table);
        let mut r = [_mm256_setzero_si256(); 8];
        for (row, chunk) in r.iter_mut().zip(coef.as_chunks::<8>().0) {
            // SAFETY: each chunk holds exactly eight i32 values.
            *row = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
        }
        let columns = pass::<PASS1_SHIFT>(r);
        let rows = transpose(pass::<PASS2_SHIFT>(transpose(columns)));
        let bias = _mm256_set1_epi32(128);
        for (row, out) in rows.iter().zip(out.as_chunks_mut::<8>().0) {
            let row = _mm256_add_epi32(*row, bias);
            let narrow = _mm_packs_epi32(
                _mm256_castsi256_si128(row),
                _mm256_extracti128_si256::<1>(row),
            );
            let bytes = _mm_packus_epi16(narrow, narrow);
            // SAFETY: each output chunk holds exactly eight bytes.
            unsafe { _mm_storel_epi64(out.as_mut_ptr().cast(), bytes) };
        }
    }
}

#[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
mod sse41 {
    use super::{PASS1_SHIFT, PASS2_SHIFT, dequantize};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Eight i32 lanes: columns 0..4 and 4..8 of one row
    type R = [__m128i; 2];

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn add(a: R, b: R) -> R {
        [_mm_add_epi32(a[0], b[0]), _mm_add_epi32(a[1], b[1])]
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn sub(a: R, b: R) -> R {
        [_mm_sub_epi32(a[0], b[0]), _mm_sub_epi32(a[1], b[1])]
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn mul(a: R, c: i32) -> R {
        let c = _mm_set1_epi32(c);
        [_mm_mullo_epi32(a[0], c), _mm_mullo_epi32(a[1], c)]
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn shl13(a: R) -> R {
        [_mm_slli_epi32::<13>(a[0]), _mm_slli_epi32::<13>(a[1])]
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn descale<const N: i32>(a: R) -> R {
        // Wrapping add then arithmetic shift, as the scalar `descale`.
        let round = _mm_set1_epi32(1 << (N - 1));
        let shift = _mm_cvtsi32_si128(N);
        [
            _mm_sra_epi32(_mm_add_epi32(a[0], round), shift),
            _mm_sra_epi32(_mm_add_epi32(a[1], round), shift),
        ]
    }

    vector_pass!("sse4.1");

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn transpose4(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> [__m128i; 4] {
        let ab_lo = _mm_unpacklo_epi32(a, b);
        let cd_lo = _mm_unpacklo_epi32(c, d);
        let ab_hi = _mm_unpackhi_epi32(a, b);
        let cd_hi = _mm_unpackhi_epi32(c, d);
        [
            _mm_unpacklo_epi64(ab_lo, cd_lo),
            _mm_unpackhi_epi64(ab_lo, cd_lo),
            _mm_unpacklo_epi64(ab_hi, cd_hi),
            _mm_unpackhi_epi64(ab_hi, cd_hi),
        ]
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn transpose(r: [R; 8]) -> [R; 8] {
        let tl = transpose4(r[0][0], r[1][0], r[2][0], r[3][0]);
        let bl = transpose4(r[4][0], r[5][0], r[6][0], r[7][0]);
        let tr = transpose4(r[0][1], r[1][1], r[2][1], r[3][1]);
        let br = transpose4(r[4][1], r[5][1], r[6][1], r[7][1]);
        [
            [tl[0], bl[0]],
            [tl[1], bl[1]],
            [tl[2], bl[2]],
            [tl[3], bl[3]],
            [tr[0], br[0]],
            [tr[1], br[1]],
            [tr[2], br[2]],
            [tr[3], br[3]],
        ]
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) fn islow(levels: &[i16; 64], table: &[u16; 64], out: &mut [u8; 64]) {
        let coef = dequantize(levels, table);
        let chunks = coef.as_chunks::<4>().0;
        let mut r = [[_mm_setzero_si128(); 2]; 8];
        for (y, row) in r.iter_mut().enumerate() {
            // SAFETY: each chunk holds exactly four i32 values.
            *row = unsafe {
                [
                    _mm_loadu_si128(chunks[2 * y].as_ptr().cast()),
                    _mm_loadu_si128(chunks[2 * y + 1].as_ptr().cast()),
                ]
            };
        }
        let columns = pass::<PASS1_SHIFT>(r);
        let rows = transpose(pass::<PASS2_SHIFT>(transpose(columns)));
        let bias = _mm_set1_epi32(128);
        for (row, out) in rows.iter().zip(out.as_chunks_mut::<8>().0) {
            let narrow = _mm_packs_epi32(_mm_add_epi32(row[0], bias), _mm_add_epi32(row[1], bias));
            let bytes = _mm_packus_epi16(narrow, narrow);
            // SAFETY: each output chunk holds exactly eight bytes.
            unsafe { _mm_storel_epi64(out.as_mut_ptr().cast(), bytes) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn compile_time_zigzag_matches_writer() {
        assert_eq!(zigzag_order(), ZIGZAG);
    }

    #[test]
    fn dc_only_matches_shortcut() {
        let mut levels = [0i16; 64];
        levels[0] = 5;
        let table = [16u16; 64];
        let mut out = [0u8; 64];
        islow_scalar(&levels, &table, &mut out);
        // 5 × 16 / 8 = 10
        assert!(out.iter().all(|&v| v == 138));
    }

    /// Every compiled version that this CPU runs, the dispatched one first
    fn versions() -> Vec<(&'static str, Islow)> {
        #[allow(unused_mut)]
        let mut versions: Vec<(&'static str, Islow)> = alloc::vec![("dispatch", islow())];
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        // SAFETY: NEON is baseline on AArch64.
        versions.push(("neon", |l, t, o| unsafe { neon::islow(l, t, o) }));
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was just established.
            versions.push(("avx2", |l, t, o| unsafe { avx2::islow(l, t, o) }));
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("sse4.1") {
            // SAFETY: SSE4.1 support was just established.
            versions.push(("sse4.1", |l, t, o| unsafe { sse41::islow(l, t, o) }));
        }
        versions
    }

    #[test]
    fn simd_matches_scalar_bit_exactly() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..20_000 {
            let table: [u16; 64] = core::array::from_fn(|_| 1 + (next() % 255) as u16);
            // Sparse and dense blocks, DC-only and AC-only rows and columns,
            // and saturating outputs at both ends.
            let density = [1, 3, 8, 64][case % 4];
            let levels: [i16; 64] = core::array::from_fn(|k| {
                if k > 0 && (next() % 64) as usize >= density {
                    return 0;
                }
                let limit = (2047 / table[usize::from(ZIGZAG[k])]).max(1) as i64;
                let value = (next() % (2 * limit as u64 + 1)) as i64 - limit;
                value as i16
            });
            let mut expected = [0u8; 64];
            islow_scalar(&levels, &table, &mut expected);
            for (name, version) in versions() {
                let mut actual = [0u8; 64];
                version(&levels, &table, &mut actual);
                assert_eq!(actual, expected, "{name} case {case} levels {levels:?}");
            }
        }
    }
}
