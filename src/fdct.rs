/*
 * Ported from mozjpeg to rust
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2015, 2020, D. R. Commander.
 *
 * Conditions of distribution and use:
 * In plain English:
 *
 * 1. We don't promise that this software works.  (But if you find any bugs,
 *    please let us know!)
 * 2. You can use this software for whatever you want.  You don't have to pay us.
 * 3. You may not pretend that you wrote this software.  If you use it in a
 *    program, you must acknowledge somewhere in your documentation that
 *    you've used the IJG code.
 *
 * In legalese:
 *
 * The authors make NO WARRANTY or representation, either express or implied,
 * with respect to this software, its quality, accuracy, merchantability, or
 * fitness for a particular purpose.  This software is provided "AS IS", and you,
 * its user, assume the entire risk as to its quality and accuracy.
 *
 * This software is copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
 * All Rights Reserved except as specified below.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * software (or portions thereof) for any purpose, without fee, subject to these
 * conditions:
 * (1) If any part of the source code for this software is distributed, then this
 * README file must be included, with this copyright and no-warranty notice
 * unaltered; and any additions, deletions, or changes to the original files
 * must be clearly indicated in accompanying documentation.
 * (2) If only executable code is distributed, then the accompanying
 * documentation must state that "this software is based in part on the work of
 * the Independent JPEG Group".
 * (3) Permission for use of this software is granted only if the user accepts
 * full responsibility for any undesirable consequences; the authors accept
 * NO LIABILITY for damages of any kind.
 *
 * These conditions apply to any software derived from or based on the IJG code,
 * not just to the unmodified library.  If you use our work, you ought to
 * acknowledge us.
 *
 * Permission is NOT granted for the use of any IJG author's name or company name
 * in advertising or publicity relating to this software or products derived from
 * it.  This software may be referred to only as "the Independent JPEG Group's
 * software".
 *
 * We specifically permit and encourage the use of this software as the basis of
 * commercial products, provided that all warranty or liability claims are
 * assumed by the product vendor.
 *
 * This file contains a slower but more accurate integer implementation of the
 * forward DCT (Discrete Cosine Transform).
 *
 * A 2-D DCT can be done by 1-D DCT on each row followed by 1-D DCT
 * on each column.  Direct algorithms are also available, but they are
 * much more complex and seem not to be any faster when reduced to code.
 *
 * This implementation is based on an algorithm described in
 *   C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 * The primary algorithm described there uses 11 multiplies and 29 adds.
 * We use their alternate method with 12 multiplies and 32 adds.
 * The advantage of this method is that no data path contains more than one
 * multiplication; this allows a very simple and accurate implementation in
 * scaled fixed-point arithmetic, with a minimal number of shifts.
 */

use crate::encoder::AlignedBlock;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
#[path = "neon/fdct.rs"]
mod neon;

pub(crate) const CONST_BITS: i32 = 13;

pub use fdct_scalar as fdct;

// Both widths use the same scalar LL&M butterflies and rounding. The i64
// instantiation handles arbitrary i16 inputs without overflowing intermediates.
macro_rules! integer_dct_kernel {
    ($kernel:ident, $int:ty) => {
        #[inline(always)]
        fn $kernel<const SHIFT_BITS: i32>(data: [$int; 8]) -> [$int; 8] {
            #[inline(always)]
            fn descale(value: $int, shift: i32) -> $int {
                (value + (1 << (shift - 1))) >> shift
            }
            let tmp0 = data[0] + data[7];
            let tmp7 = data[0] - data[7];
            let tmp1 = data[1] + data[6];
            let tmp6 = data[1] - data[6];
            let tmp2 = data[2] + data[5];
            let tmp5 = data[2] - data[5];
            let tmp3 = data[3] + data[4];
            let tmp4 = data[3] - data[4];

            let tmp10 = tmp0 + tmp3;
            let tmp13 = tmp0 - tmp3;
            let tmp11 = tmp1 + tmp2;
            let tmp12 = tmp1 - tmp2;
            let even = |value: $int| {
                if SHIFT_BITS < 0 {
                    value << -SHIFT_BITS
                } else {
                    descale(value, SHIFT_BITS)
                }
            };
            let out0 = even(tmp10 + tmp11);
            let out4 = even(tmp10 - tmp11);
            let z1 = (tmp12 + tmp13) * 4433;
            let out2 = descale(z1 + tmp13 * 6270, $crate::fdct::CONST_BITS + SHIFT_BITS);
            let out6 = descale(z1 + tmp12 * -15137, $crate::fdct::CONST_BITS + SHIFT_BITS);

            let z1 = tmp4 + tmp7;
            let z2 = tmp5 + tmp6;
            let z3 = tmp4 + tmp6;
            let z4 = tmp5 + tmp7;
            let z5 = (z3 + z4) * 9633;
            let tmp4 = tmp4 * 2446;
            let tmp5 = tmp5 * 16819;
            let tmp6 = tmp6 * 25172;
            let tmp7 = tmp7 * 12299;
            let z1 = z1 * -7373;
            let z2 = z2 * -20995;
            let z3 = z3 * -16069;
            let z4 = z4 * -3196;
            let z3 = z3 + z5;
            let z4 = z4 + z5;
            let out7 = descale(tmp4 + z1 + z3, $crate::fdct::CONST_BITS + SHIFT_BITS);
            let out5 = descale(tmp5 + z2 + z4, $crate::fdct::CONST_BITS + SHIFT_BITS);
            let out3 = descale(tmp6 + z2 + z3, $crate::fdct::CONST_BITS + SHIFT_BITS);
            let out1 = descale(tmp7 + z1 + z4, $crate::fdct::CONST_BITS + SHIFT_BITS);
            [out0, out1, out2, out3, out4, out5, out6, out7]
        }
    };
}

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub(crate) use integer_dct_kernel;

macro_rules! scalar_integer_dct {
    ($kernel:ident, $transform:ident, $int:ty) => {
        integer_dct_kernel!($kernel, $int);

        pub(crate) fn $transform(data: &AlignedBlock) -> [$int; 64] {
            let mut work = [0 as $int; 64];
            for y in 0..8 {
                let input = core::array::from_fn(|x| data.data[y * 8 + x] as $int);
                // Row pass keeps two extra fractional bits.
                work[y * 8..y * 8 + 8].copy_from_slice(&$kernel::<-2>(input));
            }
            for x in 0..8 {
                let input = core::array::from_fn(|y| work[y * 8 + x]);
                // Column pass removes those two bits, retaining the usual 8x DCT scale.
                let output = $kernel::<2>(input);
                for y in 0..8 {
                    work[y * 8 + x] = output[y];
                }
            }
            work
        }
    };
}

scalar_integer_dct!(scalar_kernel_i32, scalar_dct_i32, i32);
scalar_integer_dct!(scalar_kernel_i64, scalar_dct_i64, i64);

pub fn fdct_scalar(data: &mut AlignedBlock) {
    data.data = scalar_dct_i32(data).map(|value| value as i16);
}

/// Transform Q2 spatial samples without narrowing the Q2 coefficients.
///
/// The usual factor-of-eight transform scale is retained, as are the two
/// fractional input bits. Divide the returned coefficients by four before
/// comparing with quantizers expressed in the original sample scale.
pub(crate) fn fdct_precise(data: &AlignedBlock) -> [i32; 64] {
    if data.data.iter().all(|&value| (-512..=512).contains(&value)) {
        // The LL&M i32 bound permits ten-bit input samples with PASS1_BITS=2.
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            neon::fdct_precise(data)
        }
        #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
        {
            scalar_dct_i32(data)
        }
    } else {
        // Continuation can exceed the physical Q2 range. i64 intermediates
        // cover the entire i16 domain; final coefficients fit comfortably in i32.
        scalar_dct_i64(data).map(|value| value as i32)
    }
}

#[cfg(test)]
mod tests {

    // Inputs and outputs are taken from libjpegs jpeg_fdct_islow for a typical image

    use super::*;

    const INPUT1: [i16; 64] = [
        -70, -71, -70, -68, -67, -67, -67, -67, -72, -73, -72, -70, -69, -69, -68, -69, -75, -76,
        -74, -73, -73, -72, -71, -70, -77, -78, -77, -75, -76, -75, -73, -71, -78, -77, -77, -76,
        -79, -77, -76, -75, -78, -78, -77, -77, -77, -77, -78, -77, -79, -79, -78, -78, -78, -78,
        -79, -78, -80, -79, -78, -78, -81, -80, -78, -76,
    ];

    const OUTPUT1: [i16; 64] = [
        -4786, -66, 2, -18, 12, 12, 5, -7, 223, -37, -8, 21, 8, 5, -4, 6, 60, 6, -10, 5, 0, -2, -1,
        5, 21, 21, -15, 12, -2, -7, 1, 0, -2, -5, 16, -15, 0, 5, -4, -8, 0, -7, -4, 6, 7, -4, 5, 4,
        3, 0, 1, -5, 0, -1, 4, 1, -5, 7, 0, -3, -6, 1, 1, -4,
    ];

    const INPUT2: [i16; 64] = [
        21, 28, 11, 24, -45, -37, -55, -103, 38, -8, 31, 17, -19, 49, 15, -76, 22, -48, -36, -31,
        -23, 35, -23, -72, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20,
        -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30,
        -45, -42, -44, -15, -20, -44,
    ];

    const OUTPUT2: [i16; 64] = [
        -1420, 717, 187, 910, -244, 579, 222, -191, 461, 487, -497, -29, -220, 179, 63, -95, 213,
        414, -235, -187, -108, 74, -73, -70, -63, 311, 13, -290, 17, -38, -180, -47, -254, 201,
        116, -247, 102, -109, -185, -36, -310, 107, 73, -91, 126, -121, -99, -37, -253, 43, -15,
        53, 101, -91, -3, -37, -136, 12, -44, 81, 53, -45, 31, -24,
    ];

    #[test]
    pub fn test_fdct_libjpeg() {
        let mut i1 = AlignedBlock::new(INPUT1.clone());
        fdct(&mut i1);
        assert_eq!(i1.data, OUTPUT1);

        let mut i2 = AlignedBlock::new(INPUT2.clone());
        fdct(&mut i2);
        assert_eq!(i2.data, OUTPUT2);
    }

    #[test]
    fn precise_q2_matches_original_scalar_when_coefficients_fit() {
        let mut seed = 0x31415926u32;
        for _ in 0..256 {
            let input = core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) % 1024) as i16 - 512
            });
            let block = AlignedBlock::new(input);
            let precise = fdct_precise(&block);
            let mut old = AlignedBlock::new(input);
            fdct_scalar(&mut old);
            for (z, &value) in precise.iter().enumerate() {
                if let Ok(narrow) = i16::try_from(value) {
                    assert_eq!(narrow, old.data[z], "position {z}");
                }
            }
            assert_eq!(precise, scalar_dct_i64(&block).map(|value| value as i32));
        }
    }

    #[test]
    fn precise_retains_coefficients_beyond_i16() {
        for value in [512i16, 1024, i16::MAX, i16::MIN] {
            let output = fdct_precise(&AlignedBlock::new([value; 64]));
            assert_eq!(output[0], i32::from(value) * 64);
            assert!(output[1..].iter().all(|&v| v == 0));
        }
        let output = fdct_precise(&AlignedBlock::new([512; 64]));
        assert_eq!(output[0], 32768);
    }

    #[test]

    fn precise_extremes_agree_with_independent_f64_dct() {
        fn reference(input: &[i16; 64], u: usize, v: usize) -> f64 {
            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    sum += f64::from(input[y * 8 + x])
                        * (((2 * x + 1) * u) as f64 * core::f64::consts::PI / 16.0).cos()
                        * (((2 * y + 1) * v) as f64 * core::f64::consts::PI / 16.0).cos();
                }
            }
            let cu = if u == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let cv = if v == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            2.0 * cu * cv * sum
        }
        let mut seed = 0x27182818u32;
        for pattern in 0..20 {
            let input = core::array::from_fn(|z| match pattern {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => {
                    if (z / 8 + z % 8) % 2 == 0 {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                }
                3 => {
                    if z % 2 == 0 {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                }
                4 => {
                    if z / 8 % 2 == 0 {
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
            let output = fdct_precise(&AlignedBlock::new(input));
            // The 13-bit cosine constants introduce a small amplitude-scaled
            // approximation error, in addition to row/column integer rounding.
            let amplitude = input.iter().map(|&v| i32::from(v).abs()).max().unwrap() as f64;
            let tolerance = 2.0 + amplitude * 0.01;
            for v in 0..8 {
                for u in 0..8 {
                    let error = (f64::from(output[v * 8 + u]) - reference(&input, u, v)).abs();
                    assert!(
                        error <= tolerance,
                        "pattern {pattern}, ({u}, {v}): {error} > {tolerance}"
                    );
                }
            }
        }
    }
}
