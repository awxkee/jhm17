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
//! Four independent butterfly transforms in baseline AArch64 NEON lanes.

use core::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn forward(x: [float32x4_t; 8]) -> [float32x4_t; 8] {
    let t0 = vaddq_f32(x[0], x[7]);
    let t7 = vsubq_f32(x[0], x[7]);
    let t1 = vaddq_f32(x[1], x[6]);
    let t6 = vsubq_f32(x[1], x[6]);
    let t2 = vaddq_f32(x[2], x[5]);
    let t5 = vsubq_f32(x[2], x[5]);
    let t3 = vaddq_f32(x[3], x[4]);
    let t4 = vsubq_f32(x[3], x[4]);

    let t10 = vaddq_f32(t0, t3);
    let t13 = vsubq_f32(t0, t3);
    let t11 = vaddq_f32(t1, t2);
    let t12 = vsubq_f32(t1, t2);
    let z = vmulq_n_f32(vaddq_f32(t12, t13), 0.541_196_1);
    let y0 = vaddq_f32(t10, t11);
    let y4 = vsubq_f32(t10, t11);
    let y2 = vaddq_f32(z, vmulq_n_f32(t13, 0.765_366_85));
    let y6 = vaddq_f32(z, vmulq_n_f32(t12, -1.847_759));

    let z1 = vmulq_n_f32(vaddq_f32(t4, t7), -0.899_976_2);
    let z2 = vmulq_n_f32(vaddq_f32(t5, t6), -2.562_915_6);
    let z3 = vaddq_f32(t4, t6);
    let z4 = vaddq_f32(t5, t7);
    let z5 = vmulq_n_f32(vaddq_f32(z3, z4), 1.175_875_5);
    let z3 = vaddq_f32(vmulq_n_f32(z3, -1.961_570_5), z5);
    let z4 = vaddq_f32(vmulq_n_f32(z4, -0.390_180_65), z5);
    let y7 = vaddq_f32(vaddq_f32(vmulq_n_f32(t4, 0.298_631_34), z1), z3);
    let y5 = vaddq_f32(vaddq_f32(vmulq_n_f32(t5, 2.053_12), z2), z4);
    let y3 = vaddq_f32(vaddq_f32(vmulq_n_f32(t6, 3.072_711), z2), z3);
    let y1 = vaddq_f32(vaddq_f32(vmulq_n_f32(t7, 1.501_321_1), z1), z4);
    [y0, y1, y2, y3, y4, y5, y6, y7]
}

#[inline]
#[target_feature(enable = "neon")]
fn inverse(y: [float32x4_t; 8]) -> [float32x4_t; 8] {
    let t10 = vaddq_f32(y[0], y[4]);
    let t11 = vsubq_f32(y[0], y[4]);
    let z = vmulq_n_f32(vaddq_f32(y[2], y[6]), 0.541_196_1);
    let t13 = vaddq_f32(z, vmulq_n_f32(y[2], 0.765_366_85));
    let t12 = vaddq_f32(z, vmulq_n_f32(y[6], -1.847_759));
    let t0 = vaddq_f32(t10, t13);
    let t3 = vsubq_f32(t10, t13);
    let t1 = vaddq_f32(t11, t12);
    let t2 = vsubq_f32(t11, t12);

    let z1 = vmulq_n_f32(vaddq_f32(y[7], y[1]), -0.899_976_2);
    let z2 = vmulq_n_f32(vaddq_f32(y[5], y[3]), -2.562_915_6);
    let z3 = vaddq_f32(y[7], y[3]);
    let z4 = vaddq_f32(y[5], y[1]);
    let z5 = vmulq_n_f32(vaddq_f32(z3, z4), 1.175_875_5);
    let z3 = vaddq_f32(vmulq_n_f32(z3, -1.961_570_5), z5);
    let z4 = vaddq_f32(vmulq_n_f32(z4, -0.390_180_65), z5);
    let t4 = vaddq_f32(vaddq_f32(vmulq_n_f32(y[7], 0.298_631_34), z1), z3);
    let t5 = vaddq_f32(vaddq_f32(vmulq_n_f32(y[5], 2.053_12), z2), z4);
    let t6 = vaddq_f32(vaddq_f32(vmulq_n_f32(y[3], 3.072_711), z2), z3);
    let t7 = vaddq_f32(vaddq_f32(vmulq_n_f32(y[1], 1.501_321_1), z1), z4);
    [
        vaddq_f32(t0, t7),
        vaddq_f32(t1, t6),
        vaddq_f32(t2, t5),
        vaddq_f32(t3, t4),
        vsubq_f32(t3, t4),
        vsubq_f32(t2, t5),
        vsubq_f32(t1, t6),
        vsubq_f32(t0, t7),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn kernel<const INVERSE: bool>(input: [float32x4_t; 8]) -> [float32x4_t; 8] {
    // Separate multiply/add operations preserve the scalar rounding; FMA would
    // change coefficients at quantization boundaries.
    if INVERSE {
        inverse(input)
    } else {
        forward(input)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose([a, b, c, d]: [float32x4_t; 4]) -> [float32x4_t; 4] {
    let ab = vtrnq_f32(a, b);
    let cd = vtrnq_f32(c, d);
    [
        vcombine_f32(vget_low_f32(ab.0), vget_low_f32(cd.0)),
        vcombine_f32(vget_low_f32(ab.1), vget_low_f32(cd.1)),
        vcombine_f32(vget_high_f32(ab.0), vget_high_f32(cd.0)),
        vcombine_f32(vget_high_f32(ab.1), vget_high_f32(cd.1)),
    ]
}

#[target_feature(enable = "neon")]
fn transform_neon<const INVERSE: bool>(input: &[f32; 64], out: &mut [f32; 64]) {
    // Row transforms run first, matching the scalar evaluation order. Each
    // register carries one sample from four independent rows.
    let rows: [[float32x4_t; 8]; 2] = core::array::from_fn(|group| {
        // SAFETY: each pair of loads reads one complete eight-sample row.
        let input: [float32x4x2_t; 4] = core::array::from_fn(|y| unsafe {
            vld1q_f32_x2(input.as_ptr().add((group * 4 + y) * 8))
        });
        let lo = transpose(input.map(|row| row.0));
        let hi = transpose(input.map(|row| row.1));
        kernel::<INVERSE>([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    });
    for x in [0, 4] {
        let top = transpose(core::array::from_fn(|i| rows[0][x + i]));
        let bottom = transpose(core::array::from_fn(|i| rows[1][x + i]));
        let columns = kernel::<INVERSE>([
            top[0], top[1], top[2], top[3], bottom[0], bottom[1], bottom[2], bottom[3],
        ]);
        for (y, value) in columns.into_iter().enumerate() {
            // SAFETY: four samples fit within row y at either x = 0 or x = 4.
            unsafe { vst1q_f32(out.as_mut_ptr().add(y * 8 + x), vmulq_n_f32(value, 0.125)) };
        }
    }
}

pub(crate) fn transform<const INVERSE: bool>(input: &[f32; 64], out: &mut [f32; 64]) {
    // SAFETY: NEON is part of the AArch64 baseline.
    unsafe { transform_neon::<INVERSE>(input, out) }
}
