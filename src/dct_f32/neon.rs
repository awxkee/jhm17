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

use super::{Lane, Lane4, four_lane};
use core::arch::aarch64::*;
use core::ops::{Add, Mul, Sub};

#[derive(Clone, Copy)]
struct F32x4(float32x4_t);
impl Lane for F32x4 {}

impl Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: NEON is part of the AArch64 baseline.
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }
}
impl Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }
}
impl Mul<f32> for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        unsafe { Self(vmulq_n_f32(self.0, rhs)) }
    }
}
impl Lane4 for F32x4 {
    #[inline(always)]
    fn load(input: &[f32; 4]) -> Self {
        // SAFETY: the array contains all four floats, with no alignment requirement.
        unsafe { Self(vld1q_f32(input.as_ptr())) }
    }
    #[inline(always)]
    fn store(self, out: &mut [f32; 4]) {
        unsafe { vst1q_f32(out.as_mut_ptr(), self.0) }
    }
    #[inline(always)]
    fn transpose([a, b, c, d]: [Self; 4]) -> [Self; 4] {
        unsafe {
            let ab0 = vreinterpretq_f64_f32(vtrn1q_f32(a.0, b.0));
            let ab1 = vreinterpretq_f64_f32(vtrn2q_f32(a.0, b.0));
            let cd0 = vreinterpretq_f64_f32(vtrn1q_f32(c.0, d.0));
            let cd1 = vreinterpretq_f64_f32(vtrn2q_f32(c.0, d.0));
            [
                Self(vreinterpretq_f32_f64(vzip1q_f64(ab0, cd0))),
                Self(vreinterpretq_f32_f64(vzip1q_f64(ab1, cd1))),
                Self(vreinterpretq_f32_f64(vzip2q_f64(ab0, cd0))),
                Self(vreinterpretq_f32_f64(vzip2q_f64(ab1, cd1))),
            ]
        }
    }
}

pub(super) fn transform<const INVERSE: bool>(input: &[f32; 64], out: &mut [f32; 64]) {
    four_lane::<F32x4, INVERSE>(input, out);
}
