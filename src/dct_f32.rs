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

use core::ops::{Add, Mul, Sub};

#[cfg(test)]
#[path = "dct_f32/reference.rs"]
pub(crate) mod reference;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
#[path = "dct_f32/neon.rs"]
mod neon;
#[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
#[path = "dct_f32/sse2.rs"]
mod sse2;

pub(super) trait Lane:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f32, Output = Self>
{
}
impl Lane for f32 {}

#[inline(always)]
pub(super) fn forward<T: Lane>(x: [T; 8]) -> [T; 8] {
    let t0 = x[0] + x[7];
    let t7 = x[0] - x[7];
    let t1 = x[1] + x[6];
    let t6 = x[1] - x[6];
    let t2 = x[2] + x[5];
    let t5 = x[2] - x[5];
    let t3 = x[3] + x[4];
    let t4 = x[3] - x[4];

    let t10 = t0 + t3;
    let t13 = t0 - t3;
    let t11 = t1 + t2;
    let t12 = t1 - t2;
    let z = (t12 + t13) * 0.541_196_1;
    let y0 = t10 + t11;
    let y4 = t10 - t11;
    let y2 = z + t13 * 0.765_366_85;
    let y6 = z + t12 * -1.847_759;

    let z1 = (t4 + t7) * -0.899_976_2;
    let z2 = (t5 + t6) * -2.562_915_6;
    let z3 = t4 + t6;
    let z4 = t5 + t7;
    let z5 = (z3 + z4) * 1.175_875_5;
    let z3 = z3 * -1.961_570_5 + z5;
    let z4 = z4 * -0.390_180_65 + z5;
    let y7 = t4 * 0.298_631_34 + z1 + z3;
    let y5 = t5 * 2.053_12 + z2 + z4;
    let y3 = t6 * 3.072_711 + z2 + z3;
    let y1 = t7 * 1.501_321_1 + z1 + z4;
    [y0, y1, y2, y3, y4, y5, y6, y7]
}

#[inline(always)]
pub(super) fn inverse<T: Lane>(y: [T; 8]) -> [T; 8] {
    let t10 = y[0] + y[4];
    let t11 = y[0] - y[4];
    let z = (y[2] + y[6]) * 0.541_196_1;
    let t13 = z + y[2] * 0.765_366_85;
    let t12 = z + y[6] * -1.847_759;
    let t0 = t10 + t13;
    let t3 = t10 - t13;
    let t1 = t11 + t12;
    let t2 = t11 - t12;

    let z1 = (y[7] + y[1]) * -0.899_976_2;
    let z2 = (y[5] + y[3]) * -2.562_915_6;
    let z3 = y[7] + y[3];
    let z4 = y[5] + y[1];
    let z5 = (z3 + z4) * 1.175_875_5;
    let z3 = z3 * -1.961_570_5 + z5;
    let z4 = z4 * -0.390_180_65 + z5;
    let t4 = y[7] * 0.298_631_34 + z1 + z3;
    let t5 = y[5] * 2.053_12 + z2 + z4;
    let t6 = y[3] * 3.072_711 + z2 + z3;
    let t7 = y[1] * 1.501_321_1 + z1 + z4;
    [
        t0 + t7,
        t1 + t6,
        t2 + t5,
        t3 + t4,
        t3 - t4,
        t2 - t5,
        t1 - t6,
        t0 - t7,
    ]
}

#[inline(always)]
pub(super) fn kernel<T: Lane, const INVERSE: bool>(input: [T; 8]) -> [T; 8] {
    if INVERSE {
        inverse(input)
    } else {
        forward(input)
    }
}

#[cfg(any(test, not(all(feature = "neon", target_arch = "aarch64"))))]
pub(crate) fn scalar<const INVERSE: bool>(input: &[f32; 64], out: &mut [f32; 64]) {
    let mut work = [0.0; 64];
    for (row, source) in work
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(input.as_chunks::<8>().0)
    {
        *row = kernel::<_, INVERSE>(*source);
    }
    for x in 0..8 {
        let column = kernel::<_, INVERSE>(core::array::from_fn(|y| work[y * 8 + x]));
        for y in 0..8 {
            out[y * 8 + x] = column[y] * 0.125;
        }
    }
}

// Implementations are private: the SSE2 implementation is reached only after
// feature detection, and NEON is baseline on AArch64. Every load/store receives
// exactly four floats, so neither alignment nor an extra readable row is needed.
#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64"))
))]
trait Lane4: Lane {
    fn load(input: &[f32; 4]) -> Self;
    fn store(self, out: &mut [f32; 4]);
    fn transpose(input: [Self; 4]) -> [Self; 4];
}

#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64"))
))]
#[inline(always)]
fn four_lane<T: Lane4, const INVERSE: bool>(input: &[f32; 64], out: &mut [f32; 64]) {
    let chunks = input.as_chunks::<4>().0;
    let rows: [[T; 8]; 2] = core::array::from_fn(|group| {
        let lo = T::transpose(core::array::from_fn(|y| {
            T::load(&chunks[group * 8 + y * 2])
        }));
        let hi = T::transpose(core::array::from_fn(|y| {
            T::load(&chunks[group * 8 + y * 2 + 1])
        }));
        kernel::<_, INVERSE>([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    });
    let chunks = out.as_chunks_mut::<4>().0;
    for x in [0, 4] {
        let top = T::transpose(core::array::from_fn(|i| rows[0][x + i]));
        let bottom = T::transpose(core::array::from_fn(|i| rows[1][x + i]));
        let column = kernel::<_, INVERSE>([
            top[0], top[1], top[2], top[3], bottom[0], bottom[1], bottom[2], bottom[3],
        ]);
        for (y, value) in column.into_iter().enumerate() {
            (value * 0.125).store(&mut chunks[y * 2 + x / 4]);
        }
    }
}

type Transform = fn(&[f32; 64], &mut [f32; 64]);

/// Forward and inverse transforms selected before any blocks are processed.
#[derive(Clone, Copy)]
pub(crate) struct Transforms {
    forward: Transform,
    inverse: Transform,
}

impl Transforms {
    pub(crate) fn new() -> Self {
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            Self {
                forward: neon::transform::<false>,
                inverse: neon::transform::<true>,
            }
        }
        #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
        {
            #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
            if std::is_x86_feature_detected!("sse2") {
                return Self {
                    // SAFETY: these private pointers are installed only after
                    // establishing SSE2 support, and retained for all calls.
                    forward: |input, out| unsafe { sse2::transform::<false>(input, out) },
                    inverse: |input, out| unsafe { sse2::transform::<true>(input, out) },
                };
            }
            Self {
                forward: scalar::<false>,
                inverse: scalar::<true>,
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn scalar() -> Self {
        Self {
            forward: scalar::<false>,
            inverse: scalar::<true>,
        }
    }

    #[inline]
    pub(crate) fn fdct(&self, input: &[f32; 64], out: &mut [f32; 64]) {
        (self.forward)(input, out);
    }

    #[inline]
    pub(crate) fn idct(&self, input: &[f32; 64], out: &mut [f32; 64]) {
        (self.inverse)(input, out);
    }
}

// Tests share a cached dispatch table; the encoder owns its production table.
#[cfg(test)]
fn standalone_transforms() -> Transforms {
    static TRANSFORMS: std::sync::OnceLock<Transforms> = std::sync::OnceLock::new();
    *TRANSFORMS.get_or_init(Transforms::new)
}

#[cfg(test)]
pub(crate) fn fdct(input: &[f32; 64], out: &mut [f32; 64]) {
    standalone_transforms().fdct(input, out);
}

#[cfg(test)]
pub(crate) fn idct(input: &[f32; 64], out: &mut [f32; 64]) {
    standalone_transforms().idct(input, out);
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    // Independent cosine definition, using f64 rather than either f32 network.
    fn reference64<const INVERSE: bool>(input: &[f32; 64]) -> [f64; 64] {
        let basis: [[f64; 8]; 8] = core::array::from_fn(|u| {
            core::array::from_fn(|x| {
                let scale = if u == 0 { 1.0 / 8.0f64.sqrt() } else { 0.5 };
                scale * (core::f64::consts::PI * ((2 * x + 1) * u) as f64 / 16.0).cos()
            })
        });
        let mut rows = [0.0; 64];
        let mut out = [0.0; 64];
        for y in 0..8 {
            for x in 0..8 {
                for k in 0..8 {
                    rows[y * 8 + x] += f64::from(input[y * 8 + k])
                        * if INVERSE { basis[k][x] } else { basis[x][k] };
                }
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                for k in 0..8 {
                    out[y * 8 + x] +=
                        rows[k * 8 + x] * if INVERSE { basis[k][y] } else { basis[y][k] };
                }
            }
        }
        out
    }

    fn check(input: &[f32; 64]) {
        let max = input.iter().fold(1.0f32, |a, &b| a.max(b.abs()));
        for inverse in [false, true] {
            let mut scalar_out = [0.0; 64];
            let mut selected_out = [0.0; 64];
            let exact = if inverse {
                scalar::<true>(input, &mut scalar_out);
                idct(input, &mut selected_out);
                reference64::<true>(input)
            } else {
                scalar::<false>(input, &mut scalar_out);
                fdct(input, &mut selected_out);
                reference64::<false>(input)
            };
            assert_eq!(scalar_out.map(f32::to_bits), selected_out.map(f32::to_bits));
            for i in 0..64 {
                let error = (f64::from(scalar_out[i]) - exact[i]).abs();
                assert!(
                    error < f64::from(max) * 4e-6,
                    "inverse={inverse} coefficient={i} error={error} max_input={max}"
                );
            }
        }
        let mut coefficients = [0.0; 64];
        let mut back = [0.0; 64];
        fdct(input, &mut coefficients);
        idct(&coefficients, &mut back);
        for i in 0..64 {
            assert!(
                (back[i] - input[i]).abs() < max * 4e-6,
                "round trip {i}: {} vs {} (max={max})",
                back[i],
                input[i]
            );
        }
    }

    #[test]
    fn butterflies_match_cosine_definition_and_round_trip() {
        for value in [-8192.0, -128.0, -0.25, 0.0, 0.25, 127.0, 8191.75] {
            check(&[value; 64]);
        }
        for position in 0..64 {
            let mut impulse = [0.0; 64];
            impulse[position] = if position % 2 == 0 { 128.0 } else { -128.0 };
            check(&impulse);
        }
        for bit in 0..6 {
            check(&core::array::from_fn(|i| {
                if (i >> bit) & 1 == 0 { -128.0 } else { 127.0 }
            }));
        }
        let mut seed = 0x12345678u32;
        for amplitude in [512, 32768] {
            for _ in 0..1024 {
                check(&core::array::from_fn(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 16) as i32 % (amplitude * 2) - amplitude) as f32 * 0.25
                }));
            }
        }
    }

    #[test]
    fn layout_and_dc_scale_match_legacy_dense_transforms() {
        let input = core::array::from_fn(|i| ((i * 37) % 256) as f32 - 128.0);
        type Transform = fn(&[f32; 64], &mut [f32; 64]);
        for (selected, dense) in [
            (fdct as Transform, reference::fdct as Transform),
            (idct, reference::idct),
        ] {
            let mut actual = [0.0; 64];
            let mut expected = [0.0; 64];
            selected(&input, &mut actual);
            dense(&input, &mut expected);
            for i in 0..64 {
                assert!((actual[i] - expected[i]).abs() < 0.001);
            }
        }
        let mut dc = [0.0; 64];
        fdct(&[127.0; 64], &mut dc);
        assert_eq!(dc[0], 8.0 * 127.0);
        assert!(dc[1..].iter().all(|&v| v == 0.0));
        let mut back = [0.0; 64];
        idct(&dc, &mut back);
        assert_eq!(back, [127.0; 64]);
    }

    #[test]
    fn unaligned_buffers_preserve_their_surroundings() {
        #[repr(C, align(16))]
        struct Padded {
            before: f32,
            data: [f32; 64],
            after: f32,
        }
        let input = Padded {
            before: 12345.0,
            data: core::array::from_fn(|i| (i * 73 % 256) as f32 - 128.0),
            after: -12345.0,
        };
        let mut out = Padded {
            before: 9876.0,
            data: [0.0; 64],
            after: -9876.0,
        };
        assert_eq!(input.data.as_ptr() as usize % 16, 4);
        assert_eq!(out.data.as_ptr() as usize % 16, 4);
        for inverse in [false, true] {
            let mut expected = [0.0; 64];
            if inverse {
                scalar::<true>(&input.data, &mut expected);
                idct(&input.data, &mut out.data);
            } else {
                scalar::<false>(&input.data, &mut expected);
                fdct(&input.data, &mut out.data);
            }
            assert_eq!(expected.map(f32::to_bits), out.data.map(f32::to_bits));
            assert_eq!((out.before, out.after), (9876.0, -9876.0));
            assert_eq!((input.before, input.after), (12345.0, -12345.0));
        }
    }
}
