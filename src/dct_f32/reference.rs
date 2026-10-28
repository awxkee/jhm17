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
//! Original dense f32 transforms, retained as a differential test oracle.

/// Orthonormal 8-point DCT-II basis, `basis[u][x]`
fn basis() -> [[f32; 8]; 8] {
    // cos(π (2x+1) u / 16), scaled so the transform is orthonormal.
    const C: [f32; 8] = [
        1.0,
        0.980_785_3,
        0.923_879_5,
        0.831_469_6,
        core::f32::consts::FRAC_1_SQRT_2,
        0.555_570_2,
        0.382_683_4,
        0.195_090_3,
    ];
    let mut b = [[0f32; 8]; 8];
    for (u, row) in b.iter_mut().enumerate() {
        for (x, v) in row.iter_mut().enumerate() {
            // cos(π (2x+1) u / 16) via the symmetry of the cosine table
            let k = ((2 * x + 1) * u) % 32;
            let c = match k {
                0..=8 => C[k],
                9..=16 => -C[16 - k],
                17..=24 => -C[k - 16],
                _ => C[32 - k],
            };
            *v = c * if u == 0 { 0.353_553_4 } else { 0.5 };
        }
    }
    b
}

/// Forward DCT in the JPEG scale (`F(0,0) = 8 × mean`)
fn fdct_inner(pixels: &[f32; 64], basis: &[[f32; 8]; 8], out: &mut [f32; 64]) {
    let mut rows = [0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut acc = 0.0;
            for x in 0..8 {
                acc += pixels[y * 8 + x] * basis[u][x];
            }
            rows[y * 8 + u] = acc;
        }
    }
    for v in 0..8 {
        for u in 0..8 {
            let mut acc = 0.0;
            for y in 0..8 {
                acc += rows[y * 8 + u] * basis[v][y];
            }
            out[v * 8 + u] = acc;
        }
    }
}

fn idct_inner(coefficients: &[f32; 64], basis: &[[f32; 8]; 8], out: &mut [f32; 64]) {
    let mut rows = [0f32; 64];
    for v in 0..8 {
        for x in 0..8 {
            let mut acc = 0.0;
            for u in 0..8 {
                acc += coefficients[v * 8 + u] * basis[u][x];
            }
            rows[v * 8 + x] = acc;
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            let mut acc = 0.0;
            for v in 0..8 {
                acc += rows[v * 8 + x] * basis[v][y];
            }
            out[y * 8 + x] = acc;
        }
    }
}

pub(crate) fn fdct(input: &[f32; 64], out: &mut [f32; 64]) {
    fdct_inner(input, &basis(), out);
}

pub(crate) fn idct(input: &[f32; 64], out: &mut [f32; 64]) {
    idct_inner(input, &basis(), out);
}
