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
//! Rate-distortion optimized quantization.
//!
//! A block's AC levels are chosen by a dynamic program over the zigzag scan
//! whose state is the position of the last nonzero level. The rate is the real
//! JPEG structure: one run/size symbol plus `size` extra bits per nonzero,
//! a ZRL symbol per 16 skipped zeros and an EOB symbol when the block ends
//! early. Symbol prices estimate the final rate using Huffman code lengths fitted to
//! a plain first pass; final tables are refitted after RDO.
//! Distortion is squared error in quantizer units, mildly tilted toward the
//! pixel domain: the quantization table is the perceptual weighting.

use crate::encoder::AlignedBlock;
use crate::huffman::HuffmanTable;
use crate::quantization::QuantizationTable;
use crate::writer::ZIGZAG;
use alloc::vec::Vec;

/// Options for rate-distortion optimized quantization
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RdoOptions {
    /// Rate price per bit in quantizer-unit² of distortion.
    ///
    /// Larger values trade quality for bytes. `0.0` keeps the plain rounding quantizer.
    pub lambda: f32,

    /// Multiplier on the distortion of subsampled chroma components
    pub chroma_weight: f32,

    /// Multiplier on the distortion of full-resolution (4:4:4) chroma components
    pub chroma_weight_full: f32,

    /// Exponent of the per-position distortion weight `(step / mean step)^(2·gamma)`
    ///
    /// `0.0` measures error in quantizer units (the table is the perceptual
    /// weighting), `1.0` measures plain squared pixel error.
    pub distortion_power: f32,

    /// Blend toward the 4×4-tile perceptual weights.
    pub tile_weights: f32,

    /// Apply the tile weights to subsampled chroma components too
    pub tile_chroma: bool,

    /// Apply the tile weights to full-resolution (4:4:4) chroma components too.
    pub tile_chroma_full: bool,

    /// Adaptive quantization strength: each block's λ is scaled by
    /// `(activity / median activity)^(aq_strength / 2)`, clamped to
    /// `[aq_min, aq_max]`, so busy blocks drop more levels and flat blocks
    /// keep more. `0.0` disables it.
    pub aq_strength: f32,

    /// Lowest per-block λ scale (protection of flat blocks)
    pub aq_min: f32,

    /// Highest per-block λ scale (`1.0` = protect-only, never coarsen)
    pub aq_max: f32,

    /// Apply the per-block λ scale to chroma blocks too (from their own activity)
    pub aq_chroma: bool,

    /// Brightness tilt of the luma λ scale: `× (mean / 128)^b` from the block's
    /// DC. `0.0` disables it.
    pub aq_brightness: f32,

    /// Upper limit of the brightness factor (`1.0` = protect dark blocks only)
    pub aq_brightness_max: f32,

    /// Clip continuation of white luma pixels before the DCT
    pub clip: crate::clip::ClipOptions,
}

impl Default for RdoOptions {
    fn default() -> Self {
        RdoOptions {
            lambda: 0.05,
            chroma_weight: 3.0,
            chroma_weight_full: 1.0,
            distortion_power: 0.35,
            tile_weights: 0.5,
            tile_chroma: true,
            tile_chroma_full: false,
            aq_strength: 0.75,
            aq_min: 0.5,
            aq_max: 2.0,
            aq_chroma: true,
            aq_brightness: 0.5,
            aq_brightness_max: 8.0,
            clip: crate::clip::ClipOptions::default(),
        }
    }
}

/// Bit prices of the AC symbols of one Huffman table destination
pub(crate) struct AcPrices {
    symbol: [f32; 256],
}

impl AcPrices {
    /// Prices from symbol counts: code lengths of the table those counts would
    /// produce, with every unseen symbol counted once so it gets a long code
    /// instead of none.
    pub fn from_frequencies(freq: &[u32; 257]) -> AcPrices {
        let mut smoothed = *freq;
        for f in smoothed.iter_mut().take(256) {
            *f = (*f).max(1);
        }
        let table = HuffmanTable::new_optimized(smoothed);
        let mut symbol = [0f32; 256];
        for (s, price) in symbol.iter_mut().enumerate() {
            *price = table.get_for_value(s as u8).0 as f32;
        }
        AcPrices { symbol }
    }

    #[inline]
    fn eob(&self) -> f32 {
        self.symbol[0x00]
    }

    #[inline]
    fn zrl(&self) -> f32 {
        self.symbol[0xF0]
    }

    /// Bits of a nonzero level of `size` bits after `run` zeros
    #[inline]
    fn nonzero(&self, run: usize, size: u8) -> f32 {
        (run >> 4) as f32 * self.zrl()
            + self.symbol[((run & 15) << 4) | size as usize]
            + size as f32
    }
}

/// AC energy of a block of DCT coefficients
pub(crate) fn block_activity(coefficients: &AlignedBlock) -> f32 {
    coefficients.data[1..]
        .iter()
        .map(|&c| {
            let c = c as f32;
            c * c
        })
        .sum()
}

/// Lowest median activity AQ measures against: a pixel standard deviation of
/// one level (activity is `64² ·` variance).
///
/// Images that are mostly flat (graphics, white backgrounds) have a median
/// near zero, which would rank every smooth gradient block as busy and raise
/// its λ until the gradient collapses into DC-only blocks, i.e. banding.
/// Blocks below one level of deviation mask nothing, so their λ is never raised.
const AQ_MIN_MEDIAN_ACTIVITY: f32 = 4096.0;

/// Per-block λ scales from block activities, relative to their median:
/// `(activity / median)^(aq_strength / 2)`, clamped to `[aq_min, aq_max]`
pub(crate) fn aq_lambda_scales(activities: &[f32], options: &RdoOptions) -> Vec<f32> {
    let mut selected = activities.to_vec();
    let middle = selected.len() / 2;
    let median = if selected.is_empty() {
        1.0
    } else {
        (*selected
            .select_nth_unstable_by(middle, |a, b| a.partial_cmp(b).unwrap())
            .1)
            .max(AQ_MIN_MEDIAN_ACTIVITY)
    };
    let (lo, hi) = (options.aq_min.min(options.aq_max), options.aq_max);
    // Selection no longer needs its reordered values. Refill that allocation
    // in original block order instead of allocating a separate scales vector.
    for (scale, &a) in selected.iter_mut().zip(activities) {
        *scale = if a <= 0.0 {
            lo
        } else {
            pow_no_libm(a / median, options.aq_strength * 0.5).clamp(lo, hi)
        };
    }
    selected
}

/// Per-position distortion weights in zigzag order for one component
pub(crate) fn distortion_weights(
    table: &QuantizationTable,
    options: &RdoOptions,
    channel_weight: f32,
    tiled: bool,
) -> [f32; 64] {
    let mean = (1..64).map(|z| table.step(z)).sum::<f32>() / 63.0;
    let mut weights = [channel_weight; 64];
    if options.distortion_power != 0.0 {
        for (k, &z) in ZIGZAG.iter().enumerate().skip(1) {
            weights[k] = channel_weight
                * pow_no_libm(
                    table.step(z as usize) / mean,
                    2.0 * options.distortion_power,
                );
        }
    }
    if options.tile_weights != 0.0 && tiled {
        // Pixel SSE is step² times the quantizer-unit error.
        let mut tile = [0f32; 64];
        for (k, &z) in ZIGZAG.iter().enumerate().skip(1) {
            let step = table.step(z as usize) / mean;
            tile[k] = TILE4_WEIGHTS[z as usize] * step * step;
        }
        let scale = weights[1..].iter().sum::<f32>() / tile[1..].iter().sum::<f32>();
        let t = options.tile_weights;
        for k in 1..64 {
            weights[k] = pow_no_libm(weights[k], 1.0 - t) * pow_no_libm(tile[k] * scale, t);
        }
    }
    weights
}

/// Pixel-SSE weight of each 8×8 DCT position (natural order) under the 4×4-tile
/// weighted DCT error metric of hpvca (`cost::DISTORTION_WEIGHTS`: DC 1.8,
/// first AC 1.0, (1,1) 0.8, (0,2) 0.4, higher 0.2): `diag(Bᵀ Q B)` with `B`
/// the orthonormal 8-point DCT split into two 4-sample tiles.
#[rustfmt::skip]
#[allow(clippy::approx_constant)] // Empirical weights, not logarithm constants.
static TILE4_WEIGHTS: [f32; 64] = [
    1.8000, 1.6531, 1.0000, 0.9153, 0.4000, 0.4897, 0.2000, 0.3419,
    1.6531, 1.5255, 0.9606, 0.8663, 0.3642, 0.4571, 0.2000, 0.3290,
    1.0000, 0.9606, 0.8000, 0.6565, 0.2000, 0.3107, 0.2000, 0.2723,
    0.9153, 0.8663, 0.6565, 0.5634, 0.2202, 0.3010, 0.2000, 0.2613,
    0.4000, 0.3642, 0.2000, 0.2202, 0.2000, 0.2090, 0.2000, 0.2065,
    0.4897, 0.4571, 0.3107, 0.3010, 0.2090, 0.2322, 0.2000, 0.2192,
    0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000,
    0.3419, 0.3290, 0.2723, 0.2613, 0.2065, 0.2192, 0.2000, 0.2111,
];

/// `base.powf(exp)` for positive `base` without a math library
pub(crate) fn pow_no_libm(base: f32, exp: f32) -> f32 {
    // exp2(exp * log2(base)) via the IEEE representation plus short series.
    let bits = base.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // in [1, 2)
    // ln(m) = 2·atanh(s), s = (m − 1) / (m + 1) ≤ 1/3, four terms are within 1e-5
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let ln_m = 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (0.2 + s2 * (1.0 / 7.0))));
    let log2_m = ln_m * core::f32::consts::LOG2_E;
    let y = exp * (e as f32 + log2_m);
    let yi = y.floor_();
    let yf = y - yi;
    // 2^yf on [0, 1)
    let p = 1.0
        + yf * (core::f32::consts::LN_2
            + yf * (0.240_226_5 + yf * (0.055_504_1 + yf * (0.009_618_1 + yf * 0.001_333_3))));
    f32::from_bits(((yi as i32 + 127) as u32) << 23) * p
}

trait Floor {
    fn floor_(self) -> f32;
}

impl Floor for f32 {
    #[inline]
    fn floor_(self) -> f32 {
        let i = self as i32 as f32;
        if i > self { i - 1.0 } else { i }
    }
}

#[inline]
fn size_of(level: i32) -> u8 {
    (32 - level.leading_zeros()) as u8
}

/// Quantize one block of DCT coefficients (natural order, scaled like the
/// quantization table) into zigzag-ordered levels.
///
/// Candidates per position are the rounded level and one below it (never a
/// restore of a level that rounds to zero: anything above half a step already
/// rounds to a level, so there is nothing to bring back).
pub(crate) fn rdoq_block(
    coefficients: &AlignedBlock,
    table: &QuantizationTable,
    prices: &AcPrices,
    options: &RdoOptions,
    weights: &[f32; 64],
    lambda_scale: f32,
    final_ac: Option<&mut [u32; 257]>,
) -> AlignedBlock {
    let dc = table.quantize(coefficients.data[0], 0);
    if table.all_ac_round_to_zero(&coefficients.data) {
        if let Some(freq) = final_ac {
            freq[0] += 1;
        }
        return dc_only_block(dc);
    }
    rdoq_with_coefficients(
        |z| coefficients.data[z] as f32,
        dc,
        table,
        prices,
        options,
        weights,
        lambda_scale,
        final_ac,
    )
}

/// Quantize Q2 coefficients while keeping the writer's table in JPEG units.
pub(crate) fn rdoq_block_precise(
    coefficients: &[i32; 64],
    table: &QuantizationTable,
    prices: &AcPrices,
    options: &RdoOptions,
    weights: &[f32; 64],
    lambda_scale: f32,
    final_ac: Option<&mut [u32; 257]>,
) -> AlignedBlock {
    let dc = table.quantize_precise(coefficients[0], 0);
    if table.all_precise_ac_round_to_zero(coefficients) {
        if let Some(freq) = final_ac {
            freq[0] += 1;
        }
        return dc_only_block(dc);
    }
    rdoq_with_coefficients(
        |z| coefficients[z] as f32 * 0.25,
        dc,
        table,
        prices,
        options,
        weights,
        lambda_scale,
        final_ac,
    )
}

#[inline]
fn dc_only_block(dc: i16) -> AlignedBlock {
    let mut block = AlignedBlock::default();
    block.data[0] = dc;
    block
}

#[allow(clippy::too_many_arguments)]
fn rdoq_with_coefficients(
    coefficient: impl Fn(usize) -> f32,
    dc: i16,
    table: &QuantizationTable,
    prices: &AcPrices,
    options: &RdoOptions,
    weights: &[f32; 64],
    lambda_scale: f32,
    final_ac: Option<&mut [u32; 257]>,
) -> AlignedBlock {
    let lambda = options.lambda * lambda_scale;

    // Ideal (unrounded) levels in zigzag order; DC is coded by the plain quantizer.
    let mut ideal = [0f32; 64];
    for (k, &z) in ZIGZAG.iter().enumerate().skip(1) {
        ideal[k] = coefficient(z as usize) / table.step(z as usize);
    }

    // prefix[k] = weighted squared error of zeroing positions 1..=k
    let mut prefix = [0f32; 64];
    for k in 1..64 {
        prefix[k] = prefix[k - 1] + weights[k] * ideal[k] * ideal[k];
    }

    #[derive(Clone, Copy)]
    struct State {
        position: u8,
        prefix: f32,
        cost: f32,
    }
    let mut from = [0u8; 64];
    let mut level = [0i32; 64];
    // A completed predecessor never changes. Keep its prefix and cost beside
    // its position to avoid two indexed array lookups in every trellis edge.
    // State zero represents a path with no nonzero AC coefficients yet.
    let mut reachable = [State {
        position: 0,
        prefix: 0.0,
        cost: 0.0,
    }; 64];
    let mut reachable_len = 1;

    for k in 1..64 {
        let x = ideal[k];
        let ax = x.abs();
        let rounded = (ax + 0.5) as i32;
        if rounded == 0 {
            continue;
        }
        let candidates = [rounded, rounded - 1];
        // Within one size category the rate and future state are identical,
        // and rounding minimizes squared error. A lower level can help only
        // when it crosses a category boundary (2, 4, 8, ...).
        let count = if rounded >= 2 && (rounded as u32).is_power_of_two() {
            2
        } else {
            1
        };
        let mut best = f32::INFINITY;

        for &l in &candidates[..count] {
            let size = size_of(l);
            let d = ax - l as f32;
            let distortion = weights[k] * d * d;
            for predecessor in reachable[..reachable_len].iter().rev() {
                let tail = prefix[k - 1] - predecessor.prefix;
                if tail + distortion >= best {
                    // Every earlier predecessor zeroes at least this much; no cheaper path remains.
                    break;
                }
                let run = k - usize::from(predecessor.position) - 1;
                let cost =
                    predecessor.cost + tail + distortion + lambda * prices.nonzero(run, size);
                if cost < best {
                    best = cost;
                    from[k] = predecessor.position;
                    level[k] = if x < 0.0 { -l } else { l };
                }
            }
        }
        reachable[reachable_len] = State {
            position: k as u8,
            prefix: prefix[k],
            cost: best,
        };
        reachable_len += 1;
    }

    // Close the block: zeros after the last nonzero plus an EOB unless it sits at 63.
    let mut end = 0;
    let mut end_cost = f32::INFINITY;
    for state in &reachable[..reachable_len] {
        if state.cost == f32::INFINITY {
            continue;
        }
        let tail = prefix[63] - state.prefix;
        let eob = if state.position < 63 {
            lambda * prices.eob()
        } else {
            0.0
        };
        let cost = state.cost + tail + eob;
        if cost < end_cost {
            end_cost = cost;
            end = usize::from(state.position);
        }
    }

    let mut out = AlignedBlock::default();
    out.data[0] = dc;
    let mut k = end;
    if let Some(freq) = final_ac {
        // Count each selected nonzero's run after finding its predecessor.
        // Histogram additions commute, so traceback can visit them backwards.
        // Use emitted i16 values, including wrapped levels for extreme inputs.
        let mut next = 64;
        let mut next_size = 0;
        while k > 0 {
            let value = level[k] as i16;
            out.data[k] = value;
            if value != 0 {
                if next < 64 {
                    let run = next - k - 1;
                    freq[0xF0] += (run >> 4) as u32;
                    freq[((run & 15) << 4) | next_size] += 1;
                } else if k < 63 {
                    freq[0] += 1;
                }
                next = k;
                next_size = size_of(i32::from(value.unsigned_abs() & 0x7fff)) as usize;
            }
            k = from[k] as usize;
        }
        if next < 64 {
            let run = next - 1;
            freq[0xF0] += (run >> 4) as u32;
            freq[((run & 15) << 4) | next_size] += 1;
        } else {
            freq[0] += 1;
        }
        return out;
    }
    while k > 0 {
        out.data[k] = level[k] as i16;
        k = from[k] as usize;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aq_does_not_coarsen_gradients_on_flat_images() {
        // 60% pure white blocks put the median at zero; the rest are a smooth
        // ramp with half a level of deviation (activity 64² · 0.25).
        let activities: Vec<f32> = (0..100)
            .map(|i| if i < 60 { 0.0 } else { 1024.0 })
            .collect();
        let scales = aq_lambda_scales(&activities, &RdoOptions::default());
        assert!(
            scales[60..].iter().all(|&s| s <= 1.0),
            "{:?}",
            &scales[60..]
        );
    }
}
