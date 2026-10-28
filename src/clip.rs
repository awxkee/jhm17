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
//! Clip continuation for white pixels.
//!
//! The decoder clamps the reconstruction to 0..255, so error above white is
//! invisible, yet the quantizer pays for it. Before the DCT, blocks holding
//! white pixels are projected a few times: quantize, reconstruct, and let
//! every white pixel take the reconstruction where it overshoots, so the
//! coefficients only have to reproduce the visible part. Black pixels are
//! left alone: continuing them below black lost on every image with black
//! next to dark texture, while white next to ink wins large.

use crate::dct_f32::Transforms;
use crate::encoder::AlignedBlock;
use crate::quantization::QuantizationTable;

const WHITE: i16 = 127;

/// Smallest number of white pixels in a block for it to be processed
const MIN_WHITE_PIXELS: usize = 2;

/// Options for clip continuation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipOptions {
    /// Projection iterations (0 disables)
    pub iterations: u8,
}

impl Default for ClipOptions {
    fn default() -> Self {
        ClipOptions { iterations: 3 }
    }
}

/// Replace the white pixels of `block` (level-shifted samples) by targets the
/// quantizer can reach for free. Returns whether the block was changed.
pub(crate) fn continue_saturated(
    block: &mut AlignedBlock,
    table: &QuantizationTable,
    options: &ClipOptions,
    transforms: &Transforms,
) -> bool {
    let white = block.data.iter().filter(|&&p| p == WHITE).count();
    if options.iterations == 0 || white < MIN_WHITE_PIXELS || white == 64 {
        return false;
    }

    let source: [f32; 64] = core::array::from_fn(|i| block.data[i] as f32);
    let mut target = source;
    let mut coefficients = [0f32; 64];
    let mut recon = [0f32; 64];
    // Steps in the JPEG scale (the table carries them times 8).
    let steps: [f32; 64] = core::array::from_fn(|i| table.step(i) / 8.0);

    for _ in 0..options.iterations {
        transforms.fdct(&target, &mut coefficients);
        for (c, &q) in coefficients.iter_mut().zip(steps.iter()) {
            *c = (*c / q).round_() * q;
        }
        transforms.idct(&coefficients, &mut recon);
        for i in 0..64 {
            if block.data[i] == WHITE {
                target[i] = recon[i].max(source[i]);
            }
        }
    }

    let mut changed = false;
    for (&t, d) in target.iter().zip(block.data.iter_mut()) {
        let v = t.round_() as i16;
        if v != *d {
            changed = true;
        }
        *d = v;
    }
    changed
}

/// Continue exact white samples in a Q2 (four samples per pixel level) block.
///
/// The quantization table retains the ordinary DCT scale (eight times the
/// transmitted step); projection therefore runs in ordinary pixel units.
pub(crate) fn continue_saturated_precise(
    block: &mut AlignedBlock,
    table: &QuantizationTable,
    options: &ClipOptions,
    transforms: &Transforms,
) -> bool {
    const SCALE: f32 = 4.0;
    const PRECISE_WHITE: i16 = WHITE * 4;
    let white = block.data.iter().filter(|&&p| p == PRECISE_WHITE).count();
    if options.iterations == 0 || white < MIN_WHITE_PIXELS || white == 64 {
        return false;
    }

    let source: [f32; 64] = core::array::from_fn(|i| block.data[i] as f32 / SCALE);
    let mut target = source;
    let mut coefficients = [0f32; 64];
    let mut recon = [0f32; 64];
    let steps: [f32; 64] = core::array::from_fn(|i| table.step(i) / 8.0);

    for _ in 0..options.iterations {
        transforms.fdct(&target, &mut coefficients);
        for (c, &q) in coefficients.iter_mut().zip(steps.iter()) {
            *c = (*c / q).round_() * q;
        }
        transforms.idct(&coefficients, &mut recon);
        for i in 0..64 {
            if block.data[i] == PRECISE_WHITE {
                target[i] = recon[i].max(source[i]);
            }
        }
    }

    let mut changed = false;
    for (&t, d) in target.iter().zip(block.data.iter_mut()) {
        if *d == PRECISE_WHITE {
            // max/min also keep non-finite intermediates out of the conversion;
            // the float-to-integer cast saturates rather than wrapping.
            let scaled = (t * SCALE).max(PRECISE_WHITE as f32).min(i16::MAX as f32);
            let value = scaled.round_() as i16;
            changed |= value != *d;
            *d = value;
        }
    }
    changed
}

trait Round {
    fn round_(self) -> f32;
}

impl Round for f32 {
    #[inline]
    fn round_(self) -> f32 {
        let t = self + if self < 0.0 { -0.5 } else { 0.5 };
        t as i32 as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_f32::{fdct, idct};

    use crate::quantization::{QuantizationTableType, StepTuning};

    #[test]
    fn dct_round_trips() {
        let pixels: [f32; 64] = core::array::from_fn(|i| ((i * 37) % 23) as f32 - 11.0);
        let mut c = [0f32; 64];
        let mut back = [0f32; 64];
        fdct(&pixels, &mut c);
        let mean: f32 = pixels.iter().sum::<f32>() / 64.0;
        assert!(
            (c[0] - 8.0 * mean).abs() < 1e-3,
            "{} vs {}",
            c[0],
            8.0 * mean
        );
        idct(&c, &mut back);
        for i in 0..64 {
            assert!((back[i] - pixels[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn precise_continuation_keeps_fractional_nonwhite_samples() {
        let table = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            50,
            true,
            &StepTuning::PLAIN,
        );
        let mut block = AlignedBlock::new(core::array::from_fn(|i| {
            if i % 8 < 3 {
                -511 + (i % 3) as i16
            } else {
                WHITE * 4
            }
        }));
        let before = block;
        assert!(continue_saturated_precise(
            &mut block,
            &table,
            &ClipOptions::default(),
            &Transforms::new(),
        ));
        assert!(block.data.iter().any(|&v| v > WHITE * 4));
        for i in 0..64 {
            if before.data[i] == WHITE * 4 {
                assert!(block.data[i] >= WHITE * 4);
            } else {
                assert_eq!(block.data[i], before.data[i]);
            }
        }
    }

    #[test]
    fn precise_continuation_respects_white_and_iteration_gates() {
        let table = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            50,
            true,
            &StepTuning::PLAIN,
        );
        for (data, iterations) in [
            ([WHITE * 4; 64], 3),
            ([-511; 64], 3),
            (
                core::array::from_fn(|i| if i == 0 { WHITE * 4 } else { -3 }),
                3,
            ),
            (
                core::array::from_fn(|i| if i < 2 { WHITE * 4 } else { -3 }),
                0,
            ),
        ] {
            let mut block = AlignedBlock::new(data);
            assert!(!continue_saturated_precise(
                &mut block,
                &table,
                &ClipOptions { iterations },
                &Transforms::new(),
            ));
            assert_eq!(block.data, data);
        }
    }

    #[test]
    fn precise_continuation_handles_coarse_custom_tables_without_wrapping() {
        for step in [1, 255, 2048, u16::MAX] {
            let table = QuantizationTable::new_with_quality_tuned(
                &QuantizationTableType::Custom(alloc::boxed::Box::new([step; 64])),
                1,
                true,
                &StepTuning::PLAIN,
            );
            let mut block = AlignedBlock::new(core::array::from_fn(|i| {
                if i < 2 {
                    WHITE * 4
                } else if i % 2 == 0 {
                    i16::MIN
                } else {
                    i16::MAX
                }
            }));
            let before = block;
            continue_saturated_precise(
                &mut block,
                &table,
                &ClipOptions { iterations: 255 },
                &Transforms::new(),
            );
            for i in 0..64 {
                if before.data[i] == WHITE * 4 {
                    assert!(
                        block.data[i] >= WHITE * 4,
                        "step {step}: {:?}",
                        block.data[i]
                    );
                } else {
                    assert_eq!(block.data[i], before.data[i]);
                }
            }
        }
    }

    #[test]
    fn white_pixels_move_above_white_only() {
        let table = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            50,
            true,
            &StepTuning::PLAIN,
        );
        let mut block = AlignedBlock::default();
        // Left columns black ink on a white right side: a hard vertical edge.
        for y in 0..8 {
            for x in 0..8 {
                block.data[y * 8 + x] = if x < 3 { -128 } else { WHITE };
            }
        }
        let before = block;
        let changed = continue_saturated(
            &mut block,
            &table,
            &ClipOptions::default(),
            &Transforms::new(),
        );
        assert!(changed);
        for i in 0..64 {
            if before.data[i] == WHITE {
                assert!(
                    block.data[i] >= WHITE,
                    "white pixel {i} fell to {}",
                    block.data[i]
                );
            } else {
                assert_eq!(block.data[i], before.data[i], "non-white pixel {i} moved");
            }
        }
        let mut flat = AlignedBlock::new([WHITE; 64]);
        assert!(!continue_saturated(
            &mut flat,
            &table,
            &ClipOptions::default(),
            &Transforms::new(),
        ));
    }
}
