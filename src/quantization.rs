use crate::encoder::AlignedBlock;
use alloc::boxed::Box;
use core::num::NonZeroU16;

/// # Quantization table used for encoding
///
/// Tables are based on tables from mozjpeg
#[derive(Debug, Clone)]
pub enum QuantizationTableType {
    /// Sample quantization tables given in Annex K (Clause K.1) of Recommendation ITU-T T.81 (1992) | ISO/IEC 10918-1:1994.
    Default,

    /// Flat
    Flat,

    /// Custom, tuned for MS-SSIM
    CustomMsSsim,

    /// Custom, tuned for PSNR-HVS
    CustomPsnrHvs,

    /// ImageMagick table by N. Robidoux
    ///
    /// From <http://www.imagemagick.org/discourse-server/viewtopic.php?f=22&t=20333&p=98008#p98008>
    ImageMagick,

    /// Relevance of human vision to JPEG-DCT compression (1992) Klein, Silverstein and Carney.
    KleinSilversteinCarney,

    /// DCTune perceptual optimization of compressed dental X-Rays (1997) Watson, Taylor, Borthwick
    DentalXRays,

    /// A visual detection model for DCT coefficient quantization (12/9/93) Ahumada, Watson, Peterson
    VisualDetectionModel,

    /// An improved detection model for DCT coefficient quantization (1993) Peterson, Ahumada and Watson
    ///
    /// The encoder's default: at matched rate it beats Annex K on both SSIMULACRA2 and
    /// butteraugli, plain and with rate-distortion optimization.
    ImprovedDetectionModel,

    /// A user supplied quantization table
    Custom(Box<[u16; 64]>),
}

impl QuantizationTableType {
    fn index(&self) -> usize {
        use QuantizationTableType::*;

        match self {
            Default => 0,
            Flat => 1,
            CustomMsSsim => 2,
            CustomPsnrHvs => 3,
            ImageMagick => 4,
            KleinSilversteinCarney => 5,
            DentalXRays => 6,
            VisualDetectionModel => 7,
            ImprovedDetectionModel => 8,
            Custom(_) => panic!("Custom types not supported"),
        }
    }
}

// Tables are based on mozjpeg jcparam.c
static DEFAULT_LUMA_TABLES: [[u16; 64]; 9] = [
    [
        // Annex K
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ],
    [
        // Flat
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    [
        // Custom, tuned for MS-SSIM
        12, 17, 20, 21, 30, 34, 56, 63, 18, 20, 20, 26, 28, 51, 61, 55, 19, 20, 21, 26, 33, 58, 69,
        55, 26, 26, 26, 30, 46, 87, 86, 66, 31, 33, 36, 40, 46, 96, 100, 73, 40, 35, 46, 62, 81,
        100, 111, 91, 46, 66, 76, 86, 102, 121, 120, 101, 68, 90, 90, 96, 113, 102, 105, 103,
    ],
    [
        // Custom, tuned for PSNR-HVS
        9, 10, 12, 14, 27, 32, 51, 62, 11, 12, 14, 19, 27, 44, 59, 73, 12, 14, 18, 25, 42, 59, 79,
        78, 17, 18, 25, 42, 61, 92, 87, 92, 23, 28, 42, 75, 79, 112, 112, 99, 40, 42, 59, 84, 88,
        124, 132, 111, 42, 64, 78, 95, 105, 126, 125, 99, 70, 75, 100, 102, 116, 100, 107, 98,
    ],
    [
        // ImageMagick table by N. Robidoux
        // From http://www.imagemagick.org/discourse-server/viewtopic.php?f=22&t=20333&p=98008#p98008
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ],
    [
        // Relevance of human vision to JPEG-DCT compression (1992) Klein, Silverstein and Carney.
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ],
    [
        // DCTune perceptual optimization of compressed dental X-Rays (1997) Watson, Taylor, Borthwick
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ],
    [
        // A visual detection model for DCT coefficient quantization (12/9/93) Ahumada, Watson, Peterson
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ],
    [
        // An improved detection model for DCT coefficient quantization (1993) Peterson, Ahumada and Watson
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ],
];

// Tables are based on mozjpeg jcparam.c
static DEFAULT_CHROMA_TABLES: [[u16; 64]; 9] = [
    [
        // Annex K
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99,
        99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ],
    [
        // Flat
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    [
        // Custom, tuned for MS-SSIM
        8, 12, 15, 15, 86, 96, 96, 98, 13, 13, 15, 26, 90, 96, 99, 98, 12, 15, 18, 96, 99, 99, 99,
        99, 17, 16, 90, 96, 99, 99, 99, 99, 96, 96, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ],
    [
        //Custom, tuned for PSNR-HVS
        9, 10, 17, 19, 62, 89, 91, 97, 12, 13, 18, 29, 84, 91, 88, 98, 14, 19, 29, 93, 95, 95, 98,
        97, 20, 26, 84, 88, 95, 95, 98, 94, 26, 86, 91, 93, 97, 99, 98, 99, 99, 100, 98, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 97, 97, 99, 99, 99, 99, 97, 99,
    ],
    [
        // ImageMagick table by N. Robidoux
        // From http://www.imagemagick.org/discourse-server/viewtopic.php?f=22&t=20333&p=98008#p98008
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ],
    [
        // Relevance of human vision to JPEG-DCT compression (1992) Klein, Silverstein and Carney.
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ],
    [
        // DCTune perceptual optimization of compressed dental X-Rays (1997) Watson, Taylor, Borthwick
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ],
    [
        // A visual detection model for DCT coefficient quantization (12/9/93) Ahumada, Watson, Peterson
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ],
    [
        // An improved detection model for DCT coefficient quantization (1993) Peterson, Ahumada and Watson
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ],
];

/// How a component's quantization steps follow the quality setting
///
/// The default halves the DC step relative to the table's own scaling: DC
/// error shows as blocking, which every quality metric punishes more than the
/// bytes a finer DC costs (SS2 −7 % plain, −5.5 % under the trellis, BA flat).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepTuning {
    /// Multiplier on the DC step (1.0 = the table's own scaling)
    pub dc: f32,
    /// Multiplier on every AC step (1.0 = the table's own scaling)
    pub ac: f32,
    /// Multipliers on the AC steps by radial band of the coefficient
    /// (u + v 1–2, 3–4, 5–6, 7–9, ≥ 10), on top of `ac`
    pub bands: [f32; 5],
}

impl Default for StepTuning {
    fn default() -> Self {
        StepTuning {
            dc: 0.5,
            ac: 1.0,
            bands: [1.0; 5],
        }
    }
}

impl StepTuning {
    /// The table's own scaling, as libjpeg applies it
    pub const PLAIN: StepTuning = StepTuning {
        dc: 1.0,
        ac: 1.0,
        bands: [1.0; 5],
    };

    /// Radial band of natural index `i` by u + v: 0 for 1–2, 1 for 3–4, 2 for 5–6, 3 for 7–9, 4 for ≥ 10
    #[inline]
    pub fn band(i: usize) -> usize {
        let r = (i & 7) + (i >> 3);
        match r {
            0..=2 => 0,
            3..=4 => 1,
            5..=6 => 2,
            7..=9 => 3,
            _ => 4,
        }
    }
}

/// Deadzone of the plain (non-RDO) quantizer
///
/// A coefficient whose magnitude is below `threshold × step` becomes zero;
/// larger ones round to nearest. Thresholds are
/// given per frequency quadrant of the 8×8 block: low, horizontal-high,
/// vertical-high, high-high.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Deadzone {
    /// Zero thresholds for luma, in steps, by quadrant
    pub luma: [f32; 4],
    /// Zero thresholds for subsampled chroma, in steps, by quadrant
    pub chroma: [f32; 4],
    /// Zero thresholds for full-resolution (4:4:4) chroma, in steps, by quadrant
    pub chroma_full: [f32; 4],
}

impl Default for Deadzone {
    /// Luma zero thresholds rising from 0.58 steps at low frequencies to 0.70
    /// at the highest, chroma plain: Kodak holdout
    /// −3.2 % SS2 / −5.3 % butteraugli at matched rate over plain rounding.
    fn default() -> Self {
        Deadzone {
            luma: [0.58, 0.635, 0.66, 0.70],
            chroma: [0.5; 4],
            chroma_full: [0.65; 4],
        }
    }
}

impl Deadzone {
    /// Plain rounding, no deadzone
    pub const PLAIN: Deadzone = Deadzone {
        luma: [0.5; 4],
        chroma: [0.5; 4],
        chroma_full: [0.5; 4],
    };

    /// Zero thresholds of one component, by chroma resolution
    #[inline]
    pub(crate) fn thresholds_for(&self, luma: bool, full_resolution: bool) -> &[f32; 4] {
        if luma {
            &self.luma
        } else if full_resolution {
            &self.chroma_full
        } else {
            &self.chroma
        }
    }

    #[inline]
    fn quadrant(index: usize) -> usize {
        let (u, v) = (index & 7, index >> 3);
        (u >= 4) as usize | (((v >= 4) as usize) << 1)
    }
}

/// The plain Q2 deadzone quantizer of one component, its step, zero threshold
/// and level floor laid out per natural-order position: DC has no threshold
/// (−∞) and no floor, AC levels at or above the threshold round to at least 1.
pub(crate) struct PlainDeadzone {
    steps: [f32; 64],
    thresholds: [f32; 64],
    floors: [i32; 64],
}

impl PlainDeadzone {
    /// Quantize natural-order Q2 coefficients into zigzag-ordered levels.
    ///
    /// The arithmetic of [QuantizationTable::quantize_precise_deadzone] with
    /// its branches as selects, so the 64 divisions run as vector divisions
    /// with the same IEEE results.
    #[inline]
    pub(crate) fn quantize_precise(&self, block: &[i32; 64]) -> AlignedBlock {
        let mut natural = [0i16; 64];
        for index in 0..64 {
            let value = block[index];
            let x = value.unsigned_abs() as f32 * 0.25 / self.steps[index];
            let rounded = ((x + 0.5) as i32).max(self.floors[index]);
            let level = if x < self.thresholds[index] {
                0
            } else {
                rounded
            };
            // level ≥ 0, so clamping −level only needs its lower bound
            let positive = level.min(i16::MAX as i32);
            let negative = (-level).max(i16::MIN as i32);
            natural[index] = (if value < 0 { negative } else { positive }) as i16;
        }
        let mut out = AlignedBlock::default();
        for (level, &z) in out.data.iter_mut().zip(crate::writer::ZIGZAG.iter()) {
            *level = natural[usize::from(z) & 63];
        }
        out
    }
}

impl QuantizationTable {
    /// Whether the RDO candidate search has no nonzero AC level to visit.
    /// Steps are positive multiples of eight, so integer half-step tests match
    /// `(abs(coefficient) / step + 0.5) as i32 == 0` exactly.
    #[inline]
    pub(crate) fn all_ac_round_to_zero(&self, coefficients: &[i16; 64]) -> bool {
        coefficients[1..]
            .iter()
            .zip(&self.table[1..])
            .all(|(&value, step)| value.unsigned_abs() < step.get() / 2)
    }

    #[inline]
    pub(crate) fn all_precise_ac_round_to_zero(&self, coefficients: &[i32; 64]) -> bool {
        coefficients[1..]
            .iter()
            .zip(&self.precise_divisors[1..])
            .all(|(&value, &divisor)| value.unsigned_abs() < divisor / 2)
    }

    /// Quantize a Q2 DCT coefficient against the original JPEG quantization
    /// step. The emitted DQT is independent of the internal sample precision.
    #[inline]
    pub(crate) fn quantize_precise(&self, value: i32, index: usize) -> i16 {
        let divisor = self.precise_divisors[index];
        let rounded = value.unsigned_abs() + divisor / 2;
        // ceil(2^32 / divisor) can overestimate the exact quotient by at most
        // one, since rounded < 2^32. A single correction makes division exact.
        let mut level =
            ((u64::from(rounded) * u64::from(self.precise_reciprocals[index])) >> 32) as u32;
        level -= (level * divisor > rounded) as u32;
        // divisor >= 4, so the quotient and its negation both fit in i32.
        let level = level as i32;
        if value < 0 {
            (-level).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        } else {
            level.min(i16::MAX as i32) as i16
        }
    }

    /// Per-position constants of the plain Q2 deadzone quantizer for one component
    pub(crate) fn plain_deadzone(&self, deadzone: &[f32; 4]) -> PlainDeadzone {
        let mut plain = PlainDeadzone {
            steps: [0.0; 64],
            thresholds: [f32::NEG_INFINITY; 64],
            floors: [0; 64],
        };
        for index in 0..64 {
            plain.steps[index] = self.step(index);
            if index > 0 {
                plain.thresholds[index] = deadzone[Deadzone::quadrant(index)];
                plain.floors[index] = 1;
            }
        }
        plain
    }

    /// The original per-coefficient plain Q2 quantizer
    #[cfg(test)]
    pub(crate) fn quantize_precise_deadzone(
        &self,
        value: i32,
        index: usize,
        deadzone: &[f32; 4],
    ) -> i16 {
        let x = value.unsigned_abs() as f32 * 0.25 / self.step(index);
        let level = if index > 0 && x < deadzone[Deadzone::quadrant(index)] {
            0
        } else if index > 0 {
            ((x + 0.5) as i32).max(1)
        } else {
            (x + 0.5) as i32
        };
        if value < 0 {
            (-level).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        } else {
            level.min(i16::MAX as i32) as i16
        }
    }

    /// Quantize one coefficient with a deadzone
    #[inline]
    pub fn quantize_deadzone(&self, value: i16, index: usize, deadzone: &[f32; 4]) -> i16 {
        let x = value.unsigned_abs() as f32 / self.step(index);
        let level = if index == 0 {
            (x + 0.5) as i32
        } else if x < deadzone[Deadzone::quadrant(index)] {
            0
        } else {
            ((x + 0.5) as i32).max(1)
        };
        if value < 0 {
            -level as i16
        } else {
            level as i16
        }
    }
}

const SHIFT: u32 = 2 * 8 - 1;

fn compute_reciprocal(divisor: u32) -> (i32, i32) {
    if divisor <= 1 {
        return (1, 0);
    }

    let mut reciprocals = (1 << SHIFT) / divisor;
    let fractional = (1 << SHIFT) % divisor;

    // Correction for rounding errors in division
    let mut correction = divisor / 2;

    if fractional != 0 {
        if fractional <= correction {
            correction += 1;
        } else {
            reciprocals += 1;
        }
    }

    (reciprocals as i32, correction as i32)
}

pub struct QuantizationTable {
    table: [NonZeroU16; 64],
    reciprocals: [i32; 64],
    corrections: [i32; 64],
    precise_divisors: [u32; 64],
    precise_reciprocals: [u32; 64],
}

impl QuantizationTable {
    fn from_table(table: [NonZeroU16; 64]) -> QuantizationTable {
        let mut reciprocals = [0i32; 64];
        let mut corrections = [0i32; 64];
        let mut precise_divisors = [0u32; 64];
        let mut precise_reciprocals = [0u32; 64];

        for i in 0..64 {
            let (reciprocal, correction) = compute_reciprocal(table[i].get() as u32);

            reciprocals[i] = reciprocal;
            corrections[i] = correction;
            let divisor = u32::from(table[i].get()) * 4;
            precise_divisors[i] = divisor;
            precise_reciprocals[i] = (1u64 << 32).div_ceil(u64::from(divisor)) as u32;
        }

        QuantizationTable {
            table,
            reciprocals,
            corrections,
            precise_divisors,
            precise_reciprocals,
        }
    }

    fn get_user_table(table: &[u16; 64]) -> [NonZeroU16; 64] {
        let mut q_table = [NonZeroU16::new(1).unwrap(); 64];
        for (i, &v) in table.iter().enumerate() {
            q_table[i] = match NonZeroU16::new(v.clamp(1, 2 << 10) << 3) {
                Some(v) => v,
                None => panic!("Invalid quantization table value: {}", v),
            };
        }
        q_table
    }

    fn get_with_quality_tuned(
        table: &[u16; 64],
        quality: u8,
        tuning: &StepTuning,
    ) -> [NonZeroU16; 64] {
        let quality = quality.clamp(1, 100) as u32;

        let scale = if quality < 50 {
            5000 / quality
        } else {
            200 - quality * 2
        };

        let mut q_table = [NonZeroU16::new(1).unwrap(); 64];

        for (i, &v) in table.iter().enumerate() {
            let v = v as u32;

            let s = scale as f32 / 100.0;
            let v = if i == 0 {
                // DC: optionally finer than the AC scaling
                ((v as f32 * s * tuning.dc * 100.0 + 50.0) / 100.0) as u32
            } else {
                let gain = tuning.ac * tuning.bands[StepTuning::band(i)];
                if gain == 1.0 {
                    (v * scale + 50) / 100
                } else {
                    ((v as f32 * s * gain * 100.0 + 50.0) / 100.0) as u32
                }
            };

            let v = v.clamp(1, 255) as u16;

            // Table values are premultiplied with 8 because dct is scaled by 8
            q_table[i] = NonZeroU16::new(v << 3).unwrap();
        }
        q_table
    }

    /// Build a table for `quality` with its entries scaled by `tuning`
    pub fn new_with_quality_tuned(
        table: &QuantizationTableType,
        quality: u8,
        luma: bool,
        tuning: &StepTuning,
    ) -> QuantizationTable {
        let table = match table {
            QuantizationTableType::Custom(table) => Self::get_user_table(table),
            table => {
                let table = if luma {
                    &DEFAULT_LUMA_TABLES[table.index()]
                } else {
                    &DEFAULT_CHROMA_TABLES[table.index()]
                };
                Self::get_with_quality_tuned(table, quality, tuning)
            }
        };
        Self::from_table(table)
    }

    #[inline]
    pub fn get(&self, index: usize) -> u8 {
        (self.table[index].get() >> 3) as u8
    }

    /// Quantization step in the scale of the DCT output (table value times 8)
    #[inline]
    pub fn step(&self, index: usize) -> f32 {
        self.table[index].get() as f32
    }

    #[inline]
    pub fn quantize(&self, in_value: i16, index: usize) -> i16 {
        let value = in_value as i32;

        let reciprocal = self.reciprocals[index];
        let corrections = self.corrections[index];

        let abs_value = value.abs();

        let mut product = (abs_value + corrections) * reciprocal;
        product >>= SHIFT;

        if value != abs_value {
            product *= -1;
        }

        product as i16
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization::{Deadzone, QuantizationTable, QuantizationTableType, StepTuning};

    #[test]
    fn precise_rounding_keeps_substep_dc_and_custom_deadzones() {
        let table = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            100,
            true,
            &StepTuning::PLAIN,
        );
        assert_eq!(table.step(0), 8.0);
        assert_eq!(table.quantize_precise(15, 0), 0);
        assert_eq!(table.quantize_precise(16, 0), 1);
        assert_eq!(table.quantize_precise(-16, 0), -1);
        assert_eq!(table.quantize_precise(12, 1), 0);
        assert_eq!(table.quantize_precise_deadzone(12, 1, &[0.3; 4]), 1);
        assert_eq!(table.quantize_precise_deadzone(-12, 1, &[0.3; 4]), -1);
        assert_eq!(table.quantize_precise_deadzone(12, 0, &[0.3; 4]), 0);
    }

    #[test]
    fn vectorized_plain_deadzone_matches_per_coefficient_quantizer() {
        let mut tables = alloc::vec::Vec::new();
        for kind in [
            QuantizationTableType::Default,
            QuantizationTableType::CustomMsSsim,
            QuantizationTableType::Custom(alloc::boxed::Box::new(core::array::from_fn(|i| {
                [1, 255, 2, 254, 17][i % 5]
            }))),
        ] {
            for quality in [1, 25, 50, 75, 90, 95, 100] {
                for luma in [true, false] {
                    tables.push(QuantizationTable::new_with_quality_tuned(
                        &kind,
                        quality,
                        luma,
                        &StepTuning::PLAIN,
                    ));
                }
            }
        }
        let deadzones = [
            Deadzone::default().luma,
            Deadzone::default().chroma_full,
            Deadzone::PLAIN.luma,
            [0.0, 0.25, 1.0, 3.5],
            [f32::INFINITY, 0.5, 0.75, 0.62],
        ];
        let mut seed = 0x9e37_79b9u32;
        let mut blocks: alloc::vec::Vec<[i32; 64]> = alloc::vec![
            [0; 64],
            [i32::MAX; 64],
            [i32::MIN; 64],
            [-1; 64],
            core::array::from_fn(|i| if i % 2 == 0 {
                i32::MIN + 1
            } else {
                i32::MAX - 1
            }),
        ];
        for case in 0..2048 {
            blocks.push(core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let r = seed as i32;
                match case % 4 {
                    // ten-bit Q2 range, half-step boundaries near small steps
                    0 => r % 65_536,
                    1 => r % 2048,
                    2 => r >> (seed % 31),
                    _ => r,
                }
            }));
        }
        // Exact half-step and threshold boundaries of every position
        for table in &tables[..4] {
            for offset in [-1, 0, 1] {
                blocks.push(core::array::from_fn(|z| {
                    (table.step(z) as i32) * 2 + offset
                }));
                blocks.push(core::array::from_fn(|z| {
                    -((table.step(z) as i32) * 2 + offset)
                }));
            }
        }
        for table in &tables {
            for deadzone in &deadzones {
                let plain = table.plain_deadzone(deadzone);
                for block in &blocks {
                    let actual = plain.quantize_precise(block);
                    let natural: [i16; 64] = core::array::from_fn(|z| {
                        table.quantize_precise_deadzone(block[z], z, deadzone)
                    });
                    for (k, &z) in crate::writer::ZIGZAG.iter().enumerate() {
                        assert_eq!(
                            actual.data[k],
                            natural[usize::from(z)],
                            "position {z}, value {}, deadzone {deadzone:?}",
                            block[usize::from(z)]
                        );
                    }
                }
            }
        }
    }

    fn precise_division_reference(value: i32, step: u16) -> i16 {
        let divisor = u32::from(step) * 4;
        let magnitude = (value.unsigned_abs() + divisor / 2) / divisor;
        let signed = if value < 0 {
            -i64::from(magnitude)
        } else {
            i64::from(magnitude)
        };
        signed.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
    }

    #[test]
    fn precise_reciprocal_matches_division_for_all_jpeg_steps_and_q2_samples() {
        for step in 1u16..=255 {
            let internal_step = step * 8;
            let table = QuantizationTable::from_table(
                [core::num::NonZeroU16::new(internal_step).unwrap(); 64],
            );
            for value in -32768..=32768 {
                let expected = precise_division_reference(value, internal_step);
                assert_eq!(table.quantize_precise(value, 0), expected);
                // Also preserve the previous float decision throughout this range.
                let x = value.unsigned_abs() as f32 * 0.25 / f32::from(internal_step);
                let level = (x + 0.5) as i32;
                let old = if value < 0 { -level } else { level } as i16;
                assert_eq!(expected, old);
            }
        }
    }

    #[test]
    fn precise_reciprocal_handles_full_i32_and_custom_step_extremes() {
        let mut seed = 0x98765432u32;
        for step in [
            1u16,
            2,
            3,
            7,
            8,
            9,
            31,
            255,
            256,
            257,
            1023,
            2048,
            8191,
            8192,
            16384,
            32767,
            u16::MAX,
        ] {
            let table =
                QuantizationTable::from_table([core::num::NonZeroU16::new(step).unwrap(); 64]);
            for value in [
                i32::MIN,
                i32::MIN + 1,
                -32769,
                -32768,
                -1,
                0,
                1,
                32768,
                32769,
                i32::MAX - 1,
                i32::MAX,
            ] {
                assert_eq!(
                    table.quantize_precise(value, 0),
                    precise_division_reference(value, step)
                );
            }
            for _ in 0..16384 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let value = seed as i32;
                assert_eq!(
                    table.quantize_precise(value, 0),
                    precise_division_reference(value, step)
                );
            }
            let divisor = i64::from(step) * 4;
            for level in [0i64, 1, 2, 3, 127, 255, 1023, 8191, 32767, 65535] {
                let half = level * divisor + divisor / 2;
                for delta in -1..=1 {
                    if let Ok(value) = i32::try_from(half + delta) {
                        for signed in [value, -value] {
                            assert_eq!(
                                table.quantize_precise(signed, 0),
                                precise_division_reference(signed, step),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deadzone_zeroes_below_threshold_and_rounds_above() {
        let q = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            50,
            true,
            &StepTuning::PLAIN,
        );
        let dz = Deadzone::default();
        let step = q.step(63);
        // Highest frequency: threshold 0.70 steps.
        assert_eq!(q.quantize_deadzone((0.65 * step) as i16, 63, &dz.luma), 0);
        assert_eq!(q.quantize_deadzone((0.75 * step) as i16, 63, &dz.luma), 1);
        assert_eq!(q.quantize_deadzone(-(1.6 * step) as i16, 63, &dz.luma), -2);
        // Plain rounding reproduces the reciprocal quantizer.
        for v in [-300i16, -17, 0, 5, 40, 777] {
            assert_eq!(
                q.quantize_deadzone(v, 9, &Deadzone::PLAIN.luma),
                q.quantize(v, 9)
            );
        }
        // DC never has a deadzone.
        assert_eq!(
            q.quantize_deadzone((0.6 * q.step(0)) as i16, 0, &dz.luma),
            1
        );
    }

    #[test]
    fn test_new_100() {
        let q = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            100,
            true,
            &StepTuning::PLAIN,
        );

        for &v in &q.table {
            let v = v.get();
            assert_eq!(v, 1 << 3);
        }

        let q = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            100,
            false,
            &StepTuning::PLAIN,
        );

        for &v in &q.table {
            let v = v.get();
            assert_eq!(v, 1 << 3);
        }
    }

    #[test]
    fn test_new_100_quantize() {
        let q = QuantizationTable::new_with_quality_tuned(
            &QuantizationTableType::Default,
            100,
            true,
            &StepTuning::PLAIN,
        );

        for i in -255..255 {
            assert_eq!(i, q.quantize(i << 3, 0));
        }
    }
}
