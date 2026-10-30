//! Decoder-feedback coefficient polish (study, ported from jixel's `coef_polish`)
//!
//! After the trellis, greedy ±1 level moves per block are priced exactly under
//! frozen optimal Huffman code lengths and judged on the exact libjpeg-turbo
//! reconstruction (islow IDCT, fancy h2v2 upsampling, `ycc_rgb_convert`) by an
//! incremental SSIMULACRA 2 and/or ColorVideoVDP. A move is kept when its
//! metric change beats the bits it costs at the image's own exchange rate,
//! taken from two encodes at neighbouring quality settings.
//!
//! With `trellis_k`, each block first gets whole-block alternatives: the K
//! cheapest level vectors of a trellis over a λ sweep, judged the same way.
//! They reach joint changes (a dropped tail, a moved EOB) that no single ±1
//! move improves on its own.

use crate::dct_f32::Transforms;
use crate::encoder::AlignedBlock;
use crate::huffman::HuffmanTable;
use crate::writer::ZIGZAG;
use alloc::vec;
use alloc::vec::Vec;

/// Which metric(s) must accept a move
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolishJudge {
    Ssimulacra2,
    Cvvdp,
    /// SSIMULACRA 2 ranks, ColorVideoVDP checks the best few
    Both,
}

/// Decoder-feedback coefficient polish after quantization (cargo feature
/// `polish`, RGB input with 4:4:4 or 4:2:0 sampling).
///
/// Moves are judged on the decoded image by SSIMULACRA 2 and ColorVideoVDP
/// against their bit cost. Every mode is far slower than plain encoding.
///
/// When not [`Off`](Polish::Off), prefer the trellis:
/// [`TrellisGreedy`](Polish::TrellisGreedy) gives the smallest files, and
/// polish works best on top of the trellis quantizer ([`RdoOptions`], on by
/// default), whose levels it refines.
///
/// [`RdoOptions`]: crate::RdoOptions
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Polish {
    /// No polish
    #[default]
    Off,
    /// Greedy ±1 moves on single coefficients
    Greedy,
    /// Whole-block alternatives from a K-best trellis over a λ sweep
    Trellis,
    /// Trellis alternatives, then greedy moves: the preferred mode, with the
    /// smallest files at about twice the time of [`Polish::Greedy`]
    TrellisGreedy,
    /// Explicit options
    Custom(PolishOptions),
}

impl Polish {
    /// The options this mode runs with, `None` when off
    pub fn options(self) -> Option<PolishOptions> {
        let greedy = PolishOptions::default();
        match self {
            Polish::Off => None,
            Polish::Greedy => Some(greedy),
            Polish::Trellis => Some(PolishOptions {
                trellis_k: 2,
                rounds: 0,
                ..greedy
            }),
            Polish::TrellisGreedy => Some(PolishOptions {
                trellis_k: 2,
                ..greedy
            }),
            Polish::Custom(options) => Some(options),
        }
    }
}

/// Coefficient polish configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolishOptions {
    pub judge: PolishJudge,
    /// Candidates verified per block and round
    pub proposals: usize,
    /// Greedy rounds per block
    pub rounds: usize,
    /// Candidates costing more bits than this are not evaluated
    pub max_bits: i32,
    /// Smallest |unquantized level| a zeroed coefficient is restored from
    pub restore_threshold: f32,
    /// Last zigzag position that is moved
    pub scan_limit: usize,
    /// SSIMULACRA 2 accepted candidates ColorVideoVDP checks (judge `Both`)
    pub cvvdp_tries: usize,
    /// Quality distance of the two exchange-rate encodes
    pub quality_step: u8,
    /// Paths the K-best trellis keeps per λ for whole-block candidates,
    /// judged before the greedy rounds (`0` disables them)
    pub trellis_k: usize,
    /// Trellis candidates verified per block
    pub trellis_proposals: usize,
}

impl Default for PolishOptions {
    fn default() -> Self {
        PolishOptions {
            judge: PolishJudge::Both,
            proposals: 4,
            rounds: 2,
            max_bits: 2,
            restore_threshold: 0.25,
            scan_limit: 63,
            cvvdp_tries: 2,
            quality_step: 3,
            trellis_k: 0,
            trellis_proposals: 6,
        }
    }
}

/// Counters of one polish pass
#[derive(Clone, Copy, Debug, Default)]
pub struct PolishStats {
    pub evaluated: usize,
    pub accepted: usize,
    pub bits_delta: i64,
    pub ss2_before: f64,
    pub ss2_after: f64,
    pub jod_before: f64,
    pub jod_after: f64,
    pub ss2_slope: f64,
    pub jod_slope: f64,
    /// Evaluated and accepted moves by kind: restore, toward zero, away from
    /// zero, trellis alternative
    pub kind_evaluated: [usize; 4],
    pub kind_accepted: [usize; 4],
    /// Accepted moves per greedy round
    pub round_accepted: [usize; 4],
}

/// One component's block grid as stored by the encoder
#[derive(Clone, Copy)]
pub(crate) struct PlaneSpec {
    pub(crate) padded_cols: usize,
    pub(crate) padded_rows: usize,
    /// 1 for full resolution, 2 for h2v2 chroma
    pub(crate) sub: usize,
    pub(crate) ac_table: usize,
    pub(crate) dc_table: usize,
}

/// libjpeg-turbo's `ycc_rgb_convert` tables
struct ColorTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

impl ColorTables {
    fn new() -> Self {
        const HALF: i32 = 1 << 15;
        let fix = |x: f64| (x * 65536.0 + 0.5) as i32;
        let mut t = ColorTables {
            cr_r: [0; 256],
            cb_b: [0; 256],
            cr_g: [0; 256],
            cb_g: [0; 256],
        };
        for i in 0..256 {
            let x = i as i32 - 128;
            t.cr_r[i] = (fix(1.40200) * x + HALF) >> 16;
            t.cb_b[i] = (fix(1.77200) * x + HALF) >> 16;
            t.cr_g[i] = -fix(0.71414) * x;
            t.cb_g[i] = -fix(0.34414) * x + HALF;
        }
        t
    }

    #[inline]
    fn rgb(&self, y: i32, cb: u8, cr: u8) -> [u8; 3] {
        // Widening u8 indices proves all four 256-entry lookups in bounds.
        let (cb, cr) = (usize::from(cb), usize::from(cr));
        [
            (y + self.cr_r[cr]).clamp(0, 255) as u8,
            (y + ((self.cb_g[cb] + self.cr_g[cr]) >> 16)).clamp(0, 255) as u8,
            (y + self.cb_b[cb]).clamp(0, 255) as u8,
        ]
    }
}

/// Fancy h2v2 upsampling of one component row. Slice the two input rows
/// once, then produce the even/odd output pairs without pixel indexing.
fn h2v2_pairs(
    plane: &[u8],
    stride: usize,
    cw: usize,
    ch: usize,
    y: usize,
    x0: usize,
) -> impl Iterator<Item = [u8; 2]> + Clone {
    let cy = y / 2;
    let near = if y & 1 == 0 {
        cy.saturating_sub(1)
    } else {
        (cy + 1).min(ch - 1)
    };
    let top = &plane[cy * stride..][..cw];
    let other = &plane[near * stride..][..cw];
    let start = x0 / 2;
    let previous_col = start.saturating_sub(1);
    let first = 3 * i32::from(top[previous_col]) + i32::from(other[previous_col]);
    let sums = top[start..]
        .iter()
        .zip(&other[start..])
        .map(|(&a, &b)| 3 * i32::from(a) + i32::from(b));
    let last = sums.clone().next_back().expect("nonempty component row");
    let previous = core::iter::once(first).chain(sums.clone());
    let next = sums.clone().skip(1).chain(core::iter::once(last));
    sums.zip(previous)
        .zip(next)
        .map(|((this, previous), next)| {
            // colsum is in 0..=1020, so both rounded convex combinations
            // lie in 0..=255. Boundary sums repeat the first/last sample.
            [
                ((3 * this + previous + 8) >> 4) as u8,
                ((3 * this + next + 7) >> 4) as u8,
            ]
        })
}

#[inline]
fn convert_row(
    colors: &ColorTables,
    out: &mut [[u8; 3]],
    luma: &[u8],
    cb: impl Iterator<Item = u8>,
    cr: impl Iterator<Item = u8>,
) {
    for (((out, &y), cb), cr) in out.iter_mut().zip(luma).zip(cb).zip(cr) {
        *out = colors.rgb(i32::from(y), cb, cr);
    }
}

/// Work in output pairs so the upsamplers and RGB stores use fixed-size
/// arrays throughout. A cropped rectangle can start/end on a half pair.
#[inline]
fn convert_h2v2_row(
    colors: &ColorTables,
    mut out: &mut [[u8; 3]],
    mut luma: &[u8],
    mut cb: impl Iterator<Item = [u8; 2]>,
    mut cr: impl Iterator<Item = [u8; 2]>,
    odd: bool,
) {
    if odd
        && let (Some((first, rest)), Some((&y, ys))) = (out.split_first_mut(), luma.split_first())
    {
        *first = colors.rgb(i32::from(y), cb.next().unwrap()[1], cr.next().unwrap()[1]);
        out = rest;
        luma = ys;
    }
    let (pairs, tail) = out.as_chunks_mut::<2>();
    let (ys, ytail) = luma.as_chunks::<2>();
    for (((out, y), cb), cr) in pairs.iter_mut().zip(ys).zip(cb.by_ref()).zip(cr.by_ref()) {
        out[0] = colors.rgb(i32::from(y[0]), cb[0], cr[0]);
        out[1] = colors.rgb(i32::from(y[1]), cb[1], cr[1]);
    }
    if let ([out], &[y]) = (tail, ytail) {
        *out = colors.rgb(i32::from(y), cb.next().unwrap()[0], cr.next().unwrap()[0]);
    }
}

/// Decoded component planes and the RGB frame they convert to
struct Frame {
    width: usize,
    height: usize,
    specs: Vec<PlaneSpec>,
    /// Component samples over the padded grid
    planes: Vec<Vec<u8>>,
    rgb: Vec<u8>,
    colors: ColorTables,
    /// libjpeg's islow IDCT, the fastest version this CPU runs
    islow: crate::idct::Islow,
}

impl Frame {
    fn new(
        width: usize,
        height: usize,
        specs: &[PlaneSpec],
        blocks: &[Vec<AlignedBlock>],
        tables: &[[u16; 64]],
    ) -> Self {
        let mut frame = Frame {
            width,
            height,
            specs: specs.to_vec(),
            planes: specs
                .iter()
                .map(|s| vec![0u8; s.padded_cols * s.padded_rows * 64])
                .collect(),
            rgb: vec![0u8; width * height * 3],
            colors: ColorTables::new(),
            islow: crate::idct::islow(),
        };
        for (c, spec) in specs.iter().enumerate() {
            for (b, block) in blocks[c][..spec.padded_cols * spec.padded_rows]
                .iter()
                .enumerate()
            {
                frame.decode_block(c, b, &block.data, &tables[c]);
            }
        }
        frame.convert(0, 0, width, height);
        frame
    }

    fn stride(&self, c: usize) -> usize {
        self.specs[c].padded_cols * 8
    }

    fn decode_block(&mut self, c: usize, b: usize, levels: &[i16; 64], table: &[u16; 64]) {
        let mut out = [0u8; 64];
        (self.islow)(levels, table, &mut out);
        let cols = self.specs[c].padded_cols;
        let start = (b / cols) * 8 * cols + b % cols;
        let groups = self.planes[c].as_chunks_mut::<8>().0;
        for (row, samples) in groups[start..]
            .iter_mut()
            .step_by(cols)
            .zip(out.as_chunks::<8>().0)
        {
            *row = *samples;
        }
    }

    fn block_samples(&self, c: usize, b: usize) -> [u8; 64] {
        let cols = self.specs[c].padded_cols;
        let start = (b / cols) * 8 * cols + b % cols;
        let mut out = [0u8; 64];
        let groups = self.planes[c].as_chunks::<8>().0;
        for (samples, row) in out
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(groups[start..].iter().step_by(cols))
        {
            *samples = *row;
        }
        out
    }

    fn set_block_samples(&mut self, c: usize, b: usize, samples: &[u8; 64]) {
        let cols = self.specs[c].padded_cols;
        let start = (b / cols) * 8 * cols + b % cols;
        let groups = self.planes[c].as_chunks_mut::<8>().0;
        for (row, samples) in groups[start..]
            .iter_mut()
            .step_by(cols)
            .zip(samples.as_chunks::<8>().0)
        {
            *row = *samples;
        }
    }

    /// Recompute RGB inside `[x0, x1) × [y0, y1)`. Row slices check the
    /// rectangle once per row; zipped byte/pixel iterators bound inner loads.
    fn convert(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        if x0 == x1 || y0 == y1 {
            return;
        }
        let ys = self.stride(0);
        let planes = &self.planes;
        let gray = self.specs.len() == 1;
        let full = gray || self.specs[1].sub == 1;
        let cs = if gray { 0 } else { self.stride(1) };
        let (cw, ch) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let rows = self
            .rgb
            .chunks_exact_mut(self.width * 3)
            .zip(planes[0].chunks_exact(ys))
            .enumerate();
        for (y, (out, luma)) in rows.take(y1).skip(y0) {
            let out = out[x0 * 3..x1 * 3].as_chunks_mut::<3>().0;
            let luma = &luma[x0..x1];
            if gray {
                for (out, &y) in out.iter_mut().zip(luma) {
                    *out = [y; 3];
                }
            } else if full {
                let cb = &planes[1][y * cs..][x0..x1];
                let cr = &planes[2][y * cs..][x0..x1];
                convert_row(
                    &self.colors,
                    out,
                    luma,
                    cb.iter().copied(),
                    cr.iter().copied(),
                );
            } else {
                let cb = h2v2_pairs(&planes[1], cs, cw, ch, y, x0);
                let cr = h2v2_pairs(&planes[2], cs, cw, ch, y, x0);
                convert_h2v2_row(&self.colors, out, luma, cb, cr, x0 & 1 != 0);
            }
        }
    }

    /// Pixel rectangle a block of component `c` influences, clipped to the image
    fn influence(&self, c: usize, b: usize) -> [usize; 4] {
        let spec = self.specs[c];
        let (bx, by) = (b % spec.padded_cols, b / spec.padded_cols);
        let s = spec.sub;
        let margin = s - 1;
        let x0 = (bx * 8 * s).saturating_sub(margin);
        let y0 = (by * 8 * s).saturating_sub(margin);
        let x1 = ((bx + 1) * 8 * s + margin).min(self.width);
        let y1 = ((by + 1) * 8 * s + margin).min(self.height);
        [x0, y0, x1.max(x0), y1.max(y0)]
    }

    fn save_rect(&self, r: [usize; 4], out: &mut Vec<u8>) {
        out.clear();
        for y in r[1]..r[3] {
            let at = (y * self.width + r[0]) * 3;
            out.extend_from_slice(&self.rgb[at..at + (r[2] - r[0]) * 3]);
        }
    }

    fn load_rect(&mut self, r: [usize; 4], data: &[u8]) {
        let n = (r[2] - r[0]) * 3;
        for (i, y) in (r[1]..r[3]).enumerate() {
            let at = (y * self.width + r[0]) * 3;
            self.rgb[at..at + n].copy_from_slice(&data[i * n..i * n + n]);
        }
    }
}

/// Bits of a block's AC tokens under the given code lengths
fn ac_bits(levels: &[i16; 64], lengths: &[u8; 256]) -> i32 {
    let len = |s: usize| match lengths[s] {
        0 => 24,
        l => i32::from(l),
    };
    let Some(last) = (1..64).rev().find(|&k| levels[k] != 0) else {
        return len(0);
    };
    let mut bits = 0;
    let mut run = 0usize;
    for &level in &levels[1..=last] {
        if level == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            bits += len(0xF0);
            run -= 16;
        }
        let size = 16 - level.unsigned_abs().leading_zeros() as usize;
        bits += len((run << 4) | size) + size as i32;
        run = 0;
    }
    if last < 63 {
        bits += len(0);
    }
    bits
}

/// Local AC token costs for single-coefficient moves. Only the moved
/// coefficient and its next nonzero token can change their run lengths.
struct AcCosts<'a> {
    levels: &'a [i16; 64],
    lengths: &'a [u8; 256],
    previous: [usize; 64],
    next: [usize; 64],
}

impl<'a> AcCosts<'a> {
    fn new(levels: &'a [i16; 64], lengths: &'a [u8; 256]) -> Self {
        let mut previous = [0; 64];
        let mut next = [64; 64];
        let mut p = 0;
        for k in 1..64 {
            previous[k] = p;
            if levels[k] != 0 {
                p = k;
            }
        }
        let mut n = 64;
        for k in (1..64).rev() {
            next[k] = n;
            if levels[k] != 0 {
                n = k;
            }
        }
        Self {
            levels,
            lengths,
            previous,
            next,
        }
    }

    fn length(&self, symbol: usize) -> i32 {
        match self.lengths[symbol] {
            0 => 24,
            l => i32::from(l),
        }
    }

    fn token(&self, previous: usize, k: usize, level: i16) -> i32 {
        let run = k - previous - 1;
        let size = 16 - level.unsigned_abs().leading_zeros() as usize;
        (run / 16) as i32 * self.length(0xf0) + self.length(((run & 15) << 4) | size) + size as i32
    }

    fn suffix(&self, previous: usize, next: usize) -> i32 {
        if next < 64 {
            self.token(previous, next, self.levels[next])
        } else if previous < 63 {
            self.length(0)
        } else {
            0
        }
    }

    fn delta(&self, k: usize, new: i16) -> i32 {
        let (p, n) = (self.previous[k], self.next[k]);
        let old = self.levels[k];
        if old == new {
            0
        } else if old == 0 {
            self.token(p, k, new) + self.suffix(k, n) - self.suffix(p, n)
        } else if new == 0 {
            self.suffix(p, n) - self.token(p, k, old) - self.suffix(k, n)
        } else {
            self.token(p, k, new) - self.token(p, k, old)
        }
    }
}

fn dc_size(diff: i32) -> usize {
    32 - diff.unsigned_abs().leading_zeros() as usize
}

fn ac_symbols(levels: &[i16; 64], freq: &mut [u32; 257]) {
    let Some(last) = (1..64).rev().find(|&k| levels[k] != 0) else {
        freq[0] += 1;
        return;
    };
    let mut run = 0usize;
    for &level in &levels[1..=last] {
        if level == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            freq[0xF0] += 1;
            run -= 16;
        }
        let size = 16 - level.unsigned_abs().leading_zeros() as usize;
        freq[(run << 4) | size] += 1;
        run = 0;
    }
    if last < 63 {
        freq[0] += 1;
    }
}

fn code_lengths(freq: [u32; 257]) -> [u8; 256] {
    let table = HuffmanTable::new_optimized(freq);
    core::array::from_fn(|s| {
        if freq[s] == 0 {
            0
        } else {
            table.get_for_value(s as u8).0
        }
    })
}

/// Polisher of one image: its RGB source and prepared metric references
pub(crate) struct Polisher<'a> {
    width: usize,
    height: usize,
    specs: Vec<PlaneSpec>,
    source: &'a [[u8; 3]],
    transforms: Transforms,
    dense_basis: Option<[[f32; 8]; 8]>,
    ss2: Option<ssimu2_lite::PreparedReference>,
    cvvdp: Option<cvvdp_lite::PreparedReference>,
    options: PolishOptions,
}

// A zero restore threshold considers numerical cancellation residues and
// their signs. Preserve the previous cosine arithmetic for this setting;
// the normal path shares the encoder's butterfly transform.
fn dense_basis() -> [[f32; 8]; 8] {
    let mut basis = [[0f32; 8]; 8];
    for (u, row) in basis.iter_mut().enumerate() {
        let cu = if u == 0 {
            core::f32::consts::FRAC_1_SQRT_2
        } else {
            1.0
        };
        for (x, v) in row.iter_mut().enumerate() {
            *v = 0.5 * cu * ((2 * x + 1) as f32 * u as f32 * core::f32::consts::PI / 16.0).cos();
        }
    }
    basis
}

/// Forward DCT in JPEG's dequantized scale (level × DQT entry).
fn dense_fdct(samples: &[f32; 64], basis: &[[f32; 8]; 8]) -> [f32; 64] {
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            tmp[y * 8 + u] = (0..8).map(|x| basis[u][x] * samples[y * 8 + x]).sum();
        }
    }
    let mut out = [0f32; 64];
    for v in 0..8 {
        for u in 0..8 {
            out[v * 8 + u] = (0..8).map(|y| basis[v][y] * tmp[y * 8 + u]).sum();
        }
    }
    out
}

/// Consecutive h2v2 input pairs, repeating the last component sample at
/// the right edge. An odd last source pixel pairs with itself; an even-width
/// row repeats its last *pair*, matching clamping in component coordinates.
fn pixel_pairs(row: &[[u8; 3]], start: usize) -> impl Iterator<Item = [[u8; 3]; 2]> + Clone {
    let (pairs, tail) = row.as_chunks::<2>();
    let last = if let Some(&pixel) = tail.first() {
        [pixel; 2]
    } else {
        *pairs.last().expect("nonempty source row")
    };
    pairs[start..]
        .iter()
        .copied()
        .chain(core::iter::repeat(last))
}

#[inline]
fn source_component<const C: usize>([r, g, b]: [u8; 3]) -> f32 {
    let (r, g, b) = (f32::from(r), f32::from(g), f32::from(b));
    match C {
        0 => 0.299 * r + 0.587 * g + 0.114 * b,
        1 => -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0,
        _ => 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0,
    }
}

impl<'a> Polisher<'a> {
    pub(crate) fn new(
        source: &'a [u8],
        width: usize,
        height: usize,
        specs: Vec<PlaneSpec>,
        options: PolishOptions,
    ) -> Option<Self> {
        let image = ssimu2_lite::RgbImage::srgb8(source, width, height, width * 3).ok()?;
        let ss2 = ssimu2_lite::PreparedReference::new(image).ok()?;
        let metric = cvvdp_lite::CvvdpLite::new(cvvdp_lite::Display::macbook_pro_16()).ok()?;
        let cvvdp = metric
            .prepare_reference(cvvdp_lite::RgbImage::srgb8(source, width, height, width * 3).ok()?)
            .ok()?;

        Some(Polisher {
            width,
            height,
            specs,
            source: source[..width * height * 3].as_chunks::<3>().0,
            transforms: Transforms::new(),
            dense_basis: (options.restore_threshold <= 0.0).then(dense_basis),
            ss2: Some(ss2),
            cvvdp: Some(cvvdp),
            options,
        })
    }

    /// Form one source block without per-channel indexing. Row iterators and
    /// fixed-size pixel/pair arrays bound all loads; only row and start slices
    /// are checked. Edge replication preserves the original summation order.
    fn source_block<const C: usize>(&self, b: usize) -> [f32; 64] {
        let spec = self.specs[C];
        let s = spec.sub;
        let (cw, ch) = (self.width.div_ceil(s), self.height.div_ceil(s));
        let (bx, by) = (b % spec.padded_cols, b / spec.padded_cols);
        let start = (bx * 8).min(cw - 1);
        let mut samples = [0.0; 64];
        for (y, out) in samples.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let sy = (by * 8 + y).min(ch - 1) * s;
            let mut rows = self.source.chunks_exact(self.width).skip(sy);
            let top = rows.next().expect("source row inside image");
            match s {
                1 => {
                    let pixels = top[start..]
                        .iter()
                        .chain(core::iter::repeat(top.last().unwrap()));
                    for (out, &pixel) in out.iter_mut().zip(pixels) {
                        *out = source_component::<C>(pixel) - 128.0;
                    }
                }
                2 => {
                    let bottom = rows.next().unwrap_or(top);
                    for ((out, top), bottom) in out
                        .iter_mut()
                        .zip(pixel_pairs(top, start))
                        .zip(pixel_pairs(bottom, start))
                    {
                        let mut sum = 0.0;
                        // Same order as dy, then dx: TL, TR, BL, BR.
                        for pixel in top.into_iter().chain(bottom) {
                            sum += source_component::<C>(pixel);
                        }
                        *out = sum / 4.0 - 128.0;
                    }
                }
                _ => unreachable!("polish supports full resolution and h2v2"),
            }
        }
        samples
    }

    /// The shared butterfly DCT already returns natural order, normalized to
    /// JPEG dequantized units (DC = 8 * mean). No zigzag or scale conversion.
    fn ideal_block(&self, c: usize, b: usize) -> [f32; 64] {
        let samples = match c {
            0 => self.source_block::<0>(b),
            1 => self.source_block::<1>(b),
            2 => self.source_block::<2>(b),
            _ => unreachable!("RGB components only"),
        };
        if let Some(basis) = &self.dense_basis {
            dense_fdct(&samples, basis)
        } else {
            let mut out = [0.0; 64];
            self.transforms.fdct(&samples, &mut out);
            out
        }
    }

    fn rgb_ss2<'b>(&self, rgb: &'b [u8]) -> ssimu2_lite::RgbImage<'b> {
        ssimu2_lite::RgbImage::srgb8(rgb, self.width, self.height, self.width * 3).unwrap()
    }

    fn rgb_cv<'b>(&self, rgb: &'b [u8]) -> cvvdp_lite::RgbImage<'b> {
        cvvdp_lite::RgbImage::srgb8(rgb, self.width, self.height, self.width * 3).unwrap()
    }

    fn visible(&self, c: usize, b: usize) -> bool {
        let spec = self.specs[c];
        let (bx, by) = (b % spec.padded_cols, b / spec.padded_cols);
        bx * 8 < self.width.div_ceil(spec.sub) && by * 8 < self.height.div_ceil(spec.sub)
    }

    /// Optimal Huffman code lengths of the visible blocks: (DC, AC) per table
    fn lengths(&self, blocks: &[Vec<AlignedBlock>]) -> ([[u8; 256]; 2], [[u8; 256]; 2], u64) {
        let mut dc = [[0u32; 257]; 2];
        let mut ac = [[0u32; 257]; 2];
        for f in dc.iter_mut().chain(&mut ac) {
            f[256] = 1;
        }
        for (c, (spec, blocks)) in self.specs.iter().zip(blocks).enumerate() {
            let dc_freq = &mut dc[spec.dc_table];
            let ac_freq = &mut ac[spec.ac_table];
            let mut prev = 0i32;
            for (b, block) in blocks.iter().enumerate() {
                if !self.visible(c, b) {
                    continue;
                }
                let d = i32::from(block.data[0]);
                dc_freq[dc_size(d - prev)] += 1;
                prev = d;
                ac_symbols(&block.data, ac_freq);
            }
        }
        let dl = dc.map(code_lengths);
        let al = ac.map(code_lengths);
        // Total bits under those lengths
        let mut bits = 0u64;
        for (c, (spec, blocks)) in self.specs.iter().zip(blocks).enumerate() {
            let dc_len = &dl[spec.dc_table];
            let ac_len = &al[spec.ac_table];
            let mut prev = 0i32;
            for (b, block) in blocks.iter().enumerate() {
                if !self.visible(c, b) {
                    continue;
                }
                let d = i32::from(block.data[0]);
                let size = dc_size(d - prev);
                bits += u64::from(dc_len[size]) + size as u64;
                prev = d;
                bits += ac_bits(&block.data, ac_len) as u64;
            }
        }
        (dl, al, bits)
    }

    /// Estimated bits, SSIMULACRA 2 and JOD of an encode
    pub(crate) fn measure(
        &self,
        blocks: &[Vec<AlignedBlock>],
        tables: &[[u16; 64]],
    ) -> (u64, f64, f64) {
        let frame = Frame::new(self.width, self.height, &self.specs, blocks, tables);
        let (_, _, bits) = self.lengths(blocks);
        let ss2 = self
            .ss2
            .as_ref()
            .unwrap()
            .compare(self.rgb_ss2(&frame.rgb))
            .unwrap_or(f64::NAN);
        let jod = self
            .cvvdp
            .as_ref()
            .unwrap()
            .evaluator()
            .compare(self.rgb_cv(&frame.rgb))
            .map_or(f64::NAN, |s| s.jod);
        (bits, ss2, jod)
    }

    /// Judge `candidates` for block `b` of component `c`. The chosen one is left
    /// applied to the frame and to the metric states; the caller commits its levels.
    #[allow(clippy::too_many_arguments)]
    fn judge(
        &self,
        state: &mut JudgeState,
        stats: &mut PolishStats,
        c: usize,
        b: usize,
        table: &[u16; 64],
        candidates: &mut [Candidate],
    ) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        let o = self.options;
        let rect = state.frame.influence(c, b);
        let rect_ss2 = ssimu2_lite::Rect::new(rect[0], rect[1], rect[2], rect[3]);
        let rect_cv = cvvdp_lite::Rect::new(rect[0], rect[1], rect[2], rect[3]);
        let old_samples = state.frame.block_samples(c, b);
        state.frame.save_rect(rect, &mut state.saved);
        // Apply a candidate to the frame (coefficients stay untouched)
        let apply = |frame: &mut Frame, cand: &Candidate| {
            frame.decode_block(c, b, &cand.levels, table);
            frame.convert(rect[0], rect[1], rect[2], rect[3]);
        };
        let undo = |frame: &mut Frame, saved: &[u8]| {
            frame.set_block_samples(c, b, &old_samples);
            frame.load_rect(rect, saved);
        };
        let (slopes, total_bits) = (state.slopes, state.total_bits);
        let rate = |slope: f64, bits: i32| slope * f64::from(bits) / total_bits;

        let accepted = &mut state.accepted;
        accepted.clear();
        if let Some(ss2) = state.ss2.as_mut() {
            let last = candidates.len().saturating_sub(1);
            // End on the best proxy so its metric state can usually be committed
            // directly. Each trial still starts from the identical baseline.
            for (i, cand) in candidates.iter_mut().enumerate().rev() {
                apply(&mut state.frame, cand);
                // All trials start from the same state and touch the same
                // rectangle, so one baseline checkpoint serves the whole list.
                if i == last {
                    ss2.update_checkpointed(
                        self.rgb_ss2(&state.frame.rgb),
                        rect_ss2,
                        &mut state.ss2_cp,
                    )
                    .unwrap();
                } else {
                    ss2.update(self.rgb_ss2(&state.frame.rgb), rect_ss2)
                        .unwrap();
                }
                cand.gain = ss2.score() - state.cur_ss2 - rate(slopes.0, cand.bits);
                if i != 0 {
                    ss2.rollback(&state.ss2_cp);
                }
                undo(&mut state.frame, &state.saved);
                stats.evaluated += 1;
                stats.kind_evaluated[cand.kind as usize] += 1;
            }
            // Preserve the original stable order for equal metric gains.
            rank_candidates(candidates, accepted);
        } else {
            accepted.extend(0..candidates.len());
        }

        // The chosen candidate leaves its pixels in the frame and, when
        // ColorVideoVDP judged last, its state in `cv`.
        let mut chosen = None;
        if let Some(cv) = state.cv.as_mut() {
            let both = o.judge == PolishJudge::Both;
            let tries = if both { o.cvvdp_tries } else { accepted.len() };
            let mut best = 0.0;
            for (trial, &i) in accepted.iter().take(tries).enumerate() {
                // SAFETY: rank_candidates rebuilds accepted from this slice's
                // indices; the other branch uses 0..candidates.len(). Neither
                // the slice length nor those indices changes during these trials.
                let cand = unsafe { candidates.get_unchecked(i) };
                apply(&mut state.frame, cand);
                let s = if trial == 0 {
                    cv.update_checkpointed(
                        self.cvvdp.as_ref().unwrap(),
                        self.rgb_cv(&state.frame.rgb),
                        rect_cv,
                        &mut state.cv_cp,
                    )
                } else {
                    cv.update(
                        self.cvvdp.as_ref().unwrap(),
                        self.rgb_cv(&state.frame.rgb),
                        rect_cv,
                    )
                }
                .unwrap();
                let jod = s.jod;
                let gain = jod - state.cur_jod - rate(slopes.1, cand.bits);
                if !both {
                    stats.evaluated += 1;
                    stats.kind_evaluated[cand.kind as usize] += 1;
                }
                if both && gain > 0.0 {
                    chosen = Some(i);
                    state.cur_jod = jod;
                    break;
                }
                cv.rollback(&state.cv_cp);
                undo(&mut state.frame, &state.saved);
                if gain > best {
                    best = gain;
                    chosen = Some(i);
                }
            }
            if !both && let Some(i) = chosen {
                apply(&mut state.frame, &candidates[i]);
                state.cur_jod = cv
                    .update(
                        self.cvvdp.as_ref().unwrap(),
                        self.rgb_cv(&state.frame.rgb),
                        rect_cv,
                    )
                    .unwrap()
                    .jod;
            }
        } else if let Some(&i) = accepted.first() {
            chosen = Some(i);
            apply(&mut state.frame, &candidates[i]);
        }

        if let Some(ss2) = state.ss2.as_mut() {
            if chosen != Some(0) {
                ss2.rollback(&state.ss2_cp);
                if chosen.is_some() {
                    ss2.update(self.rgb_ss2(&state.frame.rgb), rect_ss2)
                        .unwrap();
                }
            }
            state.cur_ss2 = ss2.score();
        }
        chosen
    }

    /// Polish `blocks` in place. `slopes` are metric points per unit of ln(bits).
    pub(crate) fn polish(
        mut self,
        blocks: &mut [Vec<AlignedBlock>],
        tables: &[[u16; 64]],
        slopes: (f64, f64),
    ) -> PolishStats {
        let o = self.options;
        // Both references price the calibration encodes. Once their slopes are
        // known, release the reference that is unused by a single-metric judge.
        if o.judge == PolishJudge::Cvvdp {
            self.ss2 = None;
        } else if o.judge == PolishJudge::Ssimulacra2 {
            self.cvvdp = None;
        }
        let frame = Frame::new(self.width, self.height, &self.specs, blocks, tables);
        let (_, ac_len, total_bits) = self.lengths(blocks);
        let use_ss2 = o.judge != PolishJudge::Cvvdp;
        let use_cv = o.judge != PolishJudge::Ssimulacra2;

        // States of the metrics that do not judge stay unbuilt
        let ss2 = use_ss2.then(|| {
            self.ss2
                .as_ref()
                .unwrap()
                .incremental(self.rgb_ss2(&frame.rgb))
                .expect("ssimulacra2 state")
        });
        let cv = use_cv.then(|| {
            cvvdp_lite::IncrementalState::new(self.cvvdp.as_ref().unwrap(), self.rgb_cv(&frame.rgb))
                .expect("cvvdp state")
        });
        let mut state = JudgeState {
            cur_ss2: ss2.as_ref().map_or(f64::NAN, |s| s.score()),
            cur_jod: cv.as_ref().map_or(f64::NAN, |s| s.score().jod),
            frame,
            ss2,
            cv,
            ss2_cp: ssimu2_lite::Checkpoint::default(),
            cv_cp: cvvdp_lite::Checkpoint::default(),
            saved: Vec::new(),
            accepted: Vec::new(),
            slopes,
            total_bits: total_bits.max(1) as f64,
        };
        let mut stats = PolishStats {
            ss2_before: state.cur_ss2,
            jod_before: state.cur_jod,
            ss2_slope: slopes.0,
            jod_slope: slopes.1,
            ..PolishStats::default()
        };
        let mut paths = Vec::new();
        let mut candidates: Vec<Candidate> = Vec::new();

        for c in 0..self.specs.len() {
            let spec = self.specs[c];
            let table = &tables[c];
            let lengths = &ac_len[spec.ac_table];
            for (b, stored) in blocks[c][..spec.padded_cols * spec.padded_rows]
                .iter_mut()
                .enumerate()
            {
                if !self.visible(c, b) {
                    continue;
                }
                let ideal = self.ideal_block(c, b);
                // Unquantized levels in zigzag order
                let x: [f32; 64] = core::array::from_fn(|k| {
                    let z = usize::from(ZIGZAG[k]);
                    if k == 0 {
                        0.0
                    } else {
                        ideal[z] / f32::from(table[z])
                    }
                });
                let err = |levels: &[i16; 64]| -> f32 {
                    (1..64)
                        .map(|k| (x[k] - f32::from(levels[k])) * (x[k] - f32::from(levels[k])))
                        .sum()
                };

                // Whole-block alternatives from the K-best trellis over a λ sweep
                if o.trellis_k > 0 {
                    let block = *stored;
                    let base_bits = ac_bits(&block.data, lengths);
                    let base_err = err(&block.data);
                    candidates.clear();
                    for &lambda in &TRELLIS_LAMBDAS {
                        kbest_levels(
                            &x,
                            lengths,
                            lambda,
                            o.trellis_k,
                            o.restore_threshold,
                            &mut paths,
                        );
                        for mut levels in paths.drain(..) {
                            levels[0] = block.data[0];
                            if levels == block.data || candidates.iter().any(|c| c.levels == levels)
                            {
                                continue;
                            }
                            let bits = ac_bits(&levels, lengths) - base_bits;
                            candidates.push(Candidate {
                                kind: MoveKind::Trellis,
                                levels,
                                bits,
                                proxy: err(&levels) - base_err + 0.1 * bits as f32,
                                gain: 0.0,
                            });
                        }
                    }
                    candidates.sort_by(|a, b| a.proxy.total_cmp(&b.proxy));
                    candidates.truncate(o.trellis_proposals);
                    if let Some(i) =
                        self.judge(&mut state, &mut stats, c, b, table, &mut candidates)
                    {
                        let cand = &candidates[i];
                        stored.data = cand.levels;
                        stats.accepted += 1;
                        stats.kind_accepted[cand.kind as usize] += 1;
                        stats.bits_delta += i64::from(cand.bits);
                    }
                }

                // Greedy ±1 moves
                for round in 0..o.rounds {
                    let block = *stored;
                    let costs = AcCosts::new(&block.data, lengths);
                    candidates.clear();
                    for k in 1..=o.scan_limit.min(63) {
                        let level = block.data[k];
                        let mut moves = [None, None];
                        if level == 0 {
                            if x[k].abs() >= o.restore_threshold {
                                moves[0] =
                                    Some((MoveKind::Restore, if x[k] > 0.0 { 1 } else { -1 }));
                            }
                        } else {
                            moves[0] = Some((MoveKind::TowardZero, level - level.signum()));
                            moves[1] = Some((MoveKind::AwayFromZero, level + level.signum()));
                        }
                        for (kind, new) in moves.into_iter().flatten() {
                            let bits = costs.delta(k, new);
                            if bits > o.max_bits {
                                continue;
                            }
                            let e = |l: i16| (x[k] - f32::from(l)) * (x[k] - f32::from(l));
                            let proxy = e(new) - e(level) + 0.1 * bits as f32;
                            // Keep exactly the stable sort's first `proposals`
                            // entries, preserving generation order on ties.
                            let at =
                                candidates.partition_point(|c| !c.proxy.total_cmp(&proxy).is_gt());
                            if at >= o.proposals {
                                continue;
                            }
                            let mut levels = block.data;
                            levels[k] = new;
                            candidates.insert(
                                at,
                                Candidate {
                                    kind,
                                    levels,
                                    bits,
                                    proxy,
                                    gain: 0.0,
                                },
                            );
                            if candidates.len() > o.proposals {
                                candidates.pop();
                            }
                        }
                    }
                    if candidates.is_empty() {
                        break;
                    }
                    let Some(i) = self.judge(&mut state, &mut stats, c, b, table, &mut candidates)
                    else {
                        break;
                    };
                    let cand = &candidates[i];
                    stored.data = cand.levels;
                    stats.accepted += 1;
                    stats.kind_accepted[cand.kind as usize] += 1;
                    stats.round_accepted[round.min(3)] += 1;
                    stats.bits_delta += i64::from(cand.bits);
                }
            }
        }
        stats.ss2_after = state.ss2.as_ref().map_or(f64::NAN, |s| s.score());
        stats.jod_after = state.cv.as_ref().map_or(f64::NAN, |s| s.score().jod);
        stats
    }
}

/// The discriminant is the public statistics array's index. A valid enum
/// value proves the index is below four without a runtime check.
#[derive(Clone, Copy)]
#[repr(u8)]
enum MoveKind {
    Restore = 0,
    TowardZero = 1,
    AwayFromZero = 2,
    Trellis = 3,
}

/// A replacement for one block's levels
struct Candidate {
    /// Restore, toward zero, away from zero, trellis alternative
    kind: MoveKind,
    levels: [i16; 64],
    bits: i32,
    proxy: f32,
    gain: f64,
}

/// Rebuild the complete index list here so its provenance cannot depend on
/// the previous block. The candidate slice has a fixed length throughout.
fn rank_candidates(candidates: &[Candidate], accepted: &mut Vec<usize>) {
    accepted.clear();
    accepted.extend(
        candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (c.gain > 0.0).then_some(i)),
    );
    accepted.sort_by(|&a, &b| {
        // SAFETY: every index came from enumerating this candidate slice.
        // Sorting only permutes indices and cannot resize the candidate slice.
        unsafe {
            candidates
                .get_unchecked(b)
                .gain
                .total_cmp(&candidates.get_unchecked(a).gain)
        }
    });
}

/// The decoded frame and metric states candidates are judged against
struct JudgeState {
    frame: Frame,
    ss2: Option<ssimu2_lite::IncrementalState>,
    cv: Option<cvvdp_lite::IncrementalState>,
    cur_ss2: f64,
    cur_jod: f64,
    ss2_cp: ssimu2_lite::Checkpoint,
    cv_cp: cvvdp_lite::Checkpoint,
    saved: Vec<u8>,
    accepted: Vec<usize>,
    slopes: (f64, f64),
    total_bits: f64,
}

/// λ sweep of the trellis candidates, in quantizer-unit² per bit
const TRELLIS_LAMBDAS: [f32; 6] = [0.015, 0.03, 0.06, 0.12, 0.25, 0.5];

const KBEST_MAX: usize = 4;

#[derive(Clone, Copy)]
struct Path {
    cost: f32,
    /// Previous nonzero position and its rank in that position's list
    from: u8,
    rank: u8,
    level: i16,
}

/// Insert into a cost-sorted list of at most `k` paths
fn insert_path(list: &mut [Path; KBEST_MAX], count: &mut usize, k: usize, path: Path) {
    let i = if *count < k {
        let i = *count;
        *count += 1;
        i
    } else if path.cost < list[k - 1].cost {
        k - 1
    } else {
        return;
    };
    let (previous, tail) = list.split_at_mut(i);
    let mut slot = &mut tail[0];
    for previous in previous.iter_mut().rev() {
        if previous.cost.partial_cmp(&path.cost) != Some(core::cmp::Ordering::Greater) {
            break;
        }
        *slot = *previous;
        slot = previous;
    }
    *slot = path;
}

/// The `k` cheapest AC level vectors (zigzag order, DC left zero) of a block
/// with unquantized levels `x` under Σ (x − level)² + λ·bits, symbols priced
/// by `lengths`.
///
/// The trellis of `rdo.rs` with its state (last nonzero position) keeping
/// its `k` best paths instead of one. Each position may take the rounded
/// level, one below, one above, or 1 restored from `restore` up.
fn kbest_levels(
    x: &[f32; 64],
    lengths: &[u8; 256],
    lambda: f32,
    k: usize,
    restore: f32,
    out: &mut Vec<[i16; 64]>,
) {
    let k = k.clamp(1, KBEST_MAX);
    let len = |s: usize| match lengths[s] {
        0 => 24.0,
        l => f32::from(l),
    };
    let mut prefix = [0f32; 64];
    for i in 1..64 {
        prefix[i] = prefix[i - 1] + x[i] * x[i];
    }
    let none = Path {
        cost: f32::INFINITY,
        from: 0,
        rank: 0,
        level: 0,
    };
    let mut lists = [[none; KBEST_MAX]; 64];
    let mut counts = [0usize; 64];
    lists[0][0].cost = 0.0;
    counts[0] = 1;
    let mut states = [0usize; 64];
    let mut n_states = 1;

    for pos in 1..64 {
        let ax = x[pos].abs();
        let rounded = (ax + 0.5) as i32;
        let (lo, hi) = match rounded {
            0 if ax >= restore => (1, 1),
            0 => continue,
            r => ((r - 1).max(1), r + 1),
        };
        // states starts with 0 and appends each completed position at most
        // once. At this point every state p is strictly below pos < 64.
        let (prior_lists, current_lists) = lists.split_at_mut(pos);
        let current = &mut current_lists[0];
        let (prior_counts, current_counts) = counts.split_at_mut(pos);
        let current_count = &mut current_counts[0];
        for l in lo..=hi {
            let size = (32 - (l as u32).leading_zeros()) as usize;
            let d = ax - l as f32;
            for &p in states.iter().take(n_states) {
                debug_assert!(p < pos);
                // SAFETY: the state invariant above gives p < pos, the length
                // of both prior slices; prefix has all 64 entries. Splitting
                // at pos also keeps parent paths disjoint from the list updated.
                let (parents, count, zeros) = unsafe {
                    (
                        prior_lists.get_unchecked(p),
                        *prior_counts.get_unchecked(p),
                        *prefix.get_unchecked(p),
                    )
                };
                let run = pos - p - 1;
                let bits =
                    (run >> 4) as f32 * len(0xF0) + len(((run & 15) << 4) | size) + size as f32;
                let edge = prefix[pos - 1] - zeros + d * d + lambda * bits;
                // insert_path starts from count 0 and caps it at k <= 4.
                for (r, parent) in parents.iter().take(count).enumerate() {
                    let cost = parent.cost + edge;
                    if *current_count == k && cost >= current[k - 1].cost {
                        break;
                    }
                    let path = Path {
                        cost,
                        from: p as u8,
                        rank: r as u8,
                        level: l as i16,
                    };
                    insert_path(current, current_count, k, path);
                }
            }
        }
        if *current_count > 0 {
            debug_assert!(n_states <= pos);
            states[n_states] = pos;
            n_states += 1;
        }
    }

    // Close each path: zeros after its last nonzero plus an EOB unless it sits at 63
    let mut ends = [none; KBEST_MAX];
    let mut n_ends = 0;
    for &p in states.iter().take(n_states) {
        debug_assert!(p < 64);
        // SAFETY: only 0 and completed positions 1..64 enter states. All three
        // arrays have 64 entries, and each count remains bounded by k <= 4.
        let (parents, count, zeros) = unsafe {
            (
                lists.get_unchecked(p),
                *counts.get_unchecked(p),
                *prefix.get_unchecked(p),
            )
        };
        let tail = prefix[63] - zeros + if p < 63 { lambda * len(0) } else { 0.0 };
        for (r, entry) in parents.iter().take(count).enumerate() {
            let path = Path {
                cost: entry.cost + tail,
                from: p as u8,
                rank: r as u8,
                level: 0,
            };
            insert_path(&mut ends, &mut n_ends, k, path);
        }
    }
    for end in &ends[..n_ends] {
        let mut levels = [0i16; 64];
        let (mut p, mut r) = (usize::from(end.from), usize::from(end.rank));
        while p > 0 {
            debug_assert!(p < 64 && r < k);
            // SAFETY: ends stores only state indices p < 64 and ranks
            // r < count[p] <= k <= 4. Each path stores a predecessor p' < p
            // and its valid rank; completed parent lists are never changed.
            // This invariant is preserved on every backtracking step.
            let step = unsafe {
                let step = *lists.get_unchecked(p).get_unchecked(r);
                *levels.get_unchecked_mut(p) = if *x.get_unchecked(p) < 0.0 {
                    -step.level
                } else {
                    step.level
                };
                step
            };
            (p, r) = (usize::from(step.from), usize::from(step.rank));
        }
        out.push(levels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_insert_path(
        list: &mut [Path; KBEST_MAX],
        count: &mut usize,
        k: usize,
        path: Path,
    ) {
        let mut i = if *count < k {
            *count += 1;
            *count - 1
        } else if path.cost < list[k - 1].cost {
            k - 1
        } else {
            return;
        };
        while i > 0 && list[i - 1].cost > path.cost {
            list[i] = list[i - 1];
            i -= 1;
        }
        list[i] = path;
    }

    /// The `k` cheapest AC level vectors (zigzag order, DC left zero) of a block
    /// with unquantized levels `x` under Σ (x − level)² + λ·bits, symbols priced
    /// by `lengths`.
    ///
    /// The trellis of `rdo.rs` with its state (last nonzero position) keeping
    /// its `k` best paths instead of one. Each position may take the rounded
    /// level, one below, one above, or 1 restored from `restore` up.
    fn reference_kbest_levels(
        x: &[f32; 64],
        lengths: &[u8; 256],
        lambda: f32,
        k: usize,
        restore: f32,
        out: &mut Vec<[i16; 64]>,
    ) {
        let k = k.clamp(1, KBEST_MAX);
        let len = |s: usize| match lengths[s] {
            0 => 24.0,
            l => f32::from(l),
        };
        let mut prefix = [0f32; 64];
        for i in 1..64 {
            prefix[i] = prefix[i - 1] + x[i] * x[i];
        }
        let none = Path {
            cost: f32::INFINITY,
            from: 0,
            rank: 0,
            level: 0,
        };
        let mut lists = [[none; KBEST_MAX]; 64];
        let mut counts = [0usize; 64];
        lists[0][0].cost = 0.0;
        counts[0] = 1;
        let mut states = [0usize; 64];
        let mut n_states = 1;

        for pos in 1..64 {
            let ax = x[pos].abs();
            let rounded = (ax + 0.5) as i32;
            let (lo, hi) = match rounded {
                0 if ax >= restore => (1, 1),
                0 => continue,
                r => ((r - 1).max(1), r + 1),
            };
            for l in lo..=hi {
                let size = (32 - (l as u32).leading_zeros()) as usize;
                let d = ax - l as f32;
                for &p in &states[..n_states] {
                    let run = pos - p - 1;
                    let bits =
                        (run >> 4) as f32 * len(0xF0) + len(((run & 15) << 4) | size) + size as f32;
                    let edge = prefix[pos - 1] - prefix[p] + d * d + lambda * bits;
                    for r in 0..counts[p] {
                        let cost = lists[p][r].cost + edge;
                        if counts[pos] == k && cost >= lists[pos][k - 1].cost {
                            break;
                        }
                        let path = Path {
                            cost,
                            from: p as u8,
                            rank: r as u8,
                            level: l as i16,
                        };
                        reference_insert_path(&mut lists[pos], &mut counts[pos], k, path);
                    }
                }
            }
            if counts[pos] > 0 {
                states[n_states] = pos;
                n_states += 1;
            }
        }

        // Close each path: zeros after its last nonzero plus an EOB unless it sits at 63
        let mut ends = [none; KBEST_MAX];
        let mut n_ends = 0;
        for &p in &states[..n_states] {
            let tail = prefix[63] - prefix[p] + if p < 63 { lambda * len(0) } else { 0.0 };
            for (r, entry) in lists[p][..counts[p]].iter().enumerate() {
                let path = Path {
                    cost: entry.cost + tail,
                    from: p as u8,
                    rank: r as u8,
                    level: 0,
                };
                reference_insert_path(&mut ends, &mut n_ends, k, path);
            }
        }
        for end in &ends[..n_ends] {
            let mut levels = [0i16; 64];
            let (mut p, mut r) = (usize::from(end.from), usize::from(end.rank));
            while p > 0 {
                let step = lists[p][r];
                levels[p] = if x[p] < 0.0 { -step.level } else { step.level };
                (p, r) = (usize::from(step.from), usize::from(step.rank));
            }
            out.push(levels);
        }
    }

    #[test]
    fn index_candidates_match_checked_ranking() {
        let gains = [0.0, -0.0, -1.0, 1.0, f64::NAN, f64::INFINITY, 2.0, 1.0];
        for len in 0..32 {
            let candidates: Vec<_> = (0..len)
                .map(|i| Candidate {
                    kind: MoveKind::Restore,
                    levels: [0; 64],
                    bits: 0,
                    proxy: 0.0,
                    gain: gains[i % gains.len()],
                })
                .collect();
            let mut expected: Vec<_> = (0..len).filter(|&i| candidates[i].gain > 0.0).collect();
            expected.sort_by(|&a, &b| candidates[b].gain.total_cmp(&candidates[a].gain));
            let mut actual = vec![usize::MAX; 64];
            rank_candidates(&candidates, &mut actual);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn index_trellis_matches_checked_reference() {
        let mut seed = 0x92a4_1bc7u32;
        // Miri exercises all four state densities and both rank limits;
        // native tests additionally sweep the full quality/threshold matrix.
        let cases = if cfg!(miri) { 4 } else { 32 };
        let ranks: &[usize] = if cfg!(miri) {
            &[1, 4]
        } else {
            &[0, 1, 2, 4, usize::MAX]
        };
        let lambdas: &[f32] = if cfg!(miri) {
            &[0.0, 0.5]
        } else {
            &[0.0, 0.015, 0.5, 10.0]
        };
        let thresholds: &[f32] = if cfg!(miri) { &[0.25] } else { &[0.0, 0.25] };
        for case in 0..cases {
            let mut x = core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                if case % 4 == 0 {
                    0.0
                } else if case % 4 == 1 {
                    ((seed >> 16) % 5) as f32 * 0.5 - 1.0
                } else if case % 4 == 2 && seed & 7 != 0 {
                    0.0
                } else {
                    (seed >> 16) as f32 / 4096.0 - 8.0
                }
            });
            // Include unordered costs: the safe rewrite must preserve the
            // insertion routine's strict-comparison behavior for NaNs too.
            if case == 31 {
                x[15] = f32::NAN;
            }
            let lengths = core::array::from_fn(|s| (s % 17) as u8);
            for &k in ranks {
                for &lambda in lambdas {
                    for &restore in thresholds {
                        let (mut actual, mut expected) = (Vec::new(), Vec::new());
                        kbest_levels(&x, &lengths, lambda, k, restore, &mut actual);
                        reference_kbest_levels(&x, &lengths, lambda, k, restore, &mut expected);
                        assert_eq!(
                            actual, expected,
                            "case={case} k={k} lambda={lambda} restore={restore}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn row_upsampling_matches_scalar_at_every_crop_origin() {
        for width in [1usize, 2, 3, 8, 9, 17, 33] {
            for height in [1usize, 2, 3, 8, 9, 17] {
                let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
                let stride = cw + 3;
                let plane: Vec<u8> = (0..stride * ch).map(|i| (i * 73 + 5) as u8).collect();
                for y in 0..height {
                    let cy = y / 2;
                    let near = if y & 1 == 0 {
                        cy.saturating_sub(1)
                    } else {
                        (cy + 1).min(ch - 1)
                    };
                    let colsum = |cx| {
                        3 * i32::from(plane[cy * stride + cx])
                            + i32::from(plane[near * stride + cx])
                    };
                    let expected: Vec<u8> = (0..width)
                        .map(|x| {
                            let cx = x / 2;
                            if x & 1 == 0 {
                                ((3 * colsum(cx) + colsum(cx.saturating_sub(1)) + 8) >> 4) as u8
                            } else {
                                ((3 * colsum(cx) + colsum((cx + 1).min(cw - 1)) + 7) >> 4) as u8
                            }
                        })
                        .collect();
                    for x0 in 0..width {
                        let actual: Vec<_> = h2v2_pairs(&plane, stride, cw, ch, y, x0)
                            .flatten()
                            .skip(x0 & 1)
                            .take(width - x0)
                            .collect();
                        assert_eq!(
                            actual,
                            expected[x0..],
                            "size {width}x{height} origin {x0},{y}"
                        );
                    }
                }
            }
        }
    }

    use crate::idct::{CONST_BITS, PASS1_BITS, descale, pass};

    fn scalar_idct_reference(levels: &[i16; 64], table: &[u16; 64], out: &mut [u8; 64]) {
        let mut coef = [0i32; 64];
        for k in 0..64 {
            let z = usize::from(ZIGZAG[k]);
            coef[z] = i32::from(levels[k]) * i32::from(table[z]);
        }
        let mut ws = [0i32; 64];
        for c in 0..8 {
            let i = |r: usize| coef[r * 8 + c];
            if (1..8).all(|r| i(r) == 0) {
                let dc = i(0) << PASS1_BITS;
                for r in 0..8 {
                    ws[r * 8 + c] = dc;
                }
                continue;
            }
            let r = pass(
                [i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7)],
                CONST_BITS - PASS1_BITS,
            );
            for (row, v) in r.into_iter().enumerate() {
                ws[row * 8 + c] = v;
            }
        }
        for r in 0..8 {
            let w = &ws[r * 8..r * 8 + 8];
            let values = if w[1..].iter().all(|&v| v == 0) {
                [descale(w[0], PASS1_BITS + 3); 8]
            } else {
                pass(
                    [w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]],
                    CONST_BITS + PASS1_BITS + 3,
                )
            };
            for (x, v) in values.into_iter().enumerate() {
                out[r * 8 + x] = (v + 128).clamp(0, 255) as u8;
            }
        }
    }

    #[test]
    fn row_idct_matches_original_integer_rounding() {
        let mut seed = 0x1234_5678u32;
        for density in [0, 1, 8, 64] {
            for _ in 0..128 {
                let table = core::array::from_fn(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed % 255 + 1) as u16
                });
                let levels = core::array::from_fn(|k| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    if k == 0 {
                        (seed % 65) as i16 - 32
                    } else if (seed >> 16) % 64 < density {
                        (seed % 9) as i16 - 4
                    } else {
                        0
                    }
                });
                let (mut actual, mut expected) = ([0; 64], [0; 64]);
                (crate::idct::islow())(&levels, &table, &mut actual);
                scalar_idct_reference(&levels, &table, &mut expected);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn chunked_source_blocks_match_clamped_sampling() {
        for (width, height) in [(8usize, 8usize), (9, 17), (16, 16), (17, 33), (33, 25)] {
            let source: Vec<u8> = (0..width * height * 3)
                .map(|i| (i * 37 + i / 11) as u8)
                .collect();
            for sub in [1, 2] {
                let specs: Vec<_> = (0..3)
                    .map(|c| {
                        let s = if c == 0 { 1 } else { sub };
                        PlaneSpec {
                            padded_cols: width.div_ceil(8 * sub) * sub / s,
                            padded_rows: height.div_ceil(8 * sub) * sub / s,
                            sub: s,
                            ac_table: usize::from(c > 0),
                            dc_table: usize::from(c > 0),
                        }
                    })
                    .collect();
                let polisher = Polisher::new(
                    &source,
                    width,
                    height,
                    specs.clone(),
                    PolishOptions::default(),
                )
                .unwrap();
                for (c, spec) in specs.iter().enumerate() {
                    let s = spec.sub;
                    let (cw, ch) = (width.div_ceil(s), height.div_ceil(s));
                    for b in 0..spec.padded_cols * spec.padded_rows {
                        let (bx, by) = (b % spec.padded_cols, b / spec.padded_cols);
                        // Original scalar oracle, including clamping in component
                        // space before replicating an odd final source row/column.
                        let expected: [f32; 64] = core::array::from_fn(|i| {
                            let (x, y) =
                                ((bx * 8 + i % 8).min(cw - 1), (by * 8 + i / 8).min(ch - 1));
                            let mut sum = 0.0;
                            for dy in 0..s {
                                for dx in 0..s {
                                    let (px, py) =
                                        ((x * s + dx).min(width - 1), (y * s + dy).min(height - 1));
                                    let at = (py * width + px) * 3;
                                    let (r, g, b) = (
                                        f32::from(source[at]),
                                        f32::from(source[at + 1]),
                                        f32::from(source[at + 2]),
                                    );
                                    sum += match c {
                                        0 => 0.299 * r + 0.587 * g + 0.114 * b,
                                        1 => -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0,
                                        _ => 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0,
                                    };
                                }
                            }
                            sum / (s * s) as f32 - 128.0
                        });
                        let actual = match c {
                            0 => polisher.source_block::<0>(b),
                            1 => polisher.source_block::<1>(b),
                            _ => polisher.source_block::<2>(b),
                        };
                        assert_eq!(
                            actual.map(f32::to_bits),
                            expected.map(f32::to_bits),
                            "size {width}x{height} sub {sub} component {c} block {b}"
                        );
                        // Independent dense reference: confirm natural order and
                        // normalization through the polisher's actual DCT call.
                        let mut dense = [0.0; 64];
                        crate::dct_f32::reference::fdct(&expected, &mut dense);
                        let fast = polisher.ideal_block(c, b);
                        for (k, (&a, &b)) in fast.iter().zip(&dense).enumerate() {
                            assert!((a - b).abs() < 0.001, "frequency {k}: {a} vs {b}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn local_ac_costs_match_complete_tokenization() {
        let lengths = core::array::from_fn(|s| (s % 17) as u8);
        let mut seed = 0x1234_5678u32;
        for density in [0, 1, 4, 16, 64] {
            for _ in 0..16 {
                let mut levels = [0i16; 64];
                for level in &mut levels[1..] {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    if (seed >> 16) % 64 < density {
                        *level = ((seed >> 8) % 2047) as i16 - 1023;
                    }
                }
                let costs = AcCosts::new(&levels, &lengths);
                let base = ac_bits(&levels, &lengths);
                for k in 1..64 {
                    // Include zero insertion/removal, category boundaries and
                    // long zero runs, with absent Huffman symbols priced at 24.
                    for new in [0, -1, 1, -2, 2, -1023, 1023, levels[k] - 1, levels[k] + 1] {
                        let mut changed = levels;
                        changed[k] = new;
                        assert_eq!(
                            costs.delta(k, new),
                            ac_bits(&changed, &lengths) - base,
                            "coefficient {k}: {} -> {new}",
                            levels[k]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn trials_commit_or_restore_both_metric_states() {
        let (width, height) = (33usize, 25usize);
        let source: Vec<u8> = (0..width * height * 3)
            .map(|i| (i * 37 + i / 11) as u8)
            .collect();
        for sub in [1, 2] {
            for judge in [
                PolishJudge::Ssimulacra2,
                PolishJudge::Cvvdp,
                PolishJudge::Both,
            ] {
                let specs: Vec<_> = (0..3)
                    .map(|c| {
                        let s = if c == 0 { 1 } else { sub };
                        PlaneSpec {
                            padded_cols: width.div_ceil(8 * sub) * sub / s,
                            padded_rows: height.div_ceil(8 * sub) * sub / s,
                            sub: s,
                            ac_table: usize::from(c > 0),
                            dc_table: usize::from(c > 0),
                        }
                    })
                    .collect();
                let options = PolishOptions {
                    judge,
                    ..PolishOptions::default()
                };
                let polisher =
                    Polisher::new(&source, width, height, specs.clone(), options).unwrap();
                let mut blocks: Vec<_> = specs
                    .iter()
                    .map(|s| vec![AlignedBlock { data: [0; 64] }; s.padded_cols * s.padded_rows])
                    .collect();
                let tables = [[16; 64]; 3];
                let frame = Frame::new(width, height, &specs, &blocks, &tables);
                let ss2 = (judge != PolishJudge::Cvvdp).then(|| {
                    polisher
                        .ss2
                        .as_ref()
                        .unwrap()
                        .incremental(polisher.rgb_ss2(&frame.rgb))
                        .unwrap()
                });
                let cv = (judge != PolishJudge::Ssimulacra2).then(|| {
                    cvvdp_lite::IncrementalState::new(
                        polisher.cvvdp.as_ref().unwrap(),
                        polisher.rgb_cv(&frame.rgb),
                    )
                    .unwrap()
                });
                let mut state = JudgeState {
                    cur_ss2: ss2.as_ref().map_or(f64::NAN, |s| s.score()),
                    cur_jod: cv.as_ref().map_or(f64::NAN, |s| s.score().jod),
                    frame,
                    ss2,
                    cv,
                    ss2_cp: Default::default(),
                    cv_cp: Default::default(),
                    saved: Vec::new(),
                    accepted: Vec::new(),
                    slopes: (1.0, 1.0),
                    total_bits: 1.0,
                };
                let mut stats = PolishStats::default();
                for c in 0..3 {
                    // Force rejection, direct commit of candidate zero, and
                    // commit of another candidate, including after a rollback.
                    for (trial, bits) in [
                        [1_000_000, 1_000_000],
                        [-2_000_000, -1_000_000],
                        [-1_000_000, -2_000_000],
                        [1_000_000, 1_000_000],
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let b = trial % 2;
                        let before = state.frame.rgb.clone();
                        let old_scores = (state.cur_ss2.to_bits(), state.cur_jod.to_bits());
                        let mut candidates: Vec<_> = bits
                            .into_iter()
                            .enumerate()
                            .map(|(k, bits)| {
                                let mut levels = blocks[c][b].data;
                                levels[k + 1] += 1;
                                Candidate {
                                    kind: MoveKind::Restore,
                                    levels,
                                    bits,
                                    proxy: k as f32,
                                    gain: 0.0,
                                }
                            })
                            .collect();
                        let selected = polisher.judge(
                            &mut state,
                            &mut stats,
                            c,
                            b,
                            &tables[c],
                            &mut candidates,
                        );
                        assert_eq!(
                            selected,
                            match trial {
                                1 => Some(0),
                                2 => Some(1),
                                _ => None,
                            }
                        );
                        if let Some(i) = selected {
                            blocks[c][b].data = candidates[i].levels;
                        } else {
                            assert_eq!(state.frame.rgb, before);
                            assert_eq!(
                                (state.cur_ss2.to_bits(), state.cur_jod.to_bits()),
                                old_scores
                            );
                        }
                        let expected = Frame::new(width, height, &specs, &blocks, &tables);
                        assert_eq!(state.frame.rgb, expected.rgb);
                        assert_eq!(state.frame.planes, expected.planes);
                        if let Some(ss2) = &state.ss2 {
                            let full = polisher
                                .ss2
                                .as_ref()
                                .unwrap()
                                .compare(polisher.rgb_ss2(&expected.rgb))
                                .unwrap();
                            assert!((ss2.score() - full).abs() < 1e-5);
                            assert_eq!(state.cur_ss2.to_bits(), ss2.score().to_bits());
                        }
                        if let Some(cv) = &state.cv {
                            let full = polisher
                                .cvvdp
                                .as_ref()
                                .unwrap()
                                .evaluator()
                                .compare(polisher.rgb_cv(&expected.rgb))
                                .unwrap();
                            assert!((cv.score().jod - full.jod).abs() < 1e-5);
                            assert_eq!(state.cur_jod.to_bits(), cv.score().jod.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ac_bits_counts_eob_and_zrl() {
        let mut lengths = [4u8; 256];
        lengths[0] = 2;
        lengths[0xF0] = 3;
        let mut levels = [0i16; 64];
        assert_eq!(ac_bits(&levels, &lengths), 2);
        levels[20] = 1;
        // one ZRL (run 19 = 16 + 3), symbol 0x31 + 1 extra bit, EOB
        assert_eq!(ac_bits(&levels, &lengths), 3 + 4 + 1 + 2);
        levels[63] = -2;
        assert_eq!(ac_bits(&levels, &lengths), 3 + 4 + 1 + 3 + 3 + 4 + 2);
    }

    #[test]
    fn kbest_without_rate_starts_with_rounding() {
        let mut x = [0f32; 64];
        x[1] = 2.4;
        x[2] = -0.7;
        x[5] = 0.3;
        x[9] = 1.6;
        let lengths = [4u8; 256];
        let mut out = Vec::new();
        kbest_levels(&x, &lengths, 0.0, 3, 0.25, &mut out);
        assert_eq!(out.len(), 3);
        let mut rounded = [0i16; 64];
        (rounded[1], rounded[2], rounded[9]) = (2, -1, 2);
        assert_eq!(out[0], rounded);
        assert!(out[1] != out[0] && out[2] != out[1] && out[2] != out[0]);
        // Costs come out in order
        let err = |l: &[i16; 64]| {
            (1..64)
                .map(|k| (x[k] - f32::from(l[k])).powi(2))
                .sum::<f32>()
        };
        assert!(err(&out[0]) <= err(&out[1]) && err(&out[1]) <= err(&out[2]));
    }

    #[test]
    fn kbest_high_lambda_drops_everything() {
        let mut x = [0f32; 64];
        x[3] = 1.2;
        x[40] = -0.9;
        let lengths = [6u8; 256];
        let mut out = Vec::new();
        kbest_levels(&x, &lengths, 10.0, 1, 0.25, &mut out);
        assert_eq!(out, vec![[0i16; 64]]);
    }
}
