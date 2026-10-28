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
 * //
 * // Portions of this file are derived from the Independent JPEG Group's software,
 * // see LICENSE-IJG.
 */
//! Progressive scans.
//!
//! Spectral selection and successive approximation coded as libjpeg's phuff
//! coder does it: EOB runs across blocks, buffered correction bits in
//! refinement scans, one optimized Huffman table per scan. Every scan of a
//! component is independent of the component's other scans in size, so a
//! scan script is chosen per component by costing each candidate scan
//! exactly once and picking the cheapest combination.

use crate::EncodingError;
use crate::encoder::AlignedBlock;
use crate::huffman::{CodingClass, HuffmanTable};
use crate::marker::Marker;
use crate::writer::{JfifWrite, JfifWriter, get_code, pack_flags, significant_mask};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// How progressive scans are laid out
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressiveMode {
    /// Spectral selection only, `n` scans per component, one Huffman table per component
    Spectral(u8),
    /// libjpeg's standard script: successive approximation, per-scan tables
    Standard,
    /// Per component, the cheapest script among the candidates on exact bits
    Optimized,
}

/// One scan of one component: spectral band `ss..=se`, successive
/// approximation from bit `ah` (0 = first scan) down to bit `al`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Scan {
    pub ss: u8,
    pub se: u8,
    pub ah: u8,
    pub al: u8,
}

impl Scan {
    const fn dc(ah: u8, al: u8) -> Scan {
        Scan {
            ss: 0,
            se: 0,
            ah,
            al,
        }
    }

    const fn ac(ss: u8, se: u8, ah: u8, al: u8) -> Scan {
        Scan { ss, se, ah, al }
    }

    fn is_dc(&self) -> bool {
        self.ss == 0
    }

    fn is_refine(&self) -> bool {
        self.ah > 0
    }

    /// A DC refinement scan carries raw bits only, no Huffman table
    fn has_table(&self) -> bool {
        !(self.is_dc() && self.is_refine())
    }
}

/// Where coded symbols go: counted to size a scan, or written
trait Sink {
    fn symbol(&mut self, symbol: u8) -> Result<(), EncodingError>;
    fn bits(&mut self, value: u32, count: u8) -> Result<(), EncodingError>;
    fn restart(&mut self, index: u8) -> Result<(), EncodingError>;
}

struct Counter {
    freq: [u32; 257],
    raw_bits: u64,
    restarts: u64,
}

impl Counter {
    fn new() -> Counter {
        let mut freq = [0u32; 257];
        freq[256] = 1;
        Counter {
            freq,
            raw_bits: 0,
            restarts: 0,
        }
    }

    /// Optimized table for the counted symbols
    fn table(&self) -> HuffmanTable {
        let mut freq = self.freq;
        if freq[..256].iter().all(|&f| f == 0) {
            freq[0] = 1;
        }
        HuffmanTable::new_optimized(freq)
    }

    /// Bits of the scan with its own optimized table, headers included
    fn bits(&self, scan: &Scan) -> u64 {
        let mut bits = self.raw_bits + self.restarts * 23 + 7;
        // SOS for one component: marker + 8 bytes
        bits += 10 * 8;
        if scan.has_table() {
            let table = self.table();
            for (symbol, &f) in self.freq.iter().enumerate().take(256) {
                if f > 0 {
                    bits += f as u64 * table.get_for_value(symbol as u8).0 as u64;
                }
            }
            // DHT: marker + length + class/dest + 16 counts + values
            bits += (2 + 2 + 1 + 16 + table.values().len() as u64) * 8;
        }
        bits
    }
}

impl Sink for Counter {
    #[inline]
    fn symbol(&mut self, symbol: u8) -> Result<(), EncodingError> {
        self.freq[symbol as usize] += 1;
        Ok(())
    }

    #[inline]
    fn bits(&mut self, _value: u32, count: u8) -> Result<(), EncodingError> {
        self.raw_bits += count as u64;
        Ok(())
    }

    fn restart(&mut self, _index: u8) -> Result<(), EncodingError> {
        self.restarts += 1;
        Ok(())
    }
}

struct Emitter<'a, W: JfifWrite> {
    writer: &'a mut JfifWriter<W>,
    table: Option<&'a HuffmanTable>,
}

impl<W: JfifWrite> Sink for Emitter<'_, W> {
    #[inline]
    fn symbol(&mut self, symbol: u8) -> Result<(), EncodingError> {
        let table = self.table.expect("scan without a table emits no symbols");
        self.writer.huffman_encode(symbol, table)
    }

    #[inline]
    fn bits(&mut self, value: u32, count: u8) -> Result<(), EncodingError> {
        if count > 0 {
            self.writer.write_bits(value, count)
        } else {
            Ok(())
        }
    }

    fn restart(&mut self, index: u8) -> Result<(), EncodingError> {
        self.writer.finalize_bit_buffer()?;
        self.writer.write_marker(Marker::RST(index))
    }
}

/// EOB run and the correction bits waiting behind it
struct AcState {
    eobrun: u32,
    pending: Vec<u8>,
}

/// libjpeg keeps at most 1000 buffered correction bits
const MAX_CORRECTION_BITS: usize = 1000;

impl AcState {
    fn new() -> AcState {
        AcState {
            eobrun: 0,
            pending: Vec::with_capacity(MAX_CORRECTION_BITS + 64),
        }
    }

    fn emit_eobrun<S: Sink>(&mut self, sink: &mut S) -> Result<(), EncodingError> {
        if self.eobrun > 0 {
            let nbits = 31 - self.eobrun.leading_zeros();
            sink.symbol((nbits << 4) as u8)?;
            if nbits > 0 {
                sink.bits(self.eobrun & ((1 << nbits) - 1), nbits as u8)?;
            }
            self.eobrun = 0;
            for &bit in &self.pending {
                sink.bits(bit as u32, 1)?;
            }
            self.pending.clear();
        }
        Ok(())
    }
}

#[inline]
fn bit_length(value: u32) -> u32 {
    32 - value.leading_zeros()
}

/// libjpeg's first AC scan coder, visiting only the coefficients that stay
/// nonzero after the point transform; the positions between them are the zero
/// runs the coefficient-by-coefficient loop counted.
fn encode_ac_first<S: Sink>(
    block: &AlignedBlock,
    scan: &Scan,
    state: &mut AcState,
    sink: &mut S,
) -> Result<(), EncodingError> {
    let al = scan.al;
    let (ss, se) = (usize::from(scan.ss), usize::from(scan.se));
    let band = (u64::MAX << ss) & (u64::MAX >> (63 - se));
    let mut significant = significant_mask(&block.data, al) & band;
    let mut next = ss;
    while significant != 0 {
        let k = significant.trailing_zeros() as usize;
        significant &= significant - 1;
        let mut temp = block.data[k] as i32;
        let temp2 = if temp < 0 {
            temp = -temp;
            temp >>= al;
            !temp
        } else {
            temp >>= al;
            temp
        };
        state.emit_eobrun(sink)?;
        let mut run = (k - next) as u32;
        while run > 15 {
            sink.symbol(0xF0)?;
            run -= 16;
        }
        let nbits = bit_length(temp as u32);
        sink.symbol(((run << 4) + nbits) as u8)?;
        sink.bits((temp2 as u32) & ((1 << nbits) - 1), nbits as u8)?;
        next = k + 1;
    }
    if next <= se {
        state.eobrun += 1;
        if state.eobrun == 0x7FFF {
            state.emit_eobrun(sink)?;
        }
    }
    Ok(())
}

/// The original coefficient-by-coefficient first AC scan coder.
// Index loops mirror libjpeg's coder, which this is a port of.
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
fn reference_encode_ac_first<S: Sink>(
    block: &AlignedBlock,
    scan: &Scan,
    state: &mut AcState,
    sink: &mut S,
) -> Result<(), EncodingError> {
    let al = scan.al;
    let mut run = 0u32;
    for k in scan.ss as usize..=scan.se as usize {
        let mut temp = block.data[k] as i32;
        if temp == 0 {
            run += 1;
            continue;
        }
        let temp2 = if temp < 0 {
            temp = -temp;
            temp >>= al;
            !temp
        } else {
            temp >>= al;
            temp
        };
        if temp == 0 {
            run += 1;
            continue;
        }
        state.emit_eobrun(sink)?;
        while run > 15 {
            sink.symbol(0xF0)?;
            run -= 16;
        }
        let nbits = bit_length(temp as u32);
        sink.symbol(((run << 4) + nbits) as u8)?;
        sink.bits((temp2 as u32) & ((1 << nbits) - 1), nbits as u8)?;
        run = 0;
    }
    if run > 0 {
        state.eobrun += 1;
        if state.eobrun == 0x7FFF {
            state.emit_eobrun(sink)?;
        }
    }
    Ok(())
}

// Index loops mirror libjpeg's coder, which this is a port of.
#[allow(clippy::needless_range_loop)]
fn encode_ac_refine<S: Sink>(
    block: &AlignedBlock,
    scan: &Scan,
    state: &mut AcState,
    sink: &mut S,
) -> Result<(), EncodingError> {
    let al = scan.al;
    let (ss, se) = (scan.ss as usize, scan.se as usize);
    let mut absvalues = [0i32; 64];
    let mut eob = 0usize;
    for (k, (slot, &coefficient)) in absvalues.iter_mut().zip(block.data.iter()).enumerate() {
        if k < ss || k > se {
            continue;
        }
        let v = (coefficient as i32).unsigned_abs() >> al;
        *slot = v as i32;
        if v == 1 {
            eob = k;
        }
    }

    let mut run = 0u32;
    let mut buffered = [0u8; 64];
    let mut buffered_len = 0usize;
    for k in ss..=se {
        let temp = absvalues[k];
        if temp == 0 {
            run += 1;
            continue;
        }
        while run > 15 && k <= eob {
            state.emit_eobrun(sink)?;
            sink.symbol(0xF0)?;
            run -= 16;
            for &bit in &buffered[..buffered_len] {
                sink.bits(bit as u32, 1)?;
            }
            buffered_len = 0;
        }
        if temp > 1 {
            // Already nonzero in an earlier scan: one correction bit.
            buffered[buffered_len] = (temp & 1) as u8;
            buffered_len += 1;
            continue;
        }
        state.emit_eobrun(sink)?;
        sink.symbol(((run << 4) + 1) as u8)?;
        sink.bits(if block.data[k] < 0 { 0 } else { 1 }, 1)?;
        for &bit in &buffered[..buffered_len] {
            sink.bits(bit as u32, 1)?;
        }
        buffered_len = 0;
        run = 0;
    }
    if run > 0 || buffered_len > 0 {
        state.eobrun += 1;
        state.pending.extend_from_slice(&buffered[..buffered_len]);
        if state.eobrun == 0x7FFF || state.pending.len() > MAX_CORRECTION_BITS - 64 {
            state.emit_eobrun(sink)?;
        }
    }
    Ok(())
}

/// Code one component's blocks for `scan` into `sink`
fn encode_scan<S: Sink>(
    blocks: &[AlignedBlock],
    scan: &Scan,
    restart_interval: usize,
    sink: &mut S,
    prepared: Option<&[PreparedBlock]>,
) -> Result<(), EncodingError> {
    let mut state = AcState::new();
    let mut prev_dc = 0i32;
    let mut restarts = 0u8;
    for (b, block) in blocks.iter().enumerate() {
        if restart_interval > 0 && b > 0 && b.is_multiple_of(restart_interval) {
            state.emit_eobrun(sink)?;
            sink.restart(restarts)?;
            restarts = (restarts + 1) & 7;
            prev_dc = 0;
        }
        if scan.is_dc() {
            let value = (block.data[0] as i32) >> scan.al;
            if scan.is_refine() {
                sink.bits((value & 1) as u32, 1)?;
            } else {
                let (size, bits) = get_code((value - prev_dc) as i16);
                sink.symbol(size)?;
                sink.bits(bits as u32, size)?;
                prev_dc = value;
            }
        } else if let Some(prepared) = prepared {
            encode_prepared_ac(block, &prepared[b], scan, &mut state, sink)?;
        } else if scan.is_refine() {
            encode_ac_refine(block, scan, &mut state, sink)?;
        } else {
            encode_ac_first(block, scan, &mut state, sink)?;
        }
    }
    state.emit_eobrun(sink)
}

/// Count a scan: symbol frequencies and raw bits
fn count_scan(blocks: &[AlignedBlock], scan: &Scan, restart_interval: usize) -> Counter {
    let mut counter = Counter::new();
    encode_scan(blocks, scan, restart_interval, &mut counter, None).expect("counting cannot fail");
    counter
}

/// Counting needs magnitude categories, not coefficient signs or correction-bit
/// values. Reuse these across the candidate scans instead of rescanning 64
/// coefficients for each band and successive-approximation depth.
struct PreparedBlock {
    dc: i16,
    sizes: [u8; 64],
    nonzero: [u64; 3],
}

impl PreparedBlock {
    fn new(block: &AlignedBlock) -> Self {
        let sizes = block
            .data
            .map(|v| bit_length(u32::from(v.unsigned_abs())) as u8);
        // AC positions only: bit 0 (DC) stays clear.
        let nonzero = core::array::from_fn(|al| {
            pack_flags(&sizes.map(|size| u8::from(usize::from(size) > al))) & !1
        });
        Self {
            dc: block.data[0],
            sizes,
            nonzero,
        }
    }

    #[inline]
    fn nonzero_at(&self, al: u8) -> u64 {
        self.nonzero
            .get(usize::from(al))
            .copied()
            .unwrap_or_else(|| {
                self.sizes
                    .iter()
                    .enumerate()
                    .skip(1)
                    .fold(0, |mask, (k, &size)| mask | (u64::from(size > al) << k))
            })
    }
}

/// What counting first AC scans reads of a block: the positions that stay
/// nonzero after a point transform and their magnitude categories
trait Levels {
    /// AC positions (bit 0 clear) whose level is nonzero after shifting by `al`
    fn nonzero_at(&self, al: u8) -> u64;
    /// Bit length of the magnitude at `k` before the point transform
    fn size(&self, k: usize) -> u8;
}

impl Levels for PreparedBlock {
    #[inline]
    fn nonzero_at(&self, al: u8) -> u64 {
        PreparedBlock::nonzero_at(self, al)
    }

    #[inline]
    fn size(&self, k: usize) -> u8 {
        self.sizes[k]
    }
}

impl Levels for AlignedBlock {
    #[inline]
    fn nonzero_at(&self, al: u8) -> u64 {
        significant_mask(&self.data, al) & !1
    }

    #[inline]
    fn size(&self, k: usize) -> u8 {
        bit_length(u32::from(self.data[k].unsigned_abs())) as u8
    }
}

/// Reuse the counting masks when emitting a component's refinement script.
/// Gaps between set bits count only zero coefficients; older nonzeros still
/// buffer their correction bits in the original coefficient order.
fn encode_prepared_ac<S: Sink>(
    block: &AlignedBlock,
    prepared: &PreparedBlock,
    scan: &Scan,
    state: &mut AcState,
    sink: &mut S,
) -> Result<(), EncodingError> {
    let (ss, se) = (usize::from(scan.ss), usize::from(scan.se));
    let band = (u64::MAX << ss) & (u64::MAX >> (63 - se));
    let mut nonzero = prepared.nonzero_at(scan.al) & band;
    let mut next = ss;
    if !scan.is_refine() {
        while nonzero != 0 {
            let k = nonzero.trailing_zeros() as usize;
            nonzero &= nonzero - 1;
            let mut run = k - next;
            state.emit_eobrun(sink)?;
            while run > 15 {
                sink.symbol(0xF0)?;
                run -= 16;
            }
            let size = prepared.sizes[k] - scan.al;
            let temp = u32::from(block.data[k].unsigned_abs()) >> scan.al;
            let value = if block.data[k] < 0 { !temp } else { temp };
            sink.symbol(((run << 4) + usize::from(size)) as u8)?;
            sink.bits(value & ((1 << size) - 1), size)?;
            next = k + 1;
        }
        if next <= se {
            state.eobrun += 1;
            if state.eobrun == 0x7FFF {
                state.emit_eobrun(sink)?;
            }
        }
        return Ok(());
    }

    let older = prepared.nonzero_at(scan.al + 1);
    let new = nonzero & !older;
    let eob = if new == 0 {
        0
    } else {
        63 - new.leading_zeros() as usize
    };
    let mut run = 0usize;
    let mut buffered = [0u8; 64];
    let mut buffered_len = 0usize;
    while nonzero != 0 {
        let k = nonzero.trailing_zeros() as usize;
        nonzero &= nonzero - 1;
        run += k - next;
        next = k + 1;
        while run > 15 && k <= eob {
            state.emit_eobrun(sink)?;
            sink.symbol(0xF0)?;
            run -= 16;
            for &bit in &buffered[..buffered_len] {
                sink.bits(u32::from(bit), 1)?;
            }
            buffered_len = 0;
        }
        if older & (1 << k) != 0 {
            buffered[buffered_len] =
                ((u32::from(block.data[k].unsigned_abs()) >> scan.al) & 1) as u8;
            buffered_len += 1;
            continue;
        }
        state.emit_eobrun(sink)?;
        sink.symbol(((run << 4) + 1) as u8)?;
        sink.bits(u32::from(block.data[k] >= 0), 1)?;
        for &bit in &buffered[..buffered_len] {
            sink.bits(u32::from(bit), 1)?;
        }
        buffered_len = 0;
        run = 0;
    }
    run += se + 1 - next;
    if run > 0 || buffered_len > 0 {
        state.eobrun += 1;
        state.pending.extend_from_slice(&buffered[..buffered_len]);
        if state.eobrun == 0x7FFF || state.pending.len() > MAX_CORRECTION_BITS - 64 {
            state.emit_eobrun(sink)?;
        }
    }
    Ok(())
}

struct CountedAcState {
    eobrun: u32,
    pending: usize,
}

impl CountedAcState {
    #[inline]
    fn emit_eobrun(&mut self, counter: &mut Counter) {
        if self.eobrun > 0 {
            let nbits = 31 - self.eobrun.leading_zeros();
            counter.freq[(nbits << 4) as usize] += 1;
            counter.raw_bits += u64::from(nbits) + self.pending as u64;
            self.eobrun = 0;
            self.pending = 0;
        }
    }
}

/// The same symbols, raw-bit counts and flush boundaries as encode_scan.
/// Sparse masks skip zeros; pending correction bits only need a length.
fn count_prepared_scan(blocks: &[PreparedBlock], scan: &Scan, restart_interval: usize) -> Counter {
    let mut counter = Counter::new();
    let mut state = CountedAcState {
        eobrun: 0,
        pending: 0,
    };
    let mut prev_dc = 0i32;
    let ss = usize::from(scan.ss);
    let se = usize::from(scan.se);
    let band = (u64::MAX << ss) & (u64::MAX >> (63 - se));
    for (b, block) in blocks.iter().enumerate() {
        if restart_interval > 0 && b > 0 && b.is_multiple_of(restart_interval) {
            state.emit_eobrun(&mut counter);
            counter.restarts += 1;
            prev_dc = 0;
        }
        if scan.is_dc() {
            let value = i32::from(block.dc) >> scan.al;
            if scan.is_refine() {
                counter.raw_bits += 1;
            } else {
                let (size, _) = get_code((value - prev_dc) as i16);
                counter.freq[usize::from(size)] += 1;
                counter.raw_bits += u64::from(size);
                prev_dc = value;
            }
            continue;
        }
        let mut nonzero = block.nonzero_at(scan.al) & band;
        let mut next = ss;
        if !scan.is_refine() {
            // Summed in registers: per-symbol adds to freq[0xF0] and raw_bits
            // would chain each increment on the previous one's store.
            let mut zrl = 0;
            let mut raw_bits = 0;
            while nonzero != 0 {
                let k = nonzero.trailing_zeros() as usize;
                nonzero &= nonzero - 1;
                let run = k - next;
                state.emit_eobrun(&mut counter);
                zrl += run >> 4;
                let size = block.sizes[k] - scan.al;
                let symbol = (((run & 15) << 4) + usize::from(size)) as u8;
                counter.freq[usize::from(symbol)] += 1;
                raw_bits += u64::from(size);
                next = k + 1;
            }
            counter.freq[0xF0] += zrl as u32;
            counter.raw_bits += raw_bits;
            if next <= se {
                state.eobrun += 1;
                if state.eobrun == 0x7FFF {
                    state.emit_eobrun(&mut counter);
                }
            }
            continue;
        }

        // Coefficients that become nonzero at this bit code run/size symbols;
        // the others send a correction bit. A symbol's run counts only the
        // zeros since the previous new coefficient: ZRLs shorten a run at
        // whichever coefficient they are sent, so a gap of `z` zeros codes
        // `z >> 4` ZRLs and run `z & 15`. Correction bits before the last new
        // coefficient go out with the symbols; the rest wait for the EOB run.
        // Visiting only the new coefficients avoids a branch on every one.
        let new = nonzero & !block.nonzero_at(scan.al + 1);
        let old = nonzero & !new;
        let mut waiting = old;
        if new != 0 {
            state.emit_eobrun(&mut counter);
            let eob = 63 - new.leading_zeros();
            let sent = old & (u64::MAX >> (63 - eob));
            waiting &= !sent;
            counter.raw_bits += u64::from(new.count_ones() + sent.count_ones());
            let mut zrl = 0;
            let mut rest = new;
            while rest != 0 {
                let k = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                let between = (u64::MAX << next) & !(u64::MAX << k);
                let zeros = k - next - (old & between).count_ones() as usize;
                zrl += zeros >> 4;
                counter.freq[((zeros & 15) << 4) + 1] += 1;
                next = k + 1;
            }
            counter.freq[0xF0] += zrl as u32;
        }
        if new & (1 << se) == 0 {
            state.eobrun += 1;
            state.pending += waiting.count_ones() as usize;
            if state.eobrun == 0x7FFF || state.pending > MAX_CORRECTION_BITS - 64 {
                state.emit_eobrun(&mut counter);
            }
        }
    }
    state.emit_eobrun(&mut counter);
    counter
}

/// Count several first AC scans of the same `al` in one pass, each exactly as
/// [count_prepared_scan] would.
///
/// Inside a band, a nonzero level after an earlier nonzero of the same band
/// codes the same run/size symbol (and ZRLs) as in any other band holding both,
/// so consecutive nonzero pairs are tallied once by the segments (between band
/// boundaries) of their two positions, and each band sums the segment pairs it
/// contains. Each band still codes its own first nonzero per block (run from
/// `ss`) and its own EOB run state.
fn count_first_scans<B: Levels>(
    blocks: &[B],
    scans: &[Scan],
    restart_interval: usize,
) -> Vec<Counter> {
    let al = scans[0].al;
    debug_assert!(
        scans
            .iter()
            .all(|scan| !scan.is_dc() && !scan.is_refine() && scan.al == al)
    );
    let mut cuts: Vec<usize> = scans
        .iter()
        .flat_map(|scan| [usize::from(scan.ss), usize::from(scan.se) + 1])
        .chain([1, 64])
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    let segments = cuts.len() - 1;
    let mut segment_of = [0usize; 64];
    for (segment, bounds) in cuts.windows(2).enumerate() {
        segment_of[bounds[0]..bounds[1]].fill(segment);
    }
    // Symbol counts of nonzero pairs by (earlier, later) segment. A run/size
    // symbol's low nibble is its raw bits, so they are summed from the counts;
    // only sizes above 15 (carried into the run nibble) add bits per pair.
    let mut pairs = alloc::vec![0u32; segments * segments * 256];
    let mut pair_bits = alloc::vec![0u64; segments * segments];

    // Bands starting at the same `ss` share the first nonzero from `ss`; a band
    // codes it when it is at or before `se`. Each distinct start counts it by
    // (position, size), with position 64 for blocks without one, so no branch
    // depends on whether a band holds a nonzero.
    let mut starts: Vec<usize> = scans.iter().map(|scan| usize::from(scan.ss)).collect();
    starts.sort_unstable();
    starts.dedup();
    let mut firsts = alloc::vec![[0u32; FIRST_SLOTS]; starts.len()];

    let mut masks = Vec::with_capacity(blocks.len());
    for block in blocks {
        let nonzero = block.nonzero_at(al) & !1;
        masks.push(nonzero);

        for (&ss, first) in starts.iter().zip(&mut firsts) {
            let from = nonzero & (u64::MAX << ss);
            let k = from.trailing_zeros() as usize;
            let size = if from != 0 {
                usize::from(block.size(k & 63) - al)
            } else {
                0
            };
            first[k * SIZES + size] += 1;
        }

        let mut rest = nonzero;
        if rest != 0 {
            let mut previous = rest.trailing_zeros() as usize;
            let mut row = segment_of[previous] * segments;
            rest &= rest - 1;
            while rest != 0 {
                let k = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                let run = k - previous - 1;
                let class = row + segment_of[k];
                let size = block.size(k) - al;
                let symbol = (((run & 15) << 4) + usize::from(size)) as u8;
                // Runs of 16 or more are rare; an unconditional add to
                // the ZRL count would chain every pair on the previous store.
                if run > 15 {
                    pairs[class * 256 + 0xF0] += (run >> 4) as u32;
                }
                pairs[class * 256 + usize::from(symbol)] += 1;
                if size > 15 {
                    pair_bits[class] += u64::from(size & !15);
                }
                previous = k;
                row = segment_of[k] * segments;
            }
        }
    }

    let interval = match restart_interval {
        0 => masks.len().max(1),
        n => n,
    };
    scans
        .iter()
        .map(|scan| {
            let mut counter = Counter::new();
            let (ss, se) = (usize::from(scan.ss), usize::from(scan.se));

            let first = &firsts[starts.binary_search(&ss).expect("every start counted")];
            for k in ss..=se {
                let run = k - ss;
                for size in 0..SIZES {
                    let count = first[k * SIZES + size];
                    if count > 0 {
                        let symbol = (((run & 15) << 4) + size) as u8;
                        counter.freq[usize::from(symbol)] += count;
                        counter.freq[0xF0] += (run >> 4) as u32 * count;
                        counter.raw_bits += size as u64 * u64::from(count);
                    }
                }
            }

            let runs = count_eob_runs(&masks, scan, interval);
            for (nbits, &count) in runs.iter().enumerate() {
                counter.freq[nbits << 4] += count;
                counter.raw_bits += nbits as u64 * u64::from(count);
            }
            counter.restarts = masks.len().div_ceil(interval).saturating_sub(1) as u64;

            let first = segment_of[ss];
            let last = segment_of[se];
            for earlier in first..=last {
                for later in earlier..=last {
                    let class = earlier * segments + later;
                    for (symbol, (total, &count)) in counter.freq[..256]
                        .iter_mut()
                        .zip(&pairs[class * 256..(class + 1) * 256])
                        .enumerate()
                    {
                        *total += count;
                        counter.raw_bits += u64::from(count) * (symbol as u64 & 15);
                    }
                    counter.raw_bits += pair_bits[class];
                }
            }
            counter
        })
        .collect()
}

/// Magnitude categories a first-nonzero count tells apart (sizes up to 16)
const SIZES: usize = 17;
/// First-nonzero counts by position 0..=64 (64: none) and size
const FIRST_SLOTS: usize = 65 * SIZES;

/// EOB runs a first AC scan ends, by bit length (EOBn).
///
/// A block without a nonzero in the band, or whose band ends in zeros, adds
/// to the run; a nonzero in the band first ends the run. Up to 64 blocks at a
/// time become two bitsets, blocks with a nonzero and blocks ending in EOB.
/// A nonzero block right after another ends a run of that block's EOB alone,
/// so those are counted together; only nonzero blocks after a gap are visited.
fn count_eob_runs(masks: &[u64], scan: &Scan, interval: usize) -> [u32; 15] {
    let band = (u64::MAX << scan.ss) & (u64::MAX >> (63 - scan.se));
    let se = scan.se;
    let mut runs = [0u32; 15];
    for restart in masks.chunks(interval) {
        let mut eobrun = 0u32;
        for word in restart.chunks(64) {
            let mut nonzero = 0u64;
            let mut eob = 0u64;
            for (i, &mask) in word.iter().enumerate() {
                nonzero |= u64::from(mask & band != 0) << i;
                eob |= ((!mask >> se) & 1) << i;
            }
            if nonzero == 0 {
                eobrun += word.len() as u32;
                if eobrun >= 0x7FFF {
                    runs[14] += 1;
                    eobrun -= 0x7FFF;
                }
                continue;
            }
            // Every block before the first nonzero one adds to the carried run
            let first = nonzero.trailing_zeros();
            eobrun += first;
            if eobrun >= 0x7FFF {
                runs[14] += 1;
                eobrun -= 0x7FFF;
            }
            if eobrun > 0 {
                runs[(31 - eobrun.leading_zeros()) as usize] += 1;
            }
            runs[0] += (nonzero & (nonzero << 1) & (eob << 1)).count_ones();
            // After a gap: the gap's blocks and the EOB of the nonzero one before
            let mut gaps = nonzero & !(nonzero << 1) & !(1 << first);
            while gaps != 0 {
                let i = gaps.trailing_zeros();
                gaps &= gaps - 1;
                let previous = 63 - (nonzero & ((1 << i) - 1)).leading_zeros();
                let run = i - previous - 1 + ((eob >> previous) & 1) as u32;
                runs[(31 - run.leading_zeros()) as usize] += 1;
            }
            let last = 63 - nonzero.leading_zeros();
            eobrun = word.len() as u32 - 1 - last + ((eob >> last) & 1) as u32;
        }
        if eobrun > 0 {
            runs[(31 - eobrun.leading_zeros()) as usize] += 1;
        }
    }
    runs
}

/// Scripts (per component) that the optimizer chooses between
fn candidate_scripts(luma: bool) -> Vec<Vec<Scan>> {
    let partitions: &[&[(u8, u8)]] = if luma {
        &[
            &[(1, 63)],
            &[(1, 5), (6, 63)],
            &[(1, 2), (3, 63)],
            &[(1, 2), (3, 9), (10, 63)],
            &[(1, 8), (9, 63)],
            &[(1, 5), (6, 17), (18, 63)],
            &[(1, 3), (4, 12), (13, 63)],
        ]
    } else {
        &[&[(1, 63)], &[(1, 5), (6, 63)], &[(1, 2), (3, 63)]]
    };
    let depths: &[u8] = if luma { &[0, 1, 2] } else { &[0, 1] };
    // DC successive approximation never won on photos or graphics (54 encodes): a
    // refinement bit per block costs what the coarser first scan saves.
    let mut scripts = Vec::new();
    for dc_depth in [0u8] {
        for &depth in depths {
            for bands in partitions {
                let mut script = Vec::new();
                script.push(Scan::dc(0, dc_depth));
                for &(ss, se) in bands.iter() {
                    script.push(Scan::ac(ss, se, 0, depth));
                }
                let mut al = depth;
                while al > 0 {
                    script.push(Scan::ac(1, 63, al, al - 1));
                    al -= 1;
                }
                if dc_depth > 0 {
                    script.push(Scan::dc(1, 0));
                }
                scripts.push(script);
            }
        }
    }
    scripts
}

/// Spectral selection only: a DC scan and `scans - 1` equal AC bands
fn spectral_script(scans: u8) -> Vec<Scan> {
    let bands = (scans.clamp(2, 64) - 1) as usize;
    let per_band = 63 / bands;
    let mut script = alloc::vec![Scan::dc(0, 0)];
    let mut ss = 1usize;
    for band in 0..bands {
        let se = if band + 1 == bands {
            63
        } else {
            ss + per_band - 1
        };
        script.push(Scan::ac(ss as u8, se as u8, 0, 0));
        ss = se + 1;
    }
    script
}

/// libjpeg's `jpeg_simple_progression` for one component
fn standard_script(luma: bool) -> Vec<Scan> {
    if luma {
        alloc::vec![
            Scan::dc(0, 1),
            Scan::ac(1, 5, 0, 2),
            Scan::ac(6, 63, 0, 2),
            Scan::ac(1, 63, 2, 1),
            Scan::dc(1, 0),
            Scan::ac(1, 63, 1, 0),
        ]
    } else {
        alloc::vec![
            Scan::dc(0, 1),
            Scan::ac(1, 63, 0, 1),
            Scan::dc(1, 0),
            Scan::ac(1, 63, 1, 0),
        ]
    }
}

/// Every `RACE_STRIDE`-th block is costed when choosing a script; the chosen
/// script is then coded exactly. Kodak holdout vs sequential: every block
/// −1.44/−1.50 at 2.4× time, stride 2 −1.39/−1.46 at 1.7×, stride 4
/// −1.31/−1.38 at 1.6×.
const RACE_STRIDE: usize = 2;

/// Choose a script per component: the standard one, or the cheapest candidate on exact bits
pub(crate) fn choose_scripts(
    blocks: &[Vec<AlignedBlock>],
    luma: &[bool],
    restart_interval: usize,
    mode: ProgressiveMode,
) -> Vec<Vec<Scan>> {
    let mut scripts = Vec::new();
    for (component, component_blocks) in blocks.iter().enumerate() {
        match mode {
            ProgressiveMode::Standard => {
                scripts.push(standard_script(luma[component]));
                continue;
            }
            ProgressiveMode::Spectral(scans) => {
                scripts.push(spectral_script(scans));
                continue;
            }
            ProgressiveMode::Optimized => {}
        }
        let sample: Vec<PreparedBlock> = component_blocks
            .iter()
            .step_by(RACE_STRIDE)
            .map(PreparedBlock::new)
            .collect();
        let sampled_restart = restart_interval.div_ceil(RACE_STRIDE);
        let candidates = candidate_scripts(luma[component]);
        let mut costs: BTreeMap<Scan, u64> = BTreeMap::new();
        // First AC scans of one depth share a single pass over the sample.
        let mut first_scans: BTreeMap<u8, Vec<Scan>> = BTreeMap::new();
        for scan in candidates.iter().flatten() {
            if !scan.is_dc() && !scan.is_refine() {
                let group = first_scans.entry(scan.al).or_default();
                if !group.contains(scan) {
                    group.push(*scan);
                }
            }
        }
        for group in first_scans.values() {
            let counters = count_first_scans(&sample, group, sampled_restart);
            for (scan, counter) in group.iter().zip(&counters) {
                costs.insert(*scan, counter.bits(scan));
            }
        }
        let mut best: Option<(u64, Vec<Scan>)> = None;
        for script in candidates {
            let mut total = 0u64;
            for scan in &script {
                let bits = *costs.entry(*scan).or_insert_with(|| {
                    count_prepared_scan(&sample, scan, sampled_restart).bits(scan)
                });
                total += bits;
            }
            if best.as_ref().is_none_or(|(b, _)| total < *b) {
                best = Some((total, script));
            }
        }
        let (_, script) = best.expect("at least one candidate");
        scripts.push(script);
    }
    scripts
}

/// Order the per-component scripts into one scan sequence a decoder accepts:
/// every component's first DC scan comes first, then the rest in script order.
#[cfg(test)]
pub(crate) fn interleave_scripts(scripts: &[Vec<Scan>]) -> Vec<(usize, Scan)> {
    let mut sequence = Vec::new();
    for (c, script) in scripts.iter().enumerate() {
        sequence.push((c, script[0]));
    }
    for (c, script) in scripts.iter().enumerate() {
        for scan in &script[1..] {
            sequence.push((c, *scan));
        }
    }
    sequence
}

/// Write the same scan order while reusing coefficient metadata for counting
/// and emission on components with AC refinements. One component's metadata
/// is live at a time.
pub(crate) fn write_scripts<W: JfifWrite>(
    writer: &mut JfifWriter<W>,
    blocks: &[Vec<AlignedBlock>],
    scripts: &[Vec<Scan>],
    components: &[crate::encoder::Component],
    restart_interval: usize,
) -> Result<(), EncodingError> {
    for (c, script) in scripts.iter().enumerate() {
        write_scan(
            writer,
            &blocks[c],
            &script[0],
            &components[c],
            restart_interval,
        )?;
    }
    for (c, script) in scripts.iter().enumerate() {
        // Disjoint first-scan bands already visit each coefficient once.
        // Refinements revisit them, amortizing preparation across the scans.
        if script.iter().any(|scan| !scan.is_dc() && scan.is_refine()) {
            let prepared: Vec<_> = blocks[c].iter().map(PreparedBlock::new).collect();
            let tables = script_tables(&prepared, &script[1..], restart_interval, |scan| {
                count_prepared_scan(&prepared, scan, restart_interval)
            });
            for (scan, table) in script[1..].iter().zip(&tables) {
                write_scan_with_table(
                    writer,
                    &blocks[c],
                    scan,
                    &components[c],
                    restart_interval,
                    table.as_ref(),
                    Some(&prepared),
                )?;
            }
        } else {
            let tables = script_tables(&blocks[c], &script[1..], restart_interval, |scan| {
                count_scan(&blocks[c], scan, restart_interval)
            });
            for (scan, table) in script[1..].iter().zip(&tables) {
                write_scan_with_table(
                    writer,
                    &blocks[c],
                    scan,
                    &components[c],
                    restart_interval,
                    table.as_ref(),
                    None,
                )?;
            }
        }
    }
    Ok(())
}

/// The Huffman table of every scan that has one. First AC scans of the same
/// `al` share one counting pass; `count` counts any other scan.
fn script_tables<B: Levels>(
    blocks: &[B],
    scans: &[Scan],
    restart_interval: usize,
    count: impl Fn(&Scan) -> Counter,
) -> Vec<Option<HuffmanTable>> {
    let mut tables: Vec<Option<HuffmanTable>> = scans.iter().map(|_| None).collect();
    let mut first_scans: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
    for (i, scan) in scans.iter().enumerate() {
        if !scan.is_dc() && !scan.is_refine() {
            first_scans.entry(scan.al).or_default().push(i);
        } else if scan.has_table() {
            tables[i] = Some(count(scan).table());
        }
    }
    for group in first_scans.values() {
        let group_scans: Vec<Scan> = group.iter().map(|&i| scans[i]).collect();
        let counters = count_first_scans(blocks, &group_scans, restart_interval);
        for (&i, counter) in group.iter().zip(&counters) {
            tables[i] = Some(counter.table());
        }
    }
    tables
}

/// Write one scan: its Huffman table, the scan header and the coded data
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_scan<W: JfifWrite>(
    writer: &mut JfifWriter<W>,
    blocks: &[AlignedBlock],
    scan: &Scan,
    component: &crate::encoder::Component,
    restart_interval: usize,
) -> Result<(), EncodingError> {
    let table = if scan.has_table() {
        Some(count_scan(blocks, scan, restart_interval).table())
    } else {
        None
    };
    write_scan_with_table(
        writer,
        blocks,
        scan,
        component,
        restart_interval,
        table.as_ref(),
        None,
    )
}

fn write_scan_with_table<W: JfifWrite>(
    writer: &mut JfifWriter<W>,
    blocks: &[AlignedBlock],
    scan: &Scan,
    component: &crate::encoder::Component,
    restart_interval: usize,
    table: Option<&HuffmanTable>,
    prepared: Option<&[PreparedBlock]>,
) -> Result<(), EncodingError> {
    if let Some(table) = table {
        let (class, destination) = if scan.is_dc() {
            (CodingClass::Dc, component.dc_huffman_table)
        } else {
            (CodingClass::Ac, component.ac_huffman_table)
        };
        writer.write_huffman_segment(class, destination, table)?;
    }
    writer.write_scan_header(&[component], Some((scan.ss, scan.se)), (scan.ah, scan.al))?;
    let mut emitter = Emitter { writer, table };
    encode_scan(blocks, scan, restart_interval, &mut emitter, prepared)?;
    writer.finalize_bit_buffer()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks() -> Vec<AlignedBlock> {
        (0..40)
            .map(|b| {
                let mut block = AlignedBlock::default();
                for (k, out) in block.data.iter_mut().enumerate() {
                    let v = ((b * 31 + k * 17) % 29) as i16 - 14;
                    *out = if k == 0 {
                        v * 3
                    } else if k < 12 {
                        v
                    } else {
                        v / 7
                    };
                }
                block
            })
            .collect()
    }

    fn check_prepared(blocks: &[AlignedBlock], scans: &[Scan], restart_intervals: &[usize]) {
        #[derive(Default)]
        struct Events(Vec<(u8, u32, u8)>);

        impl Sink for Events {
            fn symbol(&mut self, symbol: u8) -> Result<(), EncodingError> {
                self.0.push((0, u32::from(symbol), 0));
                Ok(())
            }

            fn bits(&mut self, value: u32, count: u8) -> Result<(), EncodingError> {
                self.0.push((1, value, count));
                Ok(())
            }

            fn restart(&mut self, index: u8) -> Result<(), EncodingError> {
                self.0.push((2, u32::from(index), 0));
                Ok(())
            }
        }

        let prepared: Vec<_> = blocks.iter().map(PreparedBlock::new).collect();
        for scan in scans {
            for &restart in restart_intervals {
                let expected = count_scan(blocks, scan, restart);
                let actual = count_prepared_scan(&prepared, scan, restart);
                assert_eq!(actual.freq, expected.freq, "{scan:?}, restart={restart}");
                assert_eq!(
                    actual.raw_bits, expected.raw_bits,
                    "{scan:?}, restart={restart}"
                );
                assert_eq!(
                    actual.restarts, expected.restarts,
                    "{scan:?}, restart={restart}"
                );
                // Compare every symbol, bit value and restart in order against
                // the original coefficient-by-coefficient emitter. Counting
                // alone cannot catch incorrect signs or correction-bit values.
                let mut expected = Events::default();
                let mut actual = Events::default();
                encode_scan(blocks, scan, restart, &mut expected, None).unwrap();
                encode_scan(blocks, scan, restart, &mut actual, Some(&prepared)).unwrap();
                assert_eq!(actual.0, expected.0, "{scan:?}, restart={restart}");
            }
        }
    }

    #[test]
    fn prepared_scans_match_original_emitter() {
        let mut scans = alloc::collections::BTreeSet::new();
        for luma in [true, false] {
            for script in candidate_scripts(luma) {
                scans.extend(script);
            }
            scans.extend(standard_script(luma));
        }
        for al in [0, 1, 2, 4, 10, 13] {
            for (ss, se) in [(1, 1), (1, 63), (7, 32), (16, 63), (63, 63)] {
                scans.insert(Scan::ac(ss, se, 0, al));
                scans.insert(Scan::ac(ss, se, al + 1, al));
            }
        }
        let scans: Vec<_> = scans.into_iter().collect();
        check_prepared(&blocks(), &scans, &[0, 1, 2, 7, 1024]);
        let mut seed = 0x31415926u32;
        let random: Vec<_> = (0..512)
            .map(|b| {
                AlignedBlock::new(core::array::from_fn(|k| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    if k == 0 {
                        // Keep adjacent DC differences inside the existing
                        // coder's signed range; AC still exercises all i16.
                        ((seed >> 16) as i16) / 16
                    } else if b % 4 == 0 {
                        (seed >> 16) as i16
                    } else if seed & 7 == 0 {
                        ((seed >> 16) as i16) >> (b % 12)
                    } else {
                        0
                    }
                }))
            })
            .collect();
        check_prepared(&random, &scans, &[0, 1, 7, 63, 64, 257]);
        let extremes: Vec<_> = (1..64)
            .map(|k| {
                let mut block = AlignedBlock::default();
                block.data[k] = i16::MIN;
                block
            })
            .collect();
        check_prepared(&extremes, &scans, &[0, 1, 7]);
    }

    #[derive(Default, Debug, PartialEq)]
    struct Recorded(Vec<(u8, u32, u8)>);

    impl Sink for Recorded {
        fn symbol(&mut self, symbol: u8) -> Result<(), EncodingError> {
            self.0.push((0, u32::from(symbol), 0));
            Ok(())
        }

        fn bits(&mut self, value: u32, count: u8) -> Result<(), EncodingError> {
            self.0.push((1, value, count));
            Ok(())
        }

        fn restart(&mut self, index: u8) -> Result<(), EncodingError> {
            self.0.push((2, u32::from(index), 0));
            Ok(())
        }
    }

    #[test]
    fn sparse_first_ac_coder_matches_original_events() {
        let mut blocks: Vec<AlignedBlock> = crate::writer::sparse_scan_blocks(true)
            .into_iter()
            .map(AlignedBlock::new)
            .collect();
        blocks.extend(self::blocks());
        // Long all-zero stretches reach the 0x7FFF EOB-run limit.
        blocks.extend(core::iter::repeat_n(AlignedBlock::default(), 33_000));
        let mut tail = AlignedBlock::default();
        tail.data[63] = -9;
        blocks.push(tail);
        let mut scans = Vec::new();
        for al in [0, 1, 2, 5, 13] {
            for (ss, se) in [
                (1, 63),
                (1, 5),
                (6, 63),
                (1, 1),
                (63, 63),
                (7, 32),
                (17, 40),
            ] {
                scans.push(Scan::ac(ss, se, 0, al));
            }
        }
        for scan in &scans {
            let mut fast = (AcState::new(), Recorded::default());
            let mut reference = (AcState::new(), Recorded::default());
            for block in &blocks {
                encode_ac_first(block, scan, &mut fast.0, &mut fast.1).unwrap();
                reference_encode_ac_first(block, scan, &mut reference.0, &mut reference.1).unwrap();
            }
            fast.0.emit_eobrun(&mut fast.1).unwrap();
            reference.0.emit_eobrun(&mut reference.1).unwrap();
            assert_eq!(fast.1, reference.1, "{scan:?}");
        }
    }

    #[test]
    fn branch_free_prepared_masks_match_per_coefficient_loop() {
        let mut blocks: Vec<AlignedBlock> = crate::writer::sparse_scan_blocks(true)
            .into_iter()
            .map(AlignedBlock::new)
            .collect();
        blocks.extend(self::blocks());
        for block in &blocks {
            let prepared = PreparedBlock::new(block);
            let mut expected = [0u64; 3];
            for (k, &size) in prepared.sizes.iter().enumerate().skip(1) {
                for (al, mask) in expected.iter_mut().enumerate() {
                    if usize::from(size) > al {
                        *mask |= 1 << k;
                    }
                }
            }
            assert_eq!(prepared.nonzero, expected, "{:?}", block.data);
            for al in 0..16u8 {
                let significant = crate::writer::significant_mask(&block.data, al) & !1;
                assert_eq!(prepared.nonzero_at(al), significant, "al={al}");
            }
        }
    }

    /// Every first AC scan of a shared pass must count exactly as on its own.
    fn check_first_scans(blocks: &[AlignedBlock], groups: &[Vec<Scan>], restarts: &[usize]) {
        let prepared: Vec<_> = blocks.iter().map(PreparedBlock::new).collect();
        for group in groups {
            for &restart in restarts {
                let shared = count_first_scans(&prepared, group, restart);
                // Unprepared blocks feed the same pass when writing scripts
                // without refinements; it must match the emitter's count.
                let raw = count_first_scans(blocks, group, restart);
                for ((scan, actual), unprepared) in group.iter().zip(&shared).zip(&raw) {
                    let expected = count_prepared_scan(&prepared, scan, restart);
                    assert_eq!(actual.freq, expected.freq, "{scan:?}, restart={restart}");
                    assert_eq!(actual.raw_bits, expected.raw_bits, "{scan:?}");
                    assert_eq!(actual.restarts, expected.restarts, "{scan:?}");
                    assert_eq!(actual.bits(scan), expected.bits(scan), "{scan:?}");
                    let emitted = count_scan(blocks, scan, restart);
                    assert_eq!(unprepared.freq, emitted.freq, "{scan:?}, restart={restart}");
                    assert_eq!(unprepared.raw_bits, emitted.raw_bits, "{scan:?}");
                    assert_eq!(unprepared.restarts, emitted.restarts, "{scan:?}");
                }
            }
        }
    }

    fn first_scan_groups() -> Vec<Vec<Scan>> {
        let mut groups: BTreeMap<u8, Vec<Scan>> = BTreeMap::new();
        for luma in [true, false] {
            for scan in candidate_scripts(luma).into_iter().flatten() {
                if !scan.is_dc() && !scan.is_refine() {
                    let group = groups.entry(scan.al).or_default();
                    if !group.contains(&scan) {
                        group.push(scan);
                    }
                }
            }
        }
        let mut all: Vec<Vec<Scan>> = groups.into_values().collect();
        // Isolated, overlapping and edge bands, deep point transforms
        for al in [0, 1, 3, 10, 13] {
            all.push(
                [
                    (1, 1),
                    (63, 63),
                    (7, 32),
                    (16, 63),
                    (1, 63),
                    (2, 62),
                    (31, 33),
                ]
                .iter()
                .map(|&(ss, se)| Scan::ac(ss, se, 0, al))
                .collect(),
            );
        }
        all.push(alloc::vec![Scan::ac(5, 9, 0, 0)]);
        all
    }

    #[test]
    fn shared_first_scan_counts_match_single_scans() {
        let groups = first_scan_groups();
        check_first_scans(&blocks(), &groups, &[0, 1, 2, 7, 1024]);
        let mut seed = 0x2718_2818u32;
        let random: Vec<_> = (0..768)
            .map(|b| {
                AlignedBlock::new(core::array::from_fn(|k| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    if k == 0 {
                        ((seed >> 16) as i16) / 16
                    } else if b % 5 == 0 {
                        (seed >> 16) as i16
                    } else if (seed >> 8) % 8 < (b % 9) as u32 {
                        ((seed >> 16) as i16) >> (b % 13)
                    } else {
                        0
                    }
                }))
            })
            .collect();
        check_first_scans(&random, &groups, &[0, 1, 7, 63, 64, 257]);
        let extremes: Vec<_> = (1..64)
            .map(|k| {
                let mut block = AlignedBlock::default();
                block.data[k] = i16::MIN;
                block.data[64 - k] = 1;
                block
            })
            .collect();
        check_first_scans(&extremes, &groups, &[0, 1, 7]);
        // EOB runs reaching 0x7FFF, alone and between restarts
        let mut sparse = alloc::vec![AlignedBlock::default(); 32770];
        sparse[32769].data[40] = 3;
        sparse[100].data[2] = -1;
        check_first_scans(&sparse, &groups[..3], &[0, 32767, 32768]);
    }

    #[test]
    fn prepared_scans_preserve_eob_and_correction_buffer_flushes() {
        let scans = [
            Scan::ac(1, 63, 0, 0),
            Scan::ac(1, 63, 1, 0),
            Scan::ac(1, 63, 2, 1),
        ];
        let zeros = alloc::vec![AlignedBlock::default(); 32770];
        check_prepared(&zeros, &scans, &[0, 32767, 32768]);
        let mut old = AlignedBlock::new([4; 64]);
        old.data[0] = 0;
        let mut blocks = alloc::vec![old; 100];
        // New coefficients after long zero runs must emit ZRLs at the same
        // positions relative to buffered correction bits and EOB runs.
        for (b, block) in blocks.iter_mut().enumerate().step_by(3) {
            block.data[1..49].fill(0);
            block.data[63] = if b % 2 == 0 { 1 } else { -2 };
        }
        check_prepared(&blocks, &scans, &[0, 1, 15, 16, 17, 99]);
    }

    #[test]
    fn counted_bits_match_written_bits() {
        let blocks = blocks();
        for scan in [
            Scan::dc(0, 1),
            Scan::dc(1, 0),
            Scan::ac(1, 5, 0, 2),
            Scan::ac(6, 63, 0, 2),
            Scan::ac(1, 63, 2, 1),
            Scan::ac(1, 63, 1, 0),
            Scan::ac(1, 63, 0, 0),
        ] {
            let counter = count_scan(&blocks, &scan, 7);
            let mut out = Vec::new();
            let mut writer = JfifWriter::new(&mut out);
            let component = crate::encoder::Component {
                id: 0,
                quantization_table: 0,
                dc_huffman_table: 0,
                ac_huffman_table: 0,
                horizontal_sampling_factor: 1,
                vertical_sampling_factor: 1,
            };
            write_scan(&mut writer, &blocks, &scan, &component, 7).unwrap();
            let written_bits = out.len() as u64 * 8;
            let counted = counter.bits(&scan);
            // The count pads every restart by a full byte and ignores byte stuffing.
            assert!(
                written_bits + 7 * 6 >= counted
                    && written_bits < counted + out.len() as u64 / 4 + 64,
                "{scan:?}: counted {counted} written {written_bits}"
            );
        }
    }

    #[test]
    fn candidates_are_consistent_scripts() {
        for luma in [true, false] {
            for script in candidate_scripts(luma) {
                let mut bit = [u8::MAX; 64];
                for scan in &script {
                    for k in scan.ss..=scan.se {
                        if scan.ah == 0 {
                            assert_eq!(bit[k as usize], u8::MAX, "{script:?}");
                        } else {
                            assert_eq!(bit[k as usize], scan.ah, "{script:?}");
                        }
                        bit[k as usize] = scan.al;
                    }
                }
                assert!(bit.iter().all(|&b| b == 0), "{script:?}");
            }
        }
    }

    #[test]
    fn prepared_final_tables_preserve_original_scan_bytes() {
        for used in [1, 3, 4] {
            let components: Vec<_> = (0..used)
                .map(|c| crate::encoder::Component {
                    id: c as u8,
                    quantization_table: if c % 3 == 0 { 0 } else { 1 },
                    dc_huffman_table: if c % 3 == 0 { 0 } else { 1 },
                    ac_huffman_table: if c % 3 == 0 { 0 } else { 1 },
                    horizontal_sampling_factor: 1,
                    vertical_sampling_factor: 1,
                })
                .collect();
            let source: Vec<Vec<AlignedBlock>> = (0..used)
                .map(|c| {
                    blocks()
                        .into_iter()
                        .enumerate()
                        .map(|(b, mut block)| {
                            block.data[0] += c as i16 * 17;
                            if (b + c) % 7 == 0 {
                                block.data[1..].fill(0);
                            }
                            block
                        })
                        .collect()
                })
                .collect();
            let luma: Vec<_> = components
                .iter()
                .map(|c| c.quantization_table == 0)
                .collect();
            for mode in [
                ProgressiveMode::Optimized,
                ProgressiveMode::Standard,
                ProgressiveMode::Spectral(2),
                ProgressiveMode::Spectral(4),
            ] {
                for restart in [0, 1, 7] {
                    let scripts = choose_scripts(&source, &luma, restart, mode);
                    let mut expected = Vec::new();
                    let mut writer = JfifWriter::new(&mut expected);
                    for (c, scan) in interleave_scripts(&scripts) {
                        write_scan(&mut writer, &source[c], &scan, &components[c], restart)
                            .unwrap();
                    }
                    let mut actual = Vec::new();
                    write_scripts(
                        &mut JfifWriter::new(&mut actual),
                        &source,
                        &scripts,
                        &components,
                        restart,
                    )
                    .unwrap();
                    assert_eq!(
                        actual, expected,
                        "components={used}, {mode:?}, restart={restart}"
                    );
                }
            }
        }
    }
}
