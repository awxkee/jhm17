/*
 * JPEG arithmetic coding, ported and adapted to Rust from IJG/libjpeg-turbo's
 * jcarith.c and jaricom.c (separate register state, indexed statistics bins,
 * existing jhm17 block orders and scan scripts, fallible Rust output).
 * Developed 1997-2009 by Guido Vollbeding.
 * libjpeg-turbo modifications:
 * Copyright (C) 2015, 2018, 2021-2022, D. R. Commander.
 * For conditions of distribution and use, see LICENSE-IJG.
 */
//! T.81 Annex D QM coding and sequential/progressive coefficient contexts.
//! Conditioning uses the JPEG defaults: DC L=0, U=1 and AC K=5.

use crate::EncodingError;
use crate::encoder::{AlignedBlock, CodingOrder, Component};
use crate::marker::Marker;
use crate::progressive::{self, ProgressiveMode, Scan};
use crate::writer::{JfifWrite, JfifWriter};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const AC_BASE: usize = 2 * 64;
const FIXED: usize = AC_BASE + 2 * 256;

// Table D.2: Qe in the high half, next MPS state in bits 8..15, and next
// LPS state (including the MPS switch flag) in bits 0..7.
static QM: [u32; 114] = [
    0x5a1d0181, 0x2586020e, 0x11140310, 0x080b0412, 0x03d80514, 0x01da0617, 0x00e50719, 0x006f081c,
    0x0036091e, 0x001a0a21, 0x000d0b23, 0x00060c09, 0x00030d0a, 0x00010d0c, 0x5a7f0f8f, 0x3f251024,
    0x2cf21126, 0x207c1227, 0x17b91328, 0x1182142a, 0x0cef152b, 0x09a1162d, 0x072f172e, 0x055c1830,
    0x04061931, 0x03031a33, 0x02401b34, 0x01b11c36, 0x01441d38, 0x00f51e39, 0x00b71f3b, 0x008a203c,
    0x0068213e, 0x004e223f, 0x003b2320, 0x002c0921, 0x5ae125a5, 0x484c2640, 0x3a0d2741, 0x2ef12843,
    0x261f2944, 0x1f332a45, 0x19a82b46, 0x15182c48, 0x11772d49, 0x0e742e4a, 0x0bfb2f4b, 0x09f8304d,
    0x0861314e, 0x0706324f, 0x05cd3330, 0x04de3432, 0x040f3532, 0x03633633, 0x02d43734, 0x025c3835,
    0x01f83936, 0x01a43a37, 0x01603b38, 0x01253c39, 0x00f63d3a, 0x00cb3e3b, 0x00ab3f3d, 0x008f203d,
    0x5b1241c1, 0x4d044250, 0x412c4351, 0x37d84452, 0x2fe84553, 0x293c4654, 0x23794756, 0x1edf4857,
    0x1aa94957, 0x174e4a48, 0x14244b48, 0x119c4c4a, 0x0f6b4d4a, 0x0d514e4b, 0x0bb64f4d, 0x0a40304d,
    0x583251d0, 0x4d1c5258, 0x438e5359, 0x3bdd545a, 0x34ee555b, 0x2eae565c, 0x299a575d, 0x25164756,
    0x557059d8, 0x4ca95a5f, 0x44d95b60, 0x3e225c61, 0x38245d63, 0x32b45e63, 0x2e17565d, 0x56a860df,
    0x4f466165, 0x47e56266, 0x41cf6367, 0x3c3d6468, 0x375e5d63, 0x52316669, 0x4c0f676a, 0x4639686b,
    0x415e6367, 0x56276ae9, 0x50e76b6c, 0x4b85676d, 0x55976d6e, 0x504f6b6f, 0x5a106fee, 0x55226d70,
    0x59eb6ff0, 0x5a1d7171,
];

struct ArithmeticEncoder<'a, W: JfifWrite> {
    writer: &'a mut JfifWriter<W>,
    c: u32,
    a: u32,
    stacked_ff: usize,
    pending_zeros: usize,
    ct: u8,
    buffer: i16,
    statistics: [u8; FIXED + 1],
    last_dc: [i32; 4],
    dc_context: [usize; 4],
}

impl<'a, W: JfifWrite> ArithmeticEncoder<'a, W> {
    fn new(writer: &'a mut JfifWriter<W>) -> Self {
        let mut encoder = Self {
            writer,
            c: 0,
            a: 0x10000,
            stacked_ff: 0,
            pending_zeros: 0,
            ct: 11,
            buffer: -1,
            statistics: [0; FIXED + 1],
            last_dc: [0; 4],
            dc_context: [0; 4],
        };
        encoder.statistics[FIXED] = 113;
        encoder
    }

    fn zeros(&mut self) -> Result<(), EncodingError> {
        while self.pending_zeros > 0 {
            self.writer.write_u8(0)?;
            self.pending_zeros -= 1;
        }
        Ok(())
    }

    fn stuffed(&mut self, byte: u8) -> Result<(), EncodingError> {
        self.writer.write_u8(byte)?;
        if byte == 255 {
            self.writer.write_u8(0)?;
        }
        Ok(())
    }

    fn pending(&mut self) -> Result<(), EncodingError> {
        if self.buffer == 0 {
            self.pending_zeros += 1;
        } else if self.buffer >= 0 {
            self.zeros()?;
            self.writer.write_u8(self.buffer as u8)?;
        }
        if self.stacked_ff > 0 {
            self.zeros()?;
            while self.stacked_ff > 0 {
                self.stuffed(255)?;
                self.stacked_ff -= 1;
            }
        }
        Ok(())
    }

    fn carry(&mut self) -> Result<(), EncodingError> {
        if self.buffer >= 0 {
            self.zeros()?;
            self.stuffed((self.buffer + 1) as u8)?;
        }
        self.pending_zeros += self.stacked_ff;
        self.stacked_ff = 0;
        Ok(())
    }

    #[inline]
    fn decision(&mut self, bin: usize, value: bool) -> Result<(), EncodingError> {
        let state = self.statistics[bin];
        let entry = QM[usize::from(state & 127)];
        let qe = entry >> 16;
        self.a -= qe;
        if u8::from(value) != state >> 7 {
            if self.a >= qe {
                self.c += self.a;
                self.a = qe;
            }
            self.statistics[bin] = (state & 128) ^ entry as u8;
        } else {
            if self.a >= 0x8000 {
                return Ok(());
            }
            if self.a < qe {
                self.c += self.a;
                self.a = qe;
            }
            self.statistics[bin] = (state & 128) ^ (entry >> 8) as u8;
        }
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                let byte = self.c >> 19;
                if byte > 255 {
                    self.carry()?;
                    self.buffer = (byte & 255) as i16;
                } else if byte == 255 {
                    self.stacked_ff += 1;
                } else {
                    self.pending()?;
                    self.buffer = byte as i16;
                }
                self.c &= 0x7ffff;
                self.ct = 8;
            }
            if self.a >= 0x8000 {
                return Ok(());
            }
        }
    }

    // Annex D.1.8 termination, discarding final zero bytes ("Pacman").
    fn finish_interval(&mut self) -> Result<(), EncodingError> {
        let end = (self.a - 1 + self.c) & 0xffff0000;
        self.c = if end < self.c { end + 0x8000 } else { end };
        self.c <<= self.ct;
        if self.c & 0xf8000000 != 0 {
            self.carry()?;
        } else {
            self.pending()?;
        }
        if self.c & 0x07fff800 != 0 {
            self.zeros()?;
            self.stuffed((self.c >> 19) as u8)?;
            if self.c & 0x0007f800 != 0 {
                self.stuffed((self.c >> 11) as u8)?;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(), EncodingError> {
        self.finish_interval()
    }

    fn restart(&mut self, index: u8) -> Result<(), EncodingError> {
        self.finish_interval()?;
        self.writer.write_marker(Marker::RST(index))?;
        self.c = 0;
        self.a = 0x10000;
        self.stacked_ff = 0;
        self.pending_zeros = 0;
        self.ct = 11;
        self.buffer = -1;
        self.statistics.fill(0);
        self.statistics[FIXED] = 113;
        self.last_dc.fill(0);
        self.dc_context.fill(0);
        Ok(())
    }

    fn dc_first(
        &mut self,
        component: usize,
        table: u8,
        value: i16,
        al: u8,
    ) -> Result<(), EncodingError> {
        let base = usize::from(table) * 64;
        let mut st = base + self.dc_context[component];
        let dc = i32::from(value) >> al;
        let diff = dc - self.last_dc[component];
        if diff == 0 {
            self.decision(st, false)?;
            self.dc_context[component] = 0;
            return Ok(());
        }
        self.last_dc[component] = dc;
        self.decision(st, true)?;
        self.decision(st + 1, diff < 0)?;
        st += if diff < 0 { 3 } else { 2 };
        self.dc_context[component] = if diff < 0 { 8 } else { 4 };
        let value = diff.unsigned_abs() - 1;
        let mut magnitude = 0;
        if value != 0 {
            self.decision(st, true)?;
            magnitude = 1;
            let mut rest = value >> 1;
            st = base + 20;
            while rest != 0 {
                self.decision(st, true)?;
                magnitude <<= 1;
                st += 1;
                rest >>= 1;
            }
        }
        self.decision(st, false)?;
        // Default conditioning L=0, U=1: the lower threshold is zero.
        if magnitude > 1 {
            self.dc_context[component] += 8;
        }
        st += 14;
        magnitude >>= 1;
        while magnitude != 0 {
            self.decision(st, value & magnitude != 0)?;
            magnitude >>= 1;
        }
        Ok(())
    }

    fn ac_first(
        &mut self,
        table: u8,
        block: &AlignedBlock,
        ss: u8,
        se: u8,
        al: u8,
    ) -> Result<(), EncodingError> {
        let base = AC_BASE + usize::from(table) * 256;
        let (ss, se) = (usize::from(ss), usize::from(se));
        let end = (ss..=se).rfind(|&k| block.data[k].unsigned_abs() >> al != 0);
        let mut k = ss;
        while end.is_some_and(|end| k <= end) {
            let mut st = base + 3 * (k - 1);
            self.decision(st, false)?;
            while block.data[k].unsigned_abs() >> al == 0 {
                self.decision(st + 1, false)?;
                st += 3;
                k += 1;
            }
            self.decision(st + 1, true)?;
            self.decision(FIXED, block.data[k] < 0)?;
            let value = u32::from(block.data[k].unsigned_abs() >> al) - 1;
            st += 2;
            let mut magnitude = 0;
            if value != 0 {
                self.decision(st, true)?;
                magnitude = 1;
                let mut rest = value >> 1;
                if rest != 0 {
                    self.decision(st, true)?;
                    magnitude <<= 1;
                    st = base + if k <= 5 { 189 } else { 217 };
                    rest >>= 1;
                    while rest != 0 {
                        self.decision(st, true)?;
                        magnitude <<= 1;
                        st += 1;
                        rest >>= 1;
                    }
                }
            }
            self.decision(st, false)?;
            st += 14;
            magnitude >>= 1;
            while magnitude != 0 {
                self.decision(st, value & magnitude != 0)?;
                magnitude >>= 1;
            }
            k += 1;
        }
        if k <= se {
            self.decision(base + 3 * (k - 1), true)?;
        }
        Ok(())
    }

    fn ac_refine(
        &mut self,
        table: u8,
        block: &AlignedBlock,
        scan: Scan,
    ) -> Result<(), EncodingError> {
        let base = AC_BASE + usize::from(table) * 256;
        let (ss, se) = (usize::from(scan.ss), usize::from(scan.se));
        let end = (ss..=se).rfind(|&k| block.data[k].unsigned_abs() >> scan.al != 0);
        let previous_end = (ss..=se).rfind(|&k| block.data[k].unsigned_abs() >> scan.ah != 0);
        let mut k = ss;
        while end.is_some_and(|end| k <= end) {
            let mut st = base + 3 * (k - 1);
            if previous_end.is_none_or(|end| k > end) {
                self.decision(st, false)?;
            }
            loop {
                let value = block.data[k].unsigned_abs() >> scan.al;
                if value != 0 {
                    if value > 1 {
                        self.decision(st + 2, value & 1 != 0)?;
                    } else {
                        self.decision(st + 1, true)?;
                        self.decision(FIXED, block.data[k] < 0)?;
                    }
                    break;
                }
                self.decision(st + 1, false)?;
                st += 3;
                k += 1;
            }
            k += 1;
        }
        if k <= se {
            self.decision(base + 3 * (k - 1), true)?;
        }
        Ok(())
    }

    fn sequential(
        &mut self,
        index: usize,
        component: &Component,
        block: &AlignedBlock,
    ) -> Result<(), EncodingError> {
        self.dc_first(index, component.dc_huffman_table, block.data[0], 0)?;
        self.ac_first(component.ac_huffman_table, block, 1, 63, 0)
    }

    fn progressive(
        &mut self,
        component: &Component,
        block: &AlignedBlock,
        scan: Scan,
    ) -> Result<(), EncodingError> {
        match (scan.ss == 0, scan.ah == 0) {
            (true, true) => self.dc_first(0, component.dc_huffman_table, block.data[0], scan.al),
            (true, false) => self.decision(FIXED, (block.data[0] >> scan.al) & 1 != 0),
            (false, true) => {
                self.ac_first(component.ac_huffman_table, block, scan.ss, scan.se, scan.al)
            }
            (false, false) => self.ac_refine(component.ac_huffman_table, block, scan),
        }
    }
}

pub(crate) fn write_sequential<W: JfifWrite>(
    writer: &mut JfifWriter<W>,
    blocks: &[Vec<AlignedBlock>; 4],
    orders: &[CodingOrder; 4],
    components: &[Component],
    restart_interval: usize,
    interleaved: bool,
) -> Result<(), EncodingError> {
    if interleaved {
        writer.write_scan_header(&components.iter().collect::<Vec<_>>(), None, (0, 0))?;
        let per_mcu: Vec<_> = components
            .iter()
            .map(|c| {
                usize::from(c.horizontal_sampling_factor) * usize::from(c.vertical_sampling_factor)
            })
            .collect();
        let mcus = orders[0].indices.len() / per_mcu[0];
        let mut encoder = ArithmeticEncoder::new(writer);
        for mcu in 0..mcus {
            if restart_interval != 0 && mcu != 0 && mcu.is_multiple_of(restart_interval) {
                encoder.restart(((mcu / restart_interval - 1) & 7) as u8)?;
            }
            for (i, component) in components.iter().enumerate() {
                for &index in &orders[i].indices[mcu * per_mcu[i]..(mcu + 1) * per_mcu[i]] {
                    encoder.sequential(i, component, &blocks[i][index as usize])?;
                }
            }
        }
        encoder.finish()
    } else {
        for (i, component) in components.iter().enumerate() {
            writer.write_scan_header(&[component], None, (0, 0))?;
            let mut encoder = ArithmeticEncoder::new(writer);
            for (mcu, &index) in orders[i].indices.iter().enumerate() {
                if restart_interval != 0 && mcu != 0 && mcu.is_multiple_of(restart_interval) {
                    encoder.restart(((mcu / restart_interval - 1) & 7) as u8)?;
                }
                encoder.sequential(i, component, &blocks[i][index as usize])?;
            }
            encoder.finish()?;
        }
        Ok(())
    }
}

fn write_scan<W: JfifWrite>(
    writer: &mut JfifWriter<W>,
    blocks: &[AlignedBlock],
    component: &Component,
    scan: Scan,
    restart_interval: usize,
) -> Result<(), EncodingError> {
    writer.write_scan_header(&[component], Some((scan.ss, scan.se)), (scan.ah, scan.al))?;
    let mut encoder = ArithmeticEncoder::new(writer);
    for (mcu, block) in blocks.iter().enumerate() {
        if restart_interval != 0 && mcu != 0 && mcu.is_multiple_of(restart_interval) {
            encoder.restart(((mcu / restart_interval - 1) & 7) as u8)?;
        }
        encoder.progressive(component, block, scan)?;
    }
    encoder.finish()
}

pub(crate) fn write_scripts<W: JfifWrite>(
    writer: &mut JfifWriter<W>,
    blocks: &[Vec<AlignedBlock>],
    scripts: &[Vec<Scan>],
    components: &[Component],
    restart_interval: usize,
) -> Result<(), EncodingError> {
    for (c, script) in scripts.iter().enumerate() {
        write_scan(
            writer,
            &blocks[c],
            &components[c],
            script[0],
            restart_interval,
        )?;
    }
    for (c, script) in scripts.iter().enumerate() {
        for &scan in &script[1..] {
            write_scan(writer, &blocks[c], &components[c], scan, restart_interval)?;
        }
    }
    Ok(())
}

#[derive(Default)]
struct ByteCounter(u64);

impl JfifWrite for &mut ByteCounter {
    fn write_all(&mut self, bytes: &[u8]) -> Result<(), EncodingError> {
        self.0 += bytes.len() as u64;
        Ok(())
    }
}

pub(crate) fn choose_scripts(
    blocks: &[Vec<AlignedBlock>],
    components: &[Component],
    restart_interval: usize,
    mode: ProgressiveMode,
) -> Vec<Vec<Scan>> {
    blocks
        .iter()
        .zip(components)
        .map(|(blocks, component)| {
            let luma = component.quantization_table == 0;
            match mode {
                ProgressiveMode::Standard => return progressive::standard_script(luma),
                ProgressiveMode::Spectral(scans) => return progressive::spectral_script(scans),
                ProgressiveMode::Optimized => {}
            }
            // Like the Huffman race, use every second block to choose the layout;
            // the cost here is actual arithmetic bytes, including stuffing and RSTs.
            let sample: Vec<_> = blocks.iter().step_by(2).copied().collect();
            let interval = restart_interval.div_ceil(2);
            let mut costs = BTreeMap::new();
            progressive::candidate_scripts(luma)
                .into_iter()
                .map(|script| {
                    let bits: u64 = script
                        .iter()
                        .map(|&scan| {
                            *costs.entry(scan).or_insert_with(|| {
                                let mut count = ByteCounter::default();
                                write_scan(
                                    &mut JfifWriter::new(&mut count),
                                    &sample,
                                    component,
                                    scan,
                                    interval,
                                )
                                .expect("counting output cannot fail");
                                count.0
                            })
                        })
                        .sum();
                    (bits, script)
                })
                .min_by_key(|(bits, _)| *bits)
                .expect("at least one candidate")
                .1
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component() -> Component {
        Component {
            id: 1,
            quantization_table: 0,
            dc_huffman_table: 0,
            ac_huffman_table: 0,
            horizontal_sampling_factor: 1,
            vertical_sampling_factor: 1,
        }
    }

    // Also generated by tests/support/arithmetic_reference.c using libjpeg-turbo.
    // This spans dense/sparse blocks, signs, magnitude categories and DC contexts.
    fn fixture(zeros: bool) -> Vec<AlignedBlock> {
        (0..80)
            .map(|block| {
                AlignedBlock::new(core::array::from_fn(|k| {
                    if zeros {
                        return 0;
                    }
                    if k == 0 {
                        return (block * 173 % 2048) as i16 - 1024;
                    }
                    match block % 6 {
                        0 => 0,
                        1 => i16::from(k == 1),
                        2 => -i16::from(k == 63),
                        3 => {
                            if k % 17 == block % 17 {
                                if k % 2 == 0 { 2 } else { -3 }
                            } else {
                                0
                            }
                        }
                        4 => ((block * 31 + k * 73) % 2047) as i16 - 1023,
                        _ => (if k % 2 == 0 { 1 } else { -1 }) * (1 << ((block + k) % 10)),
                    }
                }))
            })
            .collect()
    }

    fn signature(bytes: &[u8]) -> (usize, u64) {
        (
            bytes.len(),
            bytes.iter().fold(0xcbf29ce484222325u64, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            }),
        )
    }

    #[test]
    fn sequential_matches_libjpeg_reference_vectors() {
        // Length and FNV-1a of the reference entropy bytes, including RSTs.
        // 80 blocks at interval 7 also exercises restart-index wraparound.
        for (restart, zeros, expected) in [
            (0, false, (2599, 0x757db73b506b5108)),
            (0, true, (2, 0x087d8c07b52b6233)),
            (7, false, (3125, 0xb7bee33321056367)),
            (7, true, (34, 0xc94e82099ea45443)),
        ] {
            let mut blocks: [Vec<AlignedBlock>; 4] = Default::default();
            blocks[0] = fixture(zeros);
            let mut orders: [CodingOrder; 4] = Default::default();
            orders[0].indices = (0..80).collect();
            let mut out = Vec::new();
            write_sequential(
                &mut JfifWriter::new(&mut out),
                &blocks,
                &orders,
                &[component()],
                restart,
                true,
            )
            .unwrap();
            assert_eq!(
                signature(&out[10..]),
                expected,
                "restart={restart}, zeros={zeros}"
            );
        }
    }

    #[test]
    fn progressive_matches_libjpeg_reference_vectors() {
        for (restart, zeros, expected) in [
            (
                0,
                false,
                [
                    (85, 0xd1a8557a40b487ad),
                    (147, 0x0a45e6394080f34c),
                    (1557, 0xfb0751f315d312b3),
                    (287, 0xf976365de86b1052),
                    (10, 0xa8545f90388b7ed1),
                    (286, 0x694d6b93fa26ccc3),
                ],
            ),
            (
                0,
                true,
                [
                    (1, 0xaf640d4c86023fcf),
                    (1, 0xaf64254c86026897),
                    (1, 0xaf64254c86026897),
                    (1, 0xaf64254c86026897),
                    (2, 0x0929e007b5bcf3c4),
                    (1, 0xaf64254c86026897),
                ],
            ),
            (
                7,
                false,
                [
                    (157, 0xb7eef762e9b711e5),
                    (204, 0x24c0f232a26bfe1a),
                    (1835, 0x22ae0b27419ebf07),
                    (378, 0xd2ceb6f892cbc5e9),
                    (34, 0xb7af853dd3a1c0ff),
                    (434, 0xcd95ee0e0d5c387e),
                ],
            ),
            (
                7,
                true,
                [
                    (34, 0x4397b3067e31c823),
                    (34, 0xf469b68662b7d403),
                    (34, 0xf469b68662b7d403),
                    (34, 0xf469b68662b7d403),
                    (34, 0x9280750a6b973ed7),
                    (34, 0xf469b68662b7d403),
                ],
            ),
        ] {
            let blocks = fixture(zeros);
            for (scan, expected) in progressive::standard_script(true).into_iter().zip(expected) {
                let mut out = Vec::new();
                write_scan(
                    &mut JfifWriter::new(&mut out),
                    &blocks,
                    &component(),
                    scan,
                    restart,
                )
                .unwrap();
                assert_eq!(
                    signature(&out[10..]),
                    expected,
                    "{scan:?}, restart={restart}, zeros={zeros}"
                );
            }
        }
    }

    #[test]
    fn arithmetic_output_errors_propagate_including_termination() {
        struct Failing(usize);
        impl JfifWrite for Failing {
            fn write_all(&mut self, bytes: &[u8]) -> Result<(), EncodingError> {
                if bytes.len() > self.0 {
                    return Err(EncodingError::Write("injected failure".into()));
                }
                self.0 -= bytes.len();
                Ok(())
            }
        }
        for budget in [0, 7, 100] {
            let mut writer = JfifWriter::new(Failing(budget));
            let mut encoder = ArithmeticEncoder::new(&mut writer);
            let blocks = fixture(false);
            let result = (|| {
                for block in &blocks {
                    encoder.sequential(0, &component(), block)?;
                }
                encoder.finish()
            })();
            assert!(matches!(result, Err(EncodingError::Write(_))));
        }
        let mut writer = JfifWriter::new(Failing(0));
        let mut encoder = ArithmeticEncoder::new(&mut writer);
        encoder.decision(FIXED, true).unwrap();
        assert_eq!(encoder.buffer, -1, "no output before termination");
        assert!(matches!(encoder.finish(), Err(EncodingError::Write(_))));
    }
}
