use crate::EncodingError;
use crate::encoder::{AlignedBlock, Component};
use crate::huffman::{CodingClass, HuffmanTable};
use crate::marker::{Marker, SOFType};
use crate::quantization::QuantizationTable;

/// Represents the pixel density of an image
///
/// For example, a 300 DPI image is represented by:
///
/// ```rust
/// # use jhm17::{PixelDensity, PixelDensityUnit};
/// let hdpi = PixelDensity::dpi(300);
/// assert_eq!(hdpi, PixelDensity {density: (300,300), unit: PixelDensityUnit::Inches})
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelDensity {
    /// A couple of values for (Xdensity, Ydensity)
    pub density: (u16, u16),
    /// The unit in which the density is measured
    pub unit: PixelDensityUnit,
}

impl PixelDensity {
    /// Creates the most common pixel density type:
    /// the horizontal and the vertical density are equal,
    /// and measured in pixels per inch.
    #[must_use]
    pub fn dpi(density: u16) -> Self {
        PixelDensity {
            density: (density, density),
            unit: PixelDensityUnit::Inches,
        }
    }
}

impl Default for PixelDensity {
    /// Returns a pixel density with a pixel aspect ratio of 1
    fn default() -> Self {
        PixelDensity {
            density: (1, 1),
            unit: PixelDensityUnit::PixelAspectRatio,
        }
    }
}

/// Represents a unit in which the density of an image is measured
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelDensityUnit {
    /// Represents the absence of a unit, the values indicate only a
    /// [pixel aspect ratio](https://en.wikipedia.org/wiki/Pixel_aspect_ratio)
    PixelAspectRatio,

    /// Pixels per inch (2.54 cm)
    Inches,

    /// Pixels per centimeter
    Centimeters,
}

/// Zig-zag sequence of quantized DCT coefficients
///
/// Figure A.6
pub static ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const BUFFER_SIZE: usize = core::mem::size_of::<usize>() * 8;

/// Output sink used by the JPEG encoder.
///
/// Implemented for every type that implements [`std::io::Write`].
pub trait JfifWrite {
    /// Writes the whole buffer. The behavior must be identical to std::io::Write::write_all
    /// # Errors
    ///
    /// Return an error if the data can't be written
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError>;
}

impl<W: std::io::Write + ?Sized> JfifWrite for W {
    #[inline(always)]
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.write_all(buf)?;
        Ok(())
    }
}

pub(crate) struct JfifWriter<W: JfifWrite> {
    w: W,
    bit_buffer: usize,
    free_bits: i8,
}

impl<W: JfifWrite> JfifWriter<W> {
    pub fn new(w: W) -> Self {
        JfifWriter {
            w,
            bit_buffer: 0,
            free_bits: BUFFER_SIZE as i8,
        }
    }

    #[inline(always)]
    pub fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.w.write_all(buf)
    }

    #[inline(always)]
    pub fn write_u8(&mut self, value: u8) -> Result<(), EncodingError> {
        self.w.write_all(&[value])
    }

    #[inline(always)]
    pub fn write_u16(&mut self, value: u16) -> Result<(), EncodingError> {
        self.w.write_all(&value.to_be_bytes())
    }

    pub fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError> {
        self.write_bits(0x7F, 7)?;
        self.flush_bit_buffer()?;
        self.bit_buffer = 0;
        self.free_bits = BUFFER_SIZE as i8;

        Ok(())
    }

    pub fn flush_bit_buffer(&mut self) -> Result<(), EncodingError> {
        while self.free_bits <= (BUFFER_SIZE as i8 - 8) {
            self.flush_byte_from_bit_buffer(self.free_bits)?;
            self.free_bits += 8;
        }

        Ok(())
    }

    #[inline(always)]
    fn flush_byte_from_bit_buffer(&mut self, free_bits: i8) -> Result<(), EncodingError> {
        let value = (self.bit_buffer >> (BUFFER_SIZE as i8 - 8 - free_bits)) & 0xFF;

        self.write_u8(value as u8)?;

        if value == 0xFF {
            self.write_u8(0x00)?;
        }

        Ok(())
    }

    #[inline(always)]
    #[allow(overflowing_literals)]
    fn write_bit_buffer(&mut self) -> Result<(), EncodingError> {
        if (self.bit_buffer
            & 0x8080808080808080
            & !self.bit_buffer.wrapping_add(0x0101010101010101))
            != 0
        {
            for i in 0..(BUFFER_SIZE / 8) {
                self.flush_byte_from_bit_buffer((i * 8) as i8)?;
            }
            Ok(())
        } else {
            self.w.write_all(&self.bit_buffer.to_be_bytes())
        }
    }

    #[inline(always)]
    pub fn write_bits(&mut self, value: u32, size: u8) -> Result<(), EncodingError> {
        let size = size as i8;
        let value = value as usize;

        let free_bits = self.free_bits - size;

        if free_bits < 0 {
            self.write_bits_overflow(value, size, free_bits)
        } else {
            self.free_bits = free_bits;
            self.bit_buffer = (self.bit_buffer << size) | value;
            Ok(())
        }
    }

    // Share the full-word flush while keeping ordinary field appends small
    // enough to inline in the sequential and progressive coefficient loops.
    #[inline(never)]
    fn write_bits_overflow(
        &mut self,
        value: usize,
        size: i8,
        free_bits: i8,
    ) -> Result<(), EncodingError> {
        self.bit_buffer = (self.bit_buffer << (size + free_bits)) | (value >> -free_bits);
        self.write_bit_buffer()?;
        self.bit_buffer = value;
        self.free_bits = free_bits + BUFFER_SIZE as i8;
        Ok(())
    }

    pub fn write_marker(&mut self, marker: Marker) -> Result<(), EncodingError> {
        self.write(&[0xFF, marker.into()])
    }

    pub fn write_segment(&mut self, marker: Marker, data: &[u8]) -> Result<(), EncodingError> {
        self.write_marker(marker)?;
        self.write_u16(data.len() as u16 + 2)?;
        self.write(data)?;

        Ok(())
    }

    pub fn write_header(&mut self, density: &PixelDensity) -> Result<(), EncodingError> {
        self.write_marker(Marker::APP(0))?;
        self.write_u16(16)?;

        self.write(b"JFIF\0")?;
        self.write(&[0x01, 0x02])?;

        match density.unit {
            PixelDensityUnit::PixelAspectRatio => {
                self.write_u8(0x00)?;
            }
            PixelDensityUnit::Inches => {
                self.write_u8(0x01)?;
            }
            PixelDensityUnit::Centimeters => {
                self.write_u8(0x02)?;
            }
        }
        let (x, y) = density.density;
        self.write_u16(x)?;
        self.write_u16(y)?;

        self.write(&[0x00, 0x00])
    }

    /// Append huffman table segment
    ///
    /// - `class`: 0 for DC or 1 for AC
    /// - `dest`: 0 for luma or 1 for chroma tables
    ///
    /// Layout:
    /// ```txt
    /// |--------|---------------|--------------------------|--------------------|--------|
    /// | 0xFFC4 | 16 bit length | 4 bit class / 4 bit dest |  16 byte num codes | values |
    /// |--------|---------------|--------------------------|--------------------|--------|
    /// ```
    ///
    pub fn write_huffman_segment(
        &mut self,
        class: CodingClass,
        destination: u8,
        table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        assert!(destination < 4, "Bad destination: {}", destination);

        self.write_marker(Marker::DHT)?;
        self.write_u16(2 + 1 + 16 + table.values().len() as u16)?;

        self.write_u8(((class as u8) << 4) | destination)?;
        self.write(table.length())?;
        self.write(table.values())?;

        Ok(())
    }

    /// Append quantization tables as one segment, table `i` to destination `i`
    ///
    /// - `precision`: 0 which means 1 byte per value.
    /// - `dest`: the table's index in `tables`
    ///
    /// Layout (the descriptor and values repeat for each table):
    /// ```txt
    /// |--------|---------------|------------------------------|--------|--------|-----|--------|
    /// | 0xFFDB | 16 bit length | 4 bit precision / 4 bit dest | V(0,0) | V(0,1) | ... | V(7,7) |
    /// |--------|---------------|------------------------------|--------|--------|-----|--------|
    /// ```
    ///
    pub fn write_quantization_segment(
        &mut self,
        tables: &[QuantizationTable],
    ) -> Result<(), EncodingError> {
        assert!(tables.len() <= 4, "Bad table count: {}", tables.len());

        self.write_marker(Marker::DQT)?;
        self.write_u16(2 + 65 * tables.len() as u16)?;

        for (destination, table) in tables.iter().enumerate() {
            self.write_u8(destination as u8)?;

            for &v in ZIGZAG.iter() {
                self.write_u8(table.get(v as usize))?;
            }
        }

        Ok(())
    }

    pub fn write_dri(&mut self, restart_interval: u16) -> Result<(), EncodingError> {
        self.write_marker(Marker::DRI)?;
        self.write_u16(4)?;
        self.write_u16(restart_interval)
    }

    #[inline]
    pub fn huffman_encode(&mut self, val: u8, table: &HuffmanTable) -> Result<(), EncodingError> {
        let &(size, code) = table.get_for_value(val);
        self.write_bits(code as u32, size)
    }

    #[inline]
    pub fn huffman_encode_value(
        &mut self,
        size: u8,
        symbol: u8,
        value: u16,
        table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let &(num_bits, code) = table.get_for_value(symbol);

        let mut temp = value as u32;
        temp |= (code as u32) << size;
        let size = size + num_bits;

        self.write_bits(temp, size)
    }

    pub fn write_block(
        &mut self,
        block: &AlignedBlock,
        prev_dc: i16,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        self.write_dc(block.data[0], prev_dc, dc_table)?;
        self.write_ac_block(block, 1, 64, ac_table)
    }

    pub fn write_dc(
        &mut self,
        value: i16,
        prev_dc: i16,
        dc_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let diff = value - prev_dc;
        let (size, value) = get_code(diff);

        self.huffman_encode_value(size, size, value, dc_table)?;

        Ok(())
    }

    pub fn write_ac_block(
        &mut self,
        block: &AlignedBlock,
        start: usize,
        end: usize,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        debug_assert!(start <= end && end <= 64);
        if start >= end {
            return Ok(());
        }
        // Visit only nonzero positions in [start, end); gaps are zero runs.
        let range = (u64::MAX << start) & (u64::MAX >> (64 - end));
        let mut mask = nonzero_mask(&block.data) & range;
        let mut next = start;

        while mask != 0 {
            let k = mask.trailing_zeros() as usize;
            let run = k - next;
            for _ in 0..run >> 4 {
                self.huffman_encode(0xF0, ac_table)?;
            }

            let (size, value) = get_code(block.data[k]);
            let symbol = (((run & 15) as u8) << 4) | size;

            self.huffman_encode_value(size, symbol, value, ac_table)?;

            next = k + 1;
            mask &= mask - 1;
        }

        if next < end {
            self.huffman_encode(0x00, ac_table)?;
        }

        Ok(())
    }

    /// The original coefficient-by-coefficient AC emitter.
    #[cfg(test)]
    pub(crate) fn reference_write_ac_block(
        &mut self,
        block: &AlignedBlock,
        start: usize,
        end: usize,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let mut zero_run = 0;

        for &value in &block.data[start..end] {
            if value == 0 {
                zero_run += 1;
            } else {
                while zero_run > 15 {
                    self.huffman_encode(0xF0, ac_table)?;
                    zero_run -= 16;
                }

                let (size, value) = get_code(value);
                let symbol = (zero_run << 4) | size;

                self.huffman_encode_value(size, symbol, value, ac_table)?;

                zero_run = 0;
            }
        }

        if zero_run > 0 {
            self.huffman_encode(0x00, ac_table)?;
        }

        Ok(())
    }

    pub fn write_frame_header(
        &mut self,
        width: u16,
        height: u16,
        components: &[Component],
        progressive: bool,
    ) -> Result<(), EncodingError> {
        if progressive {
            self.write_marker(Marker::SOF(SOFType::ProgressiveDCT))?;
        } else {
            self.write_marker(Marker::SOF(SOFType::BaselineDCT))?;
        }

        self.write_u16(2 + 1 + 2 + 2 + 1 + (components.len() as u16) * 3)?;

        // Precision
        self.write_u8(8)?;

        self.write_u16(height)?;
        self.write_u16(width)?;

        self.write_u8(components.len() as u8)?;

        for component in components.iter() {
            self.write_u8(component.id)?;
            self.write_u8(
                (component.horizontal_sampling_factor << 4) | component.vertical_sampling_factor,
            )?;
            self.write_u8(component.quantization_table)?;
        }

        Ok(())
    }

    pub fn write_scan_header(
        &mut self,
        components: &[&Component],
        spectral: Option<(u8, u8)>,
        approximation: (u8, u8),
    ) -> Result<(), EncodingError> {
        self.write_marker(Marker::SOS)?;

        self.write_u16(2 + 1 + (components.len() as u16) * 2 + 3)?;

        self.write_u8(components.len() as u8)?;

        for component in components.iter() {
            self.write_u8(component.id)?;
            self.write_u8((component.dc_huffman_table << 4) | component.ac_huffman_table)?;
        }

        let (spectral_start, spectral_end) = spectral.unwrap_or((0, 63));

        // Start of spectral or predictor selection
        self.write_u8(spectral_start)?;

        // End of spectral selection
        self.write_u8(spectral_end)?;

        // Successive approximation bit position high and low
        self.write_u8((approximation.0 << 4) | approximation.1)?;

        Ok(())
    }
}

/// Bit `k` is set when `data[k]` is nonzero.
#[inline]
pub(crate) fn nonzero_mask(data: &[i16; 64]) -> u64 {
    pack_flags(&data.map(|v| u8::from(v != 0)))
}

/// Bit `k` is set when `|data[k]| >> al` is nonzero: the coefficients a first
/// progressive scan with point transform `al` codes.
#[inline]
pub(crate) fn significant_mask(data: &[i16; 64], al: u8) -> u64 {
    pack_flags(&data.map(|v| u8::from(u32::from(v.unsigned_abs()) >> al != 0)))
}

/// Bit `k` is set when `flags[k]` is 1; every flag must be 0 or 1.
#[inline]
pub(crate) fn pack_flags(flags: &[u8; 64]) -> u64 {
    // A multiply gathers each group's eight low bits into its top byte (the
    // partial products never overlap).
    let mut mask = 0;
    for (j, group) in flags.as_chunks::<8>().0.iter().enumerate() {
        let bits = u64::from_le_bytes(*group).wrapping_mul(0x0102_0408_1020_4080) >> 56;
        mask |= bits << (8 * j);
    }
    mask
}

#[inline]
pub(crate) fn get_code(value: i16) -> (u8, u16) {
    let temp = value - (value.is_negative() as i16);
    let temp2 = value.abs();

    /*
     * Doing this instead of 16 - temp2.leading_zeros()
     * Gives the compiler the information that leadings_zeros
     * is always called on a non zero value, which removes a branch on x86
     */
    let num_bits = 15 - (temp2 << 1 | 1).leading_zeros() as u16;

    let coefficient = temp & ((1 << num_bits as usize) - 1);

    (num_bits as u8, coefficient as u16)
}

/// Zigzag blocks for the sparse-scan oracles: empty, full, every impulse,
/// every trailing run length, runs across ZRL boundaries and random blocks of
/// several densities with magnitude categories 1–14 (`get_code` overflows at
/// 15 in debug builds; real AC levels stay far below), signs included.
#[cfg(test)]
pub(crate) fn sparse_scan_blocks(include_min: bool) -> alloc::vec::Vec<[i16; 64]> {
    let mut blocks = alloc::vec![[0i16; 64], [1i16; 64], [-1i16; 64], [16383; 64]];
    for k in 0..64 {
        let mut impulse = [0i16; 64];
        impulse[k] = if k % 2 == 0 { 1 } else { -1 };
        blocks.push(impulse);
        let mut prefix = [0i16; 64];
        prefix[..=k].fill(3);
        blocks.push(prefix);
    }
    for gap in [15, 16, 17, 31, 32, 33, 47, 48, 61, 62] {
        let mut block = [0i16; 64];
        block[1] = 5;
        if 2 + gap < 64 {
            block[2 + gap] = -7;
        }
        blocks.push(block);
        let mut tail = [0i16; 64];
        tail[63 - gap.min(62)] = 2;
        tail[63] = -2;
        blocks.push(tail);
    }
    let mut seed = 0x2545_f491u32;
    for i in 0..4096 {
        let mut block = [0i16; 64];
        let density = i % 17;
        for value in &mut block {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            if (seed >> 8) % 16 < density {
                let size = (seed >> 16) % 14;
                let magnitude = ((1u32 << size) | (seed >> 4) & ((1 << size) - 1)) as i16;
                *value = if seed & 1 == 0 { magnitude } else { -magnitude };
                if include_min && (seed >> 12) % 61 == 0 {
                    *value = i16::MIN;
                }
            }
        }
        blocks.push(block);
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn nonzero_mask_matches_per_coefficient_bits() {
        for data in sparse_scan_blocks(true) {
            let expected = data
                .iter()
                .enumerate()
                .fold(0u64, |mask, (k, &v)| mask | (u64::from(v != 0) << k));
            assert_eq!(nonzero_mask(&data), expected, "{data:?}");
        }
    }

    #[test]
    fn sparse_ac_emitter_matches_original_bytes() {
        // Skewed counts give every AC symbol a code with lengths up to 16.
        let mut freq = [0u32; 257];
        for (s, f) in freq.iter_mut().enumerate().take(256) {
            *f = 1 + ((s as u32).wrapping_mul(2_654_435_761) >> 20) % 5000 / (1 + (s as u32 & 15));
        }
        freq[256] = 1;
        let table = HuffmanTable::new_optimized(freq);
        let ranges = [
            (1, 64),
            (0, 64),
            (1, 2),
            (1, 6),
            (6, 64),
            (17, 40),
            (63, 64),
            (9, 9),
        ];
        let blocks = sparse_scan_blocks(false);
        for &(start, end) in &ranges {
            let mut fast = JfifWriter::new(Vec::new());
            let mut reference = JfifWriter::new(Vec::new());
            for (b, data) in blocks.iter().enumerate() {
                let block = AlignedBlock::new(*data);
                fast.write_ac_block(&block, start, end, &table).unwrap();
                reference
                    .reference_write_ac_block(&block, start, end, &table)
                    .unwrap();
                if b % 64 == 0 {
                    fast.flush_bit_buffer().unwrap();
                    reference.flush_bit_buffer().unwrap();
                    assert_eq!(fast.w, reference.w, "range {start}..{end}, block {b}");
                }
            }
            fast.finalize_bit_buffer().unwrap();
            reference.finalize_bit_buffer().unwrap();
            assert_eq!(fast.w, reference.w, "range {start}..{end}");
        }
    }

    #[derive(Default)]
    struct BitReference {
        bytes: Vec<u8>,
        partial: u8,
        used: u8,
    }

    impl BitReference {
        fn bits(&mut self, value: u32, count: u8) {
            for shift in (0..count).rev() {
                self.partial = (self.partial << 1) | ((value >> shift) & 1) as u8;
                self.used += 1;
                if self.used == 8 {
                    self.bytes.push(self.partial);
                    if self.partial == 255 {
                        self.bytes.push(0);
                    }
                    self.partial = 0;
                    self.used = 0;
                }
            }
        }

        fn pad(&mut self) {
            while self.used != 0 {
                self.bits(1, 1);
            }
        }
    }

    #[test]
    fn bit_writer_matches_independent_bit_stream() {
        let mut seed = 0x31415926u32;
        for alignment in 0..BUFFER_SIZE {
            let mut writer = JfifWriter::new(Vec::new());
            let mut reference = BitReference::default();
            for bit in 0..alignment {
                let value = ((alignment + bit) & 1) as u32;
                writer.write_bits(value, 1).unwrap();
                reference.bits(value, 1);
            }
            for i in 0..512 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let count = ((i % 33) as u8).min(BUFFER_SIZE as u8 - 1);
                let mask = if count == 32 {
                    u32::MAX
                } else {
                    (1u32 << count) - 1
                };
                let value = match i % 4 {
                    0 => mask,
                    1 => 0,
                    2 => seed & mask,
                    _ => 0xaa55aa55 & mask,
                };
                writer.write_bits(value, count).unwrap();
                reference.bits(value, count);
                if i % 17 == 0 {
                    writer.flush_bit_buffer().unwrap();
                    assert_eq!(
                        writer.w, reference.bytes,
                        "alignment={alignment}, field={i}"
                    );
                }
                if i % 97 == 0 {
                    writer.finalize_bit_buffer().unwrap();
                    reference.pad();
                    let restart = ((i / 97 + alignment) & 7) as u8;
                    writer.write_marker(Marker::RST(restart)).unwrap();
                    reference.bytes.extend_from_slice(&[255, 0xd0 + restart]);
                    assert_eq!(
                        writer.w, reference.bytes,
                        "restart, alignment={alignment}, field={i}"
                    );
                }
            }
            writer.finalize_bit_buffer().unwrap();
            reference.pad();
            assert_eq!(writer.w, reference.bytes, "final alignment={alignment}");
        }
    }

    #[test]
    fn bit_writer_propagates_word_and_stuffing_errors() {
        struct Failing {
            calls: usize,
        }
        impl JfifWrite for Failing {
            fn write_all(&mut self, _: &[u8]) -> Result<(), EncodingError> {
                self.calls += 1;
                Err(EncodingError::Write("injected failure".into()))
            }
        }
        for bit in [0, 1] {
            let mut writer = JfifWriter::new(Failing { calls: 0 });
            for _ in 0..BUFFER_SIZE {
                writer.write_bits(bit, 1).unwrap();
            }
            assert!(matches!(
                writer.write_bits(bit, 1),
                Err(EncodingError::Write(_))
            ));
            assert_eq!(writer.w.calls, 1);
        }
    }
}
