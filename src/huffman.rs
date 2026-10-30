/*
 * The default huffman tables are taken from
 * section K.3 Typical Huffman tables for 8-bit precision luminance and chrominance
 */

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;

// A tree with 256 symbols and the reserved pseudo-symbol can have depth 256
// before its code lengths are limited to JPEG's 16 bits.
const MAX_CODE_SIZE: usize = 256;

#[derive(Copy, Clone, Debug)]
pub enum CodingClass {
    Dc = 0,
    Ac = 1,
}

static DEFAULT_LUMA_DC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

static DEFAULT_LUMA_DC_VALUES: [u8; 12] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];

static DEFAULT_CHROMA_DC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
];

static DEFAULT_CHROMA_DC_VALUES: [u8; 12] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];

static DEFAULT_LUMA_AC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
];

static DEFAULT_LUMA_AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

static DEFAULT_CHROMA_AC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77,
];

static DEFAULT_CHROMA_AC_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

pub struct HuffmanTable {
    lookup_table: [(u8, u16); 256],
    length: [u8; 16],
    values: Vec<u8>,
}

impl HuffmanTable {
    pub fn new(length: &[u8; 16], values: &[u8]) -> HuffmanTable {
        HuffmanTable {
            lookup_table: create_lookup_table(length, values),
            length: *length,
            values: values.to_vec(),
        }
    }

    pub fn default_luma_dc() -> HuffmanTable {
        Self::new(&DEFAULT_LUMA_DC_CODE_LENGTHS, &DEFAULT_LUMA_DC_VALUES)
    }

    pub fn default_luma_ac() -> HuffmanTable {
        Self::new(&DEFAULT_LUMA_AC_CODE_LENGTHS, &DEFAULT_LUMA_AC_VALUES)
    }

    pub fn default_chroma_dc() -> HuffmanTable {
        Self::new(&DEFAULT_CHROMA_DC_CODE_LENGTHS, &DEFAULT_CHROMA_DC_VALUES)
    }

    pub fn default_chroma_ac() -> HuffmanTable {
        Self::new(&DEFAULT_CHROMA_AC_CODE_LENGTHS, &DEFAULT_CHROMA_AC_VALUES)
    }

    /// Generates an optimized huffman table as described in Section K.2
    #[allow(clippy::needless_range_loop)]
    pub fn new_optimized(mut freq: [u32; 257]) -> HuffmanTable {
        let codesize = optimized_code_sizes(&mut freq);

        // Find the number of codes of each size
        // Figure K.2

        // Intermediate buckets can contain all 256 real symbols; the reserved
        // pseudo-symbol is removed only after length limiting.
        let mut bits = [0u16; MAX_CODE_SIZE + 1];
        let mut max_size = 0;

        for &size in codesize.iter() {
            if size > 0 {
                bits[size] += 1;
                max_size = max_size.max(size);
            }
        }

        // Limiting code lengths to 16 bits
        // Figure K.3

        let mut i = max_size;

        while i > 16 {
            while bits[i] > 0 {
                let mut j = i - 2;
                while bits[j] == 0 {
                    j -= 1;
                }

                bits[i] -= 2;
                bits[i - 1] += 1;
                bits[j + 1] += 2;
                bits[j] -= 1;
            }

            i -= 1;
        }

        while bits[i] == 0 {
            debug_assert!(i > 0, "Error creating codesizes for frequency {:?}", freq);
            i -= 1;
        }
        bits[i] -= 1;

        // Sorting of input values according to code size
        // Figure K.4
        let (huffval, k) = values_by_code_size(&codesize);

        let mut length = [0u8; 16];
        for (i, v) in length.iter_mut().enumerate() {
            *v = u8::try_from(bits[i + 1]).expect("too many Huffman codes of one length");
        }

        let values = huffval[0..k].to_vec();

        HuffmanTable {
            lookup_table: create_lookup_table(&length, &values),
            length,
            values,
        }
    }

    #[inline]
    pub fn get_for_value(&self, value: u8) -> &(u8, u16) {
        let res = &self.lookup_table[value as usize];
        debug_assert!(res.0 > 0, "Got zero size code for value: {}", value);
        res
    }

    pub fn length(&self) -> &[u8; 16] {
        &self.length
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }
}

// Create huffman table code sizes as defined in Figure C.1
fn create_sizes(code_lengths: &[u8; 16]) -> [u8; 256] {
    let mut sizes = [0u8; 256];

    let mut k = 0;

    for (i, &length) in code_lengths.iter().enumerate() {
        for _ in 0..length {
            sizes[k] = (i + 1) as u8;
            k += 1;
        }
    }

    sizes
}

// Create huffman table codes as defined in Figure C.2
fn create_codes(sizes: &[u8; 256]) -> [u16; 256] {
    let mut codes = [0u16; 256];

    let mut current_code = 0;
    let mut current_size = sizes[0];

    for (&size, code) in sizes.iter().take_while(|s| **s != 0).zip(codes.iter_mut()) {
        if current_size != size {
            let size_diff = size - current_size;
            current_code <<= size_diff as usize;
            current_size = size;
        }

        *code = current_code;
        current_code += 1;
    }

    codes
}

// Create huffman table codes as defined in Figure C.3
fn create_lookup_table(code_length: &[u8; 16], values: &[u8]) -> [(u8, u16); 256] {
    let sizes = create_sizes(code_length);
    let codes = create_codes(&sizes);

    let mut lookup_table = [(0u8, 0u16); 256];

    for (i, &value) in values.iter().enumerate() {
        lookup_table[value as usize] = (sizes[i], codes[i]);
    }

    lookup_table
}

/// Symbols 0..=255 ordered by code size, ascending symbol within a size,
/// as Figure K.4's size-by-symbol double loop orders them. A counting sort
/// visits each symbol once instead of once per size, including tree depths
/// above 32 before length limiting. Zero sizes denote absent symbols.
fn values_by_code_size(codesize: &[usize; 257]) -> ([u8; 256], usize) {
    // start[size] = number of symbols with a code size in 1..size
    let mut start = [0u16; MAX_CODE_SIZE + 2];
    let mut max_size = 0;
    for &size in &codesize[..256] {
        if (1..=MAX_CODE_SIZE).contains(&size) {
            start[size + 1] += 1;
            max_size = max_size.max(size);
        }
    }
    for size in 2..=max_size + 1 {
        start[size] += start[size - 1];
    }
    let count = usize::from(start[max_size + 1]);
    let mut huffval = [0u8; 256];
    for (symbol, &size) in codesize[..256].iter().enumerate() {
        if (1..=MAX_CODE_SIZE).contains(&size) {
            huffval[usize::from(start[size])] = symbol as u8;
            start[size] += 1;
        }
    }
    (huffval, count)
}

/// The IJG merge order: least frequency first, highest symbol index on ties.
fn optimized_code_sizes(freq: &mut [u32; 257]) -> [usize; 257] {
    // A single integer comparison orders both fields. Inverted frequencies
    // make BinaryHeap's maximum the least frequent remaining root; nine low
    // bits retain the original highest-index tie break, including symbol 256.
    let priority = |frequency: u32, index: usize| (u64::from(!frequency) << 9) | index as u64;
    let mut roots: BinaryHeap<u64> = freq
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f != 0)
        .map(|(i, &f)| priority(f, i))
        .collect();
    let mut others = [-1i32; 257];
    let mut codesize = [0usize; 257];
    while let Some(first) = roots.pop() {
        let Some(second) = roots.pop() else {
            break;
        };
        let root = (first & 511) as usize;
        let mut v1 = root;
        let mut v2 = (second & 511) as usize;
        freq[v1] += freq[v2];
        freq[v2] = 0;
        // Keep the original positive-frequency rule even if a release-build
        // u32 merge wraps to zero; that root must then leave the active set.
        if freq[root] != 0 {
            roots.push(priority(freq[root], root));
        }

        codesize[v1] += 1;
        while others[v1] >= 0 {
            v1 = others[v1] as usize;
            codesize[v1] += 1;
        }
        others[v1] = v2 as i32;
        codesize[v2] += 1;
        while others[v2] >= 0 {
            v2 = others[v2] as usize;
            codesize[v2] += 1;
        }
    }
    codesize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_code_sizes(mut freq: [u32; 257]) -> [usize; 257] {
        let mut others = [-1i32; 257];
        let mut codesize = [0usize; 257];

        // Find Huffman code sizes
        // Figure K.1
        loop {
            let mut v1 = None;
            let mut v1_min = u32::MAX;

            // Find the largest value with the least non zero frequency
            for (i, &f) in freq.iter().enumerate() {
                if f > 0 && f <= v1_min {
                    v1_min = f;
                    v1 = Some(i);
                }
            }

            let mut v1 = match v1 {
                Some(v) => v,
                None => break,
            };

            let mut v2 = None;
            let mut v2_min = u32::MAX;

            // Find the next largest value with the least non zero frequency
            for (i, &f) in freq.iter().enumerate() {
                if f > 0 && f <= v2_min && i != v1 {
                    v2_min = f;
                    v2 = Some(i);
                }
            }

            let mut v2 = match v2 {
                Some(v) => v,
                None => break,
            };

            freq[v1] += freq[v2];
            freq[v2] = 0;

            codesize[v1] += 1;
            while others[v1] >= 0 {
                v1 = others[v1] as usize;
                codesize[v1] += 1;
            }

            others[v1] = v2 as i32;

            codesize[v2] += 1;
            while others[v2] >= 0 {
                v2 = others[v2] as usize;
                codesize[v2] += 1;
            }
        }

        codesize
    }

    /// The original Figure K.4 double loop.
    #[allow(clippy::needless_range_loop)]
    fn reference_values_by_code_size(codesize: &[usize; 257]) -> ([u8; 256], usize) {
        let mut huffval = [0u8; 256];
        let mut k = 0;
        for i in 1..=MAX_CODE_SIZE {
            for j in 0..=255 {
                if codesize[j] == i {
                    huffval[k] = j as u8;
                    k += 1;
                }
            }
        }
        (huffval, k)
    }

    #[test]
    fn counting_sort_matches_figure_k4_order() {
        let check = |codesize: [usize; 257]| {
            assert_eq!(
                values_by_code_size(&codesize),
                reference_values_by_code_size(&codesize),
                "{codesize:?}"
            );
        };
        check([0; 257]);
        check([1; 257]);
        check([32; 257]);
        check([33; 257]);
        check([MAX_CODE_SIZE; 257]);
        check([MAX_CODE_SIZE + 1; 257]);
        for size in 0..=40 {
            for symbol in [0, 1, 127, 254, 255, 256] {
                let mut codesize = [0; 257];
                codesize[symbol] = size;
                check(codesize);
            }
        }
        let mut seed = 0x0bad_5eedu32;
        for case in 0..4096 {
            let codesize = core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let r = (seed >> 8) as usize;
                match case % 4 {
                    0 => r % 34,
                    1 => {
                        if r % 3 == 0 {
                            0
                        } else {
                            1 + r % 16
                        }
                    }
                    2 => r % 3 + 7,
                    _ => r % 64,
                }
            });
            check(codesize);
        }
        // Real histograms through the merge, before and after length limiting
        let mut seed = 0x31415926u32;
        for case in 0..1024 {
            let mut freq: [u32; 257] = core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                match case % 3 {
                    0 => seed % 4,
                    1 => 1 << (seed % 20),
                    _ => {
                        if seed & 7 == 0 {
                            seed % 10000
                        } else {
                            0
                        }
                    }
                }
            });
            freq[256] = 1;
            check(optimized_code_sizes(&mut freq));
        }
    }

    fn check_optimized_table(freq: [u32; 257]) {
        let table = HuffmanTable::new_optimized(freq);
        let mut lengths = [0u16; 16];
        let mut seen = [false; 256];
        let mut kraft = 0u32;
        for &symbol in table.values() {
            let symbol = usize::from(symbol);
            assert!(freq[symbol] > 0);
            assert!(!seen[symbol]);
            seen[symbol] = true;
            let &(size, code) = table.get_for_value(symbol as u8);
            assert!((1..=16).contains(&size));
            assert!(u32::from(code) < (1u32 << size) - 1, "all-ones code");
            lengths[usize::from(size) - 1] += 1;
            kraft += 1 << (16 - size);
        }
        assert!(kraft < 1 << 16, "reserved code must remain unused");
        for symbol in 0..256 {
            assert_eq!(seen[symbol], freq[symbol] != 0, "symbol {symbol}");
        }
        for (actual, expected) in table.length().iter().zip(lengths) {
            assert_eq!(u16::from(*actual), expected);
        }
        // Canonical codes must be prefix-free, including after redistribution.
        for (i, &a) in table.values().iter().enumerate() {
            for &b in &table.values()[i + 1..] {
                let &(a_size, a_code) = table.get_for_value(a);
                let &(b_size, b_code) = table.get_for_value(b);
                assert!(a_size <= b_size);
                assert_ne!(u32::from(a_code), u32::from(b_code) >> (b_size - a_size));
            }
        }
    }

    #[test]
    fn length_limiting_preserves_symbols_from_trees_deeper_than_32() {
        for count in 34..=44 {
            let mut freq = [0u32; 257];
            let symbols = [0, 0xf0]
                .into_iter()
                .chain((0..16).flat_map(|run| (1..=10).map(move |size| (run << 4) | size)));
            let (mut a, mut b) = (1u32, 2u32);
            for symbol in symbols.take(count) {
                freq[symbol] = a;
                (a, b) = (b, a + b);
            }
            freq[256] = 1;
            let mut scratch = freq;
            assert!(
                optimized_code_sizes(&mut scratch)
                    .into_iter()
                    .max()
                    .unwrap()
                    > 32
            );
            check_optimized_table(freq);
        }
    }

    #[test]
    fn intermediate_length_buckets_can_hold_256_symbols() {
        let mut freq = [1u32; 257];
        freq[0] = 1_000_000;
        let mut scratch = freq;
        let sizes = optimized_code_sizes(&mut scratch);
        assert_eq!(sizes.iter().filter(|&&size| size == 9).count(), 256);
        check_optimized_table(freq);
    }

    #[test]
    fn heap_preserves_original_merge_order() {
        let check = |freq: [u32; 257]| {
            let mut actual = freq;
            assert_eq!(
                optimized_code_sizes(&mut actual),
                reference_code_sizes(freq)
            );
        };
        // Equal counts exercise the highest-index tie break, including the
        // pseudo-symbol that prevents an all-ones Huffman code.
        check([1; 257]);
        check([0; 257]);
        for count in 1..=257 {
            let mut freq = [0; 257];
            freq[..count].fill(1);
            check(freq);
        }
        for symbol in 0..256 {
            let mut freq = [0; 257];
            freq[symbol] = 100000;
            freq[256] = 1;
            check(freq);
        }
        let mut seed = 0x31415926u32;
        for case in 0..1024 {
            let mut freq = core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                match case % 4 {
                    0 => seed % 4,
                    1 => {
                        if seed & 7 == 0 {
                            seed % 10000
                        } else {
                            0
                        }
                    }
                    2 => seed % 1000000,
                    _ => 1 << (seed % 20),
                }
            });
            freq[256] = 1;
            check(freq);
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn wrapped_zero_frequency_leaves_the_active_set() {
        let mut frequencies = [0; 257];
        frequencies[0] = u32::MAX;
        frequencies[1] = u32::MAX;
        frequencies[2] = u32::MAX;
        frequencies[256] = 1;
        let mut actual = frequencies;
        assert_eq!(
            optimized_code_sizes(&mut actual),
            reference_code_sizes(frequencies)
        );
    }
}
