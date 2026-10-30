use jhm17::{ColorType, Encoder, ProgressiveMode, SamplePrecision};

fn source() -> Vec<u8> {
    (0..37 * 21 * 3)
        .map(|i| (i * 73 + i / 7 * 19) as u8)
        .collect()
}

fn encode(arithmetic: bool, mode: Option<ProgressiveMode>, optimized: bool) -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = Encoder::new(&mut output, 80);
    encoder.set_arithmetic_coding(arithmetic);
    encoder.set_progressive_mode(mode);
    encoder.set_optimized_huffman_tables(optimized);
    encoder.set_rdo(None);
    encoder.set_sample_precision(SamplePrecision::Integer);
    encoder.set_restart_interval(1);
    encoder.encode(&source(), 37, 21, ColorType::Rgb).unwrap();
    output
}

fn headers(jpeg: &[u8]) -> Vec<(u8, &[u8])> {
    let mut headers = Vec::new();
    assert_eq!(&jpeg[..2], &[255, 216]);
    let mut pos = 2;
    while pos < jpeg.len() {
        assert_eq!(jpeg[pos], 255);
        let marker = jpeg[pos + 1];
        pos += 2;
        if marker == 217 {
            break;
        }
        let length = usize::from(u16::from_be_bytes([jpeg[pos], jpeg[pos + 1]]));
        headers.push((marker, &jpeg[pos + 2..pos + length]));
        pos += length;
        if marker == 218 {
            loop {
                if jpeg[pos] != 255 {
                    pos += 1;
                } else if jpeg[pos + 1] == 0 || (208..=215).contains(&jpeg[pos + 1]) {
                    pos += 2;
                } else {
                    break;
                }
            }
        }
    }
    headers
}

#[test]
fn arithmetic_is_opt_in_and_can_be_switched_off() {
    let mut output = Vec::new();
    let mut encoder = Encoder::new(&mut output, 80);
    assert!(!encoder.arithmetic_coding());
    encoder.set_arithmetic_coding(true);
    assert!(encoder.arithmetic_coding());
    encoder.set_arithmetic_coding(false);
    assert!(!encoder.arithmetic_coding());
    encoder.set_rdo(None);
    encoder.set_sample_precision(SamplePrecision::Integer);
    encoder.set_restart_interval(1);
    encoder.encode(&source(), 37, 21, ColorType::Rgb).unwrap();
    assert_eq!(output, encode(false, None, true));
    assert!(headers(&output).iter().any(|&(marker, _)| marker == 0xc0));
    assert!(!headers(&output).iter().any(|&(marker, _)| marker == 0xcc));
    assert_eq!(
        jpeg_decoder::Decoder::new(output.as_slice())
            .decode()
            .unwrap()
            .len(),
        37 * 21 * 3
    );
}

#[test]
fn arithmetic_uses_sof9_or_sof10_and_dac_without_huffman_tables() {
    for mode in [
        None,
        Some(ProgressiveMode::Standard),
        Some(ProgressiveMode::Optimized),
        Some(ProgressiveMode::Spectral(4)),
    ] {
        let output = encode(true, mode, true);
        assert_eq!(
            output,
            encode(true, mode, false),
            "Huffman optimization must not affect arithmetic output"
        );
        let headers = headers(&output);
        let frame = if mode.is_some() { 0xca } else { 0xc9 };
        assert!(headers.iter().any(|&(marker, _)| marker == frame));
        assert!(!headers.iter().any(|&(marker, _)| marker == 0xc4));
        let conditioning: Vec<_> = headers
            .iter()
            .filter(|&&(marker, _)| marker == 0xcc)
            .collect();
        assert_eq!(conditioning.len(), 1);
        assert_eq!(conditioning[0].1, &[0, 0x10, 0x10, 5, 1, 0x10, 0x11, 5]);
        assert!(
            headers
                .iter()
                .any(|&(marker, data)| marker == 0xdd && data == [0, 1])
        );
    }
}

#[test]
fn grayscale_only_defines_its_used_conditioning_table() {
    let mut output = Vec::new();
    let mut encoder = Encoder::new(&mut output, 80);
    encoder.set_arithmetic_coding(true);
    encoder.encode(&[128; 64], 8, 8, ColorType::Luma).unwrap();
    let headers = headers(&output);
    let dac = headers.iter().find(|&&(marker, _)| marker == 0xcc).unwrap();
    assert_eq!(dac.1, &[0, 0x10, 0x10, 5]);
}
