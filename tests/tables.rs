use jhm17::{ColorType, Encoder, ProgressiveMode, SamplingFactor, StepTuning};

fn source(width: u16, height: u16) -> Vec<u8> {
    (0..usize::from(width) * usize::from(height))
        .flat_map(|i| {
            let x = i % usize::from(width);
            let y = i / usize::from(width);
            [
                ((x * 17 + y * 3 + 23) % 256) as u8,
                ((x * 5 + y * 19 + 109) % 256) as u8,
                ((x * 11 + y * 7 + 207) % 256) as u8,
            ]
        })
        .collect()
}

struct Headers {
    /// Number of DQT segments
    segments: usize,
    /// Destination and values of each transmitted table, in stream order
    tables: Vec<(u8, Vec<u8>)>,
    /// Quantization table selector of each frame component
    selectors: Vec<u8>,
}

fn headers(jpeg: &[u8]) -> Headers {
    assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
    let mut pos = 2;
    let mut headers = Headers {
        segments: 0,
        tables: Vec::new(),
        selectors: Vec::new(),
    };
    loop {
        assert_eq!(jpeg[pos], 0xff);
        let marker = jpeg[pos + 1];
        pos += 2;
        if marker == 0xda {
            break;
        }
        let length = usize::from(u16::from_be_bytes([jpeg[pos], jpeg[pos + 1]]));
        let end = pos + length;
        match marker {
            0xdb => {
                headers.segments += 1;
                pos += 2;
                while pos < end {
                    let descriptor = jpeg[pos];
                    assert_eq!(descriptor >> 4, 0, "8-bit tables only");
                    headers
                        .tables
                        .push((descriptor & 15, jpeg[pos + 1..pos + 65].to_vec()));
                    pos += 65;
                }
                assert_eq!(pos, end, "DQT length must cover its tables exactly");
            }
            0xc0..=0xc2 => {
                let components = usize::from(jpeg[pos + 7]);
                for c in 0..components {
                    headers.selectors.push(jpeg[pos + 8 + c * 3 + 2]);
                }
            }
            _ => {}
        }
        pos = end;
    }
    headers
}

fn encode(
    data: &[u8],
    color: ColorType,
    sampling: SamplingFactor,
    progressive: Option<ProgressiveMode>,
    configure: impl FnOnce(&mut Encoder<&mut Vec<u8>>),
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = Encoder::new(&mut out, 80);
    encoder.set_sampling_factor(sampling);
    encoder.set_progressive_mode(progressive);
    configure(&mut encoder);
    encoder.encode(data, 37, 21, color).unwrap();
    out
}

fn assert_decodes(jpeg: &[u8], components: usize) {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg);
    let pixels = decoder.decode().unwrap();
    assert_eq!(pixels.len(), 37 * 21 * components);
}

#[test]
fn all_tables_share_one_dqt_segment() {
    let rgb = source(37, 21);
    for progressive in [None, Some(ProgressiveMode::Standard)] {
        let jpeg = encode(
            &rgb,
            ColorType::Rgb,
            SamplingFactor::F_2_2,
            progressive,
            |_| {},
        );
        let headers = headers(&jpeg);
        assert_eq!(headers.segments, 1);
        let destinations: Vec<u8> = headers.tables.iter().map(|t| t.0).collect();
        // 4:2:0 scales Cb apart from Cr, so Cr keeps its own table.
        assert_eq!(destinations, [0, 1, 2]);
        assert_eq!(headers.selectors, [0, 1, 2]);
        assert_decodes(&jpeg, 3);
    }
}

#[test]
fn equal_chroma_tables_are_sent_once() {
    let rgb = source(37, 21);
    for progressive in [None, Some(ProgressiveMode::Standard)] {
        let jpeg = encode(
            &rgb,
            ColorType::Rgb,
            SamplingFactor::F_1_1,
            progressive,
            |_| {},
        );
        let headers = headers(&jpeg);
        assert_eq!(headers.segments, 1);
        let destinations: Vec<u8> = headers.tables.iter().map(|t| t.0).collect();
        assert_eq!(destinations, [0, 1]);
        assert_eq!(headers.selectors, [0, 1, 1]);
        assert_decodes(&jpeg, 3);
    }
}

#[test]
fn distinct_cr_tuning_keeps_a_third_table() {
    let rgb = source(37, 21);
    let jpeg = encode(
        &rgb,
        ColorType::Rgb,
        SamplingFactor::F_1_1,
        None,
        |encoder| {
            let cr = StepTuning {
                ac: 1.5,
                ..StepTuning::default()
            };
            encoder.set_chroma_step_tuning(StepTuning::default(), cr);
        },
    );
    let headers = headers(&jpeg);
    assert_eq!(headers.selectors, [0, 1, 2]);
    assert_eq!(headers.tables.len(), 3);
    assert_ne!(headers.tables[1].1, headers.tables[2].1);
    assert_decodes(&jpeg, 3);
}

#[test]
fn grayscale_sends_only_the_luma_table() {
    let gray: Vec<u8> = source(37, 21).chunks(3).map(|p| p[0]).collect();
    let jpeg = encode(&gray, ColorType::Luma, SamplingFactor::F_1_1, None, |_| {});
    let headers = headers(&jpeg);
    assert_eq!(headers.segments, 1);
    assert_eq!(headers.tables.len(), 1);
    assert_eq!(headers.tables[0].0, 0);
    assert_eq!(headers.selectors, [0]);
    assert_decodes(&jpeg, 1);
}
