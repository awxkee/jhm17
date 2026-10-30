use jhm17::{ColorType, Encoder, ProgressiveMode, SamplePrecision, SamplingFactor};

#[derive(Default)]
struct Headers {
    segments: usize,
    definitions: usize,
    repeated: usize,
    scans: usize,
}

fn headers(jpeg: &[u8]) -> Headers {
    assert_eq!(&jpeg[..2], &[255, 216]);
    let mut pos = 2;
    let mut headers = Headers::default();
    let mut previous: [Vec<u8>; 8] = core::array::from_fn(|_| Vec::new());
    while pos < jpeg.len() {
        assert_eq!(jpeg[pos], 255);
        let marker = jpeg[pos + 1];
        pos += 2;
        if marker == 217 {
            break;
        }
        let length = usize::from(u16::from_be_bytes([jpeg[pos], jpeg[pos + 1]]));
        let end = pos + length;
        if marker == 196 {
            headers.segments += 1;
            pos += 2;
            while pos < end {
                let descriptor = jpeg[pos];
                let key = usize::from(descriptor >> 4) * 4 + usize::from(descriptor & 15);
                let count: usize = jpeg[pos + 1..pos + 17]
                    .iter()
                    .map(|&v| usize::from(v))
                    .sum();
                let definition = &jpeg[pos + 1..pos + 17 + count];
                if previous[key] == definition {
                    headers.repeated += 1;
                }
                previous[key] = definition.to_vec();
                headers.definitions += 1;
                pos += 17 + count;
            }
            assert_eq!(pos, end, "DHT must end at its declared length");
        }
        pos = end;
        if marker == 218 {
            headers.scans += 1;
            // Skip entropy data, including byte stuffing and restart markers.
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
fn sequential_tables_share_one_dht_segment() {
    for (color, channels, tables) in [
        (ColorType::Luma, 1, 2),
        (ColorType::Rgb, 3, 4),
        (ColorType::Cmyk, 4, 4),
    ] {
        let pixels: Vec<u8> = (0..37 * 21 * channels)
            .map(|i| (i * 73 + i / 7 * 19) as u8)
            .collect();
        for sampling in [SamplingFactor::F_1_1, SamplingFactor::F_2_2] {
            for optimized in [false, true] {
                let mut output = Vec::new();
                let mut encoder = Encoder::new(&mut output, 80);
                encoder.set_rdo(None);
                encoder.set_sample_precision(SamplePrecision::Integer);
                encoder.set_sampling_factor(sampling);
                encoder.set_optimized_huffman_tables(optimized);
                encoder.encode(&pixels, 37, 21, color).unwrap();
                let headers = headers(&output);
                assert_eq!(headers.segments, 1);
                assert_eq!(headers.definitions, tables);
                let decoded = jpeg_decoder::Decoder::new(output.as_slice())
                    .decode()
                    .unwrap();
                assert_eq!(decoded.len(), pixels.len());
            }
        }
    }
}

#[test]
fn progressive_scans_reuse_identical_tables_without_changing_pixels() {
    let pixels = vec![128; 64 * 64 * 3];
    let encode = |mode, restart| {
        let mut output = Vec::new();
        let mut encoder = Encoder::new(&mut output, 75);
        encoder.set_progressive_mode(mode);
        encoder.set_restart_interval(restart);
        encoder.encode(&pixels, 64, 64, ColorType::Rgb).unwrap();
        output
    };
    for restart in [0, 7] {
        let sequential = encode(None, restart);
        let reference = jpeg_decoder::Decoder::new(sequential.as_slice())
            .decode()
            .unwrap();
        for mode in [
            ProgressiveMode::Optimized,
            ProgressiveMode::Standard,
            ProgressiveMode::Spectral(4),
        ] {
            let jpeg = encode(Some(mode), restart);
            let headers = headers(&jpeg);
            assert_eq!(headers.repeated, 0, "{mode:?}, restart={restart}");
            assert!(headers.definitions < headers.scans);
            let decoded = jpeg_decoder::Decoder::new(jpeg.as_slice())
                .decode()
                .unwrap();
            assert_eq!(decoded, reference, "{mode:?}, restart={restart}");
        }
    }
}
