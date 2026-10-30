//! Rate/quality of jhm17 with and without the decoder-feedback polish.
//!
//! Every encode is decoded by libjpeg-turbo's `djpeg` and scored with
//! SSIMULACRA 2 (ssimu2-lite), ColorVideoVDP (cvvdp-lite, macbook_pro_16) and,
//! with `--ba`, butteraugli's 3-norm. Output lines follow the stats tool's log
//! format, one `== image` header per line.
//!
//! cargo run --release --features polish --example polish_bench -- \
//!     --sampling 420 --qualities 30,50,70,90 --jobs 8 [--ba] [--arms jhm17,polish] IMAGES...

use jhm17::{
    ChromaSubsamplingMethod, ColorType, Encoder, Polish, PolishJudge, PolishStats, ProgressiveMode,
    RdoOptions, SamplingFactor,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// `jhm17` ([`Polish::Off`]), `polish` ([`Polish::Greedy`]), `kbest`
/// ([`Polish::Trellis`]), `kbest-greedy` ([`Polish::TrellisGreedy`]),
/// `polish-ss2` / `polish-cvvdp` (greedy with one judge), optionally followed
/// by `:p=3,b=0,r=1,s=40,t=0.4,k=2,kp=6` (proposals, max bits, rounds, scan
/// limit, restore threshold, trellis paths per λ, trellis candidates verified)
fn arm_polish(arm: &str) -> Polish {
    let (name, knobs) = arm.split_once(':').unwrap_or((arm, ""));
    let mode = match name {
        "jhm17" => Polish::Off,
        "polish" | "polish-ss2" | "polish-cvvdp" => Polish::Greedy,
        "kbest" => Polish::Trellis,
        "kbest-greedy" => Polish::TrellisGreedy,
        _ => panic!("unknown arm {arm}"),
    };
    let judge = match name {
        "polish-ss2" => PolishJudge::Ssimulacra2,
        "polish-cvvdp" => PolishJudge::Cvvdp,
        _ => PolishJudge::Both,
    };
    let Some(mut o) = mode.options() else {
        return mode;
    };
    if knobs.is_empty() && judge == o.judge {
        return mode;
    }
    o.judge = judge;
    for kv in knobs.split(',').filter(|k| !k.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        match k {
            "p" => o.proposals = v.parse().unwrap(),
            "b" => o.max_bits = v.parse().unwrap(),
            "r" => o.rounds = v.parse().unwrap(),
            "s" => o.scan_limit = v.parse().unwrap(),
            "t" => o.restore_threshold = v.parse().unwrap(),
            "c" => o.cvvdp_tries = v.parse().unwrap(),
            "k" => o.trellis_k = v.parse().unwrap(),
            "kp" => o.trellis_proposals = v.parse().unwrap(),
            _ => panic!("unknown knob {k}"),
        }
    }
    Polish::Custom(o)
}

fn encode(
    rgb: &[u8],
    w: usize,
    h: usize,
    q: u8,
    full: bool,
    polish: Polish,
) -> (Vec<u8>, Option<PolishStats>) {
    let mut out = Vec::new();
    let sink = Arc::new(Mutex::new(None));
    let mut e = Encoder::new(&mut out, q);
    e.set_sampling_factor(if full {
        SamplingFactor::F_1_1
    } else {
        SamplingFactor::F_2_2
    });
    e.set_chroma_subsampling_method(ChromaSubsamplingMethod::Average);
    e.set_optimized_huffman_tables(true);
    e.set_rdo(Some(RdoOptions::default()));
    e.set_progressive_mode(Some(ProgressiveMode::Optimized));
    e.set_polish(polish);
    e.set_polish_stats_sink(sink.clone());
    e.encode(rgb, w as u16, h as u16, ColorType::Rgb).unwrap();
    let stats = *sink.lock().unwrap();
    (out, stats)
}

const MOZJPEG: &str = "/opt/homebrew/opt/mozjpeg/bin/cjpeg";
const CJPEGLI: &str = "/Users/radzivon/RustroverProjects/jpegli/build/tools/cjpegli";
const REDUX: &str = "target/ext/cjpegli-redux";

/// Encode with an external encoder from the source PPM; `None` for jhm17 arms
fn external(arm: &str, ppm: &Path, out: &Path, q: u8, full: bool) -> Option<Vec<u8>> {
    let mut cmd = match arm {
        "mozjpeg" => {
            let mut c = Command::new(MOZJPEG);
            c.args([
                "-quality",
                &q.to_string(),
                "-sample",
                if full { "1x1" } else { "2x2" },
            ])
            .arg("-outfile")
            .arg(out)
            .arg(ppm);
            c
        }
        "cjpegli" | "redux" | "redux-sharp" => {
            let mut c = Command::new(if arm == "cjpegli" { CJPEGLI } else { REDUX });
            c.arg(ppm).arg(out).args([
                "-q",
                &q.to_string(),
                if full {
                    "--chroma_subsampling=444"
                } else {
                    "--chroma_subsampling=420"
                },
                "--quiet",
            ]);
            if arm == "redux-sharp" {
                c.arg("--sharp_yuv");
            }
            c
        }
        _ => return None,
    };
    let status = cmd.status().unwrap();
    assert!(status.success(), "{arm} failed");
    Some(std::fs::read(out).unwrap())
}

fn write_ppm(path: &Path, rgb: &[u8], w: usize, h: usize) {
    let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
    data.extend_from_slice(rgb);
    std::fs::write(path, data).unwrap();
}

fn read_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while data[i].is_ascii_whitespace() {
            i += 1;
        }
        if data[i] == b'#' {
            while data[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&data[s..i]).to_string());
    }
    assert_eq!(fields[0], "P6");
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    (w, h, data[i + 1..i + 1 + w * h * 3].to_vec())
}

fn butteraugli(reference: &Path, test: &Path) -> (f64, f64) {
    let out = Command::new("butteraugli_main")
        .arg(reference)
        .arg(test)
        .args(["--pnorm", "3"])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    let max = lines
        .next()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or(f64::NAN);
    let norm = text
        .lines()
        .find_map(|l| l.strip_prefix("3-norm:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(f64::NAN);
    (norm, max)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut images = Vec::new();
    let mut qualities: Vec<u8> = vec![25, 30, 40, 50, 60, 70, 80, 90, 95];
    let mut full = false;
    let mut jobs = 8;
    let mut ba = false;
    let mut arms: Vec<String> = vec!["jhm17".into(), "polish".into()];
    let mut tmp = std::env::temp_dir().join("polish_bench");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--sampling" => full = args.next().unwrap() == "444",
            "--qualities" => {
                qualities = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|q| q.parse().unwrap())
                    .collect()
            }
            "--jobs" => jobs = args.next().unwrap().parse().unwrap(),
            "--ba" => ba = true,
            "--arms" => arms = args.next().unwrap().split(';').map(String::from).collect(),
            "--tmp" => tmp = PathBuf::from(args.next().unwrap()),
            _ => images.push(PathBuf::from(a)),
        }
    }
    std::fs::create_dir_all(&tmp).unwrap();
    let sources: Vec<(PathBuf, usize, usize, Vec<u8>)> = images
        .iter()
        .map(|p| {
            let img = image::open(p).unwrap().to_rgb8();
            let (w, h) = img.dimensions();
            (p.clone(), w as usize, h as usize, img.into_raw())
        })
        .collect();
    for (i, (_, w, h, rgb)) in sources.iter().enumerate() {
        write_ppm(&tmp.join(format!("src{i}.ppm")), rgb, *w, *h);
    }
    let mut tasks = Vec::new();
    for i in 0..sources.len() {
        for arm in &arms {
            for &q in &qualities {
                tasks.push((i, arm.clone(), q));
            }
        }
    }
    let next = AtomicUsize::new(0);
    let print = Mutex::new(());
    let metric = cvvdp_lite::CvvdpLite::new(cvvdp_lite::Display::macbook_pro_16()).unwrap();
    std::thread::scope(|scope| {
        for worker in 0..jobs {
            let (tasks, sources, next, print, metric, tmp) =
                (&tasks, &sources, &next, &print, &metric, &tmp);
            scope.spawn(move || {
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    let Some((i, arm, q)) = tasks.get(t) else {
                        break;
                    };
                    let (path, w, h, rgb) = &sources[*i];
                    let start = Instant::now();
                    let jpg = tmp.join(format!("w{worker}.jpg"));
                    let source = tmp.join(format!("src{i}.ppm"));
                    let (jpeg, stats) = match external(arm, &source, &jpg, *q, full) {
                        Some(jpeg) => (jpeg, None),
                        None => encode(rgb, *w, *h, *q, full, arm_polish(arm)),
                    };
                    let secs = start.elapsed().as_secs_f64();
                    let ppm = tmp.join(format!("w{worker}.ppm"));
                    std::fs::write(&jpg, &jpeg).unwrap();
                    let status = Command::new("djpeg")
                        .args(["-ppm", "-outfile"])
                        .arg(&ppm)
                        .arg(&jpg)
                        .status()
                        .unwrap();
                    assert!(status.success());
                    let (dw, dh, decoded) = read_ppm(&std::fs::read(&ppm).unwrap());
                    assert_eq!((dw, dh), (*w, *h));
                    let ss2 = ssimu2_lite::compare(
                        ssimu2_lite::RgbImage::srgb8(rgb, *w, *h, w * 3).unwrap(),
                        ssimu2_lite::RgbImage::srgb8(&decoded, *w, *h, w * 3).unwrap(),
                    )
                    .unwrap();
                    let jod = metric
                        .compare(
                            cvvdp_lite::RgbImage::srgb8(rgb, *w, *h, w * 3).unwrap(),
                            cvvdp_lite::RgbImage::srgb8(&decoded, *w, *h, w * 3).unwrap(),
                        )
                        .unwrap()
                        .jod;
                    let ba = if ba {
                        let (n, m) = butteraugli(path, &ppm);
                        format!("  BA3 {n:.4} (max {m:.4})")
                    } else {
                        String::new()
                    };
                    let extra = stats.map_or(String::new(), |s| {
                        format!(
                            "  polish {}/{} moves, {:+} bits, in-loop SS2 {:.3}->{:.3} JOD {:.4}->{:.4}, slopes {:.2}/{:.4}, kinds {:?}/{:?} rounds {:?}",
                            s.accepted,
                            s.evaluated,
                            s.bits_delta,
                            s.ss2_before,
                            s.ss2_after,
                            s.jod_before,
                            s.jod_after,
                            s.ss2_slope,
                            s.jod_slope,
                            s.kind_accepted,
                            s.kind_evaluated,
                            s.round_accepted
                        )
                    });
                    let bytes = jpeg.len();
                    let bpp = bytes as f64 * 8.0 / (*w * *h) as f64;
                    let _lock = print.lock().unwrap();
                    println!("== {} {w}x{h}", path.display());
                    println!(
                        "  {} q{q}  {bytes} B  {bpp:.4} bpp  SS2 {ss2:.3}{ba}  CVVDP {jod:.4}  time {secs:.2}s{extra}",
                        arm.replace(' ', "")
                    );
                }
            });
        }
    });
}
