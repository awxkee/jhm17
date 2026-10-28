use crate::dct_f32::Transforms;
use crate::huffman::{CodingClass, HuffmanTable};
use crate::image_buffer::*;
use crate::marker::Marker;
use crate::progressive::{self, ProgressiveMode};
use crate::quantization::{Deadzone, QuantizationTable, QuantizationTableType, StepTuning};
use crate::rdo::{
    AcPrices, RdoOptions, aq_lambda_scales, block_activity, distortion_weights, rdoq_block,
};
use crate::writer::{JfifWrite, JfifWriter, ZIGZAG};
use crate::{EncodingError, PixelDensity};

use alloc::vec;
use alloc::vec::Vec;

use std::io::BufWriter;

use std::fs::File;

use std::path::Path;

/// # Color types used in encoding
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum JpegColorType {
    /// One component grayscale colorspace
    Luma,

    /// Three component YCbCr colorspace
    Ycbcr,

    /// 4 Component CMYK colorspace
    Cmyk,

    /// 4 Component YCbCrK colorspace
    Ycck,
}

#[derive(Copy, Clone)]
#[repr(C, align(32))]
pub(crate) struct AlignedBlock {
    pub(crate) data: [i16; 64],
}

impl AlignedBlock {
    pub(crate) const fn new(data: [i16; 64]) -> Self {
        AlignedBlock { data }
    }
}

impl Default for AlignedBlock {
    fn default() -> Self {
        AlignedBlock { data: [0i16; 64] }
    }
}

impl JpegColorType {
    pub(crate) fn get_num_components(self) -> usize {
        use JpegColorType::*;

        match self {
            Luma => 1,
            Ycbcr => 3,
            Cmyk | Ycck => 4,
        }
    }
}

/// # Color types for input images
///
/// Available color input formats for [Encoder::encode]. Other types can be used
/// by implementing an [ImageBuffer](crate::ImageBuffer).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ColorType {
    /// Grayscale with 1 byte per pixel
    Luma,

    /// RGB with 3 bytes per pixel
    Rgb,

    /// Red, Green, Blue with 4 bytes per pixel. The alpha channel will be ignored during encoding.
    Rgba,

    /// RGB with 3 bytes per pixel
    Bgr,

    /// RGBA with 4 bytes per pixel. The alpha channel will be ignored during encoding.
    Bgra,

    /// YCbCr with 3 bytes per pixel.
    Ycbcr,

    /// CMYK with 4 bytes per pixel.
    Cmyk,

    /// CMYK with 4 bytes per pixel. Encoded as YCCK (YCbCrK)
    CmykAsYcck,

    /// YCCK (YCbCrK) with 4 bytes per pixel.
    Ycck,
}

impl ColorType {
    pub(crate) fn get_bytes_per_pixel(self) -> usize {
        use ColorType::*;

        match self {
            Luma => 1,
            Rgb | Bgr | Ycbcr => 3,
            Rgba | Bgra | Cmyk | CmykAsYcck | Ycck => 4,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// # Sampling factors for chroma subsampling
///
/// ## Warning
/// Sampling factor of 4 are not supported by all decoders or applications
#[allow(non_camel_case_types)]
pub enum SamplingFactor {
    F_1_1 = 1 << 4 | 1,
    F_2_1 = 2 << 4 | 1,
    F_1_2 = 1 << 4 | 2,
    F_2_2 = 2 << 4 | 2,
    F_4_1 = 4 << 4 | 1,
    F_4_2 = 4 << 4 | 2,
    F_1_4 = 1 << 4 | 4,
    F_2_4 = 2 << 4 | 4,

    /// Alias for F_1_1
    R_4_4_4 = 0x80 | 1 << 4 | 1,

    /// Alias for F_1_2
    R_4_4_0 = 0x80 | 1 << 4 | 2,

    /// Alias for F_1_4
    R_4_4_1 = 0x80 | 1 << 4 | 4,

    /// Alias for F_2_1
    R_4_2_2 = 0x80 | 2 << 4 | 1,

    /// Alias for F_2_2
    R_4_2_0 = 0x80 | 2 << 4 | 2,

    /// Alias for F_2_4
    R_4_2_1 = 0x80 | 2 << 4 | 4,

    /// Alias for F_4_1
    R_4_1_1 = 0x80 | 4 << 4 | 1,

    /// Alias for F_4_2
    R_4_1_0 = 0x80 | 4 << 4 | 2,
}

impl SamplingFactor {
    /// Get variant for supplied factors or None if not supported
    pub fn from_factors(horizontal: u8, vertical: u8) -> Option<SamplingFactor> {
        use SamplingFactor::*;

        match (horizontal, vertical) {
            (1, 1) => Some(F_1_1),
            (1, 2) => Some(F_1_2),
            (1, 4) => Some(F_1_4),
            (2, 1) => Some(F_2_1),
            (2, 2) => Some(F_2_2),
            (2, 4) => Some(F_2_4),
            (4, 1) => Some(F_4_1),
            (4, 2) => Some(F_4_2),
            _ => None,
        }
    }

    pub(crate) fn get_sampling_factors(self) -> (u8, u8) {
        let value = self as u8;
        ((value >> 4) & 0x07, value & 0xf)
    }

    pub(crate) fn supports_interleaved(self) -> bool {
        use SamplingFactor::*;

        // Interleaved mode is only supported with h/v sampling factors of 1 or 2.
        // Sampling factors of 4 needs sequential encoding
        matches!(
            self,
            F_1_1 | F_2_1 | F_1_2 | F_2_2 | R_4_4_4 | R_4_4_0 | R_4_2_2 | R_4_2_0
        )
    }
}

/// Method for reducing each chroma block to a single sample when subsampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsamplingMethod {
    /// Use the top-left pixel of each block
    Nearest,
    /// Box-average each block, matching libjpeg's `h2v2_downsample` (default)
    Average,
}

/// Precision retained during conversion from RGB to JPEG components.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplePrecision {
    /// Use the original byte component planes.
    Integer,
    /// Retain fractional luma while keeping byte-rounded chroma samples.
    Luma,
    /// Retain fractional luma and chroma through subsampling and quantization.
    #[default]
    All,
}

/// Layout of one component's stored blocks: an MCU-aligned grid, of which
/// only `cols × rows` are inside the image. Interleaved scans code the whole
/// padded grid; non-interleaved scans code only the blocks inside the image.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BlockGrid {
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    pub(crate) padded_cols: usize,
    pub(crate) padded_rows: usize,
}

/// The order one scan codes a component's stored blocks in, plus the number
/// of those blocks between restart markers (0 = no restarts)
#[derive(Clone, Default)]
pub(crate) struct CodingOrder {
    // u16 JPEG dimensions bound each component grid to at most 8192² blocks.
    pub(crate) indices: Vec<u32>,
    pub(crate) restart_blocks: usize,
}

impl BlockGrid {
    /// Raster order over the blocks inside the image (a non-interleaved scan)
    pub fn raster_order(&self, restart_interval: usize) -> CodingOrder {
        let mut indices = Vec::with_capacity(self.cols * self.rows);
        for y in 0..self.rows {
            indices
                .extend((y * self.padded_cols..y * self.padded_cols + self.cols).map(|i| i as u32));
        }
        CodingOrder {
            indices,
            restart_blocks: restart_interval,
        }
    }

    /// Keep the non-interleaved scan's raster blocks in the existing allocation.
    pub fn compact_blocks(&self, mut blocks: Vec<AlignedBlock>) -> Vec<AlignedBlock> {
        if self.cols != self.padded_cols {
            for y in 1..self.rows {
                let start = y * self.padded_cols;
                blocks.copy_within(start..start + self.cols, y * self.cols);
            }
        }
        blocks.truncate(self.cols * self.rows);
        blocks
    }
}

pub(crate) struct Component {
    pub id: u8,
    pub quantization_table: u8,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
}

macro_rules! add_component {
    ($components:expr, $id:expr, $dest:expr, $h_sample:expr, $v_sample:expr) => {
        $components.push(Component {
            id: $id,
            quantization_table: $dest,
            dc_huffman_table: $dest,
            ac_huffman_table: $dest,
            horizontal_sampling_factor: $h_sample,
            vertical_sampling_factor: $v_sample,
        });
    };
}

/// # The JPEG encoder
pub struct Encoder<W: JfifWrite> {
    writer: JfifWriter<W>,
    functions: EncoderFunctions<W>,
    density: PixelDensity,
    quality: u8,

    components: Vec<Component>,
    quantization_tables: [QuantizationTableType; 2],
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],

    sampling_factor: SamplingFactor,
    chroma_subsampling_method: ChromaSubsamplingMethod,
    sample_precision: SamplePrecision,

    progressive: Option<ProgressiveMode>,

    restart_interval: Option<u16>,

    optimize_huffman_table: bool,

    rdo: Option<RdoOptions>,

    /// Y, Cb, Cr
    step_tuning: [StepTuning; 3],

    deadzone: Deadzone,

    auto_chroma_scale: bool,

    app_segments: Vec<(u8, Vec<u8>)>,
}

impl<W: JfifWrite> Encoder<W> {
    /// Create a new encoder with the given quality
    ///
    /// The quality must be between 1 and 100 where 100 is the highest image quality.<br>
    /// By default, quality settings below 90 use a chroma subsampling (2x2 / 4:2:0) which can
    /// be changed with [set_sampling_factor](Encoder::set_sampling_factor).<br>
    /// The default quantization table is [QuantizationTableType::ImprovedDetectionModel];
    /// files are larger than libjpeg's at the same quality number and better at the same size.
    pub fn new(w: W, quality: u8) -> Encoder<W> {
        let huffman_tables = [
            (
                HuffmanTable::default_luma_dc(),
                HuffmanTable::default_luma_ac(),
            ),
            (
                HuffmanTable::default_chroma_dc(),
                HuffmanTable::default_chroma_ac(),
            ),
        ];

        // Peterson/Ahumada/Watson's detection-model table beats Annex K on both
        // SSIMULACRA2 and butteraugli at matched rate, plain and with RDO.
        let quantization_tables = [
            QuantizationTableType::ImprovedDetectionModel,
            QuantizationTableType::ImprovedDetectionModel,
        ];

        let sampling_factor = if quality < 90 {
            SamplingFactor::F_2_2
        } else {
            SamplingFactor::F_1_1
        };

        Encoder {
            writer: JfifWriter::new(w),
            functions: EncoderFunctions::new(),
            density: PixelDensity::default(),
            quality,
            components: vec![],
            quantization_tables,
            huffman_tables,
            sampling_factor,
            chroma_subsampling_method: ChromaSubsamplingMethod::Average,
            sample_precision: SamplePrecision::All,
            progressive: None,
            restart_interval: None,
            optimize_huffman_table: true,
            rdo: Some(RdoOptions::default()),
            step_tuning: [
                StepTuning {
                    bands: DEFAULT_LUMA_BANDS,
                    ..StepTuning::default()
                },
                StepTuning::default(),
                StepTuning::default(),
            ],
            deadzone: Deadzone::default(),
            auto_chroma_scale: true,
            app_segments: Vec::new(),
        }
    }

    /// Set pixel density for the image
    ///
    /// By default, this value is None which is equal to "1 pixel per pixel".
    pub fn set_density(&mut self, density: PixelDensity) {
        self.density = density;
    }

    /// Return pixel density
    pub fn density(&self) -> PixelDensity {
        self.density
    }

    /// Set quality setting. Quality must be between 1 and 100 where 100 is the highest image quality.
    pub fn set_quality(&mut self, quality: u8) {
        self.quality = quality;
    }

    /// Get quality setting
    pub fn quality(&self) -> u8 {
        self.quality
    }

    /// Set chroma subsampling factor
    pub fn set_sampling_factor(&mut self, sampling: SamplingFactor) {
        self.sampling_factor = sampling;
    }

    /// Get chroma subsampling factor
    pub fn sampling_factor(&self) -> SamplingFactor {
        self.sampling_factor
    }

    /// Set the chroma subsampling method
    pub fn set_chroma_subsampling_method(&mut self, method: ChromaSubsamplingMethod) {
        self.chroma_subsampling_method = method;
    }

    /// Get the chroma subsampling method
    pub fn chroma_subsampling_method(&self) -> ChromaSubsamplingMethod {
        self.chroma_subsampling_method
    }

    /// Set RGB conversion precision. RGB-family buffers retain fractional
    /// samples by default; byte-only custom buffers continue using their
    /// existing component samples.
    pub fn set_sample_precision(&mut self, precision: SamplePrecision) {
        self.sample_precision = precision;
    }

    /// Return the configured RGB conversion precision.
    pub fn sample_precision(&self) -> SamplePrecision {
        self.sample_precision
    }

    /// Set quantization tables for luma and chroma components
    pub fn set_quantization_tables(
        &mut self,
        luma: QuantizationTableType,
        chroma: QuantizationTableType,
    ) {
        self.quantization_tables = [luma, chroma];
    }

    /// Get configured quantization tables
    pub fn quantization_tables(&self) -> &[QuantizationTableType; 2] {
        &self.quantization_tables
    }

    /// Controls if progressive encoding is used.
    ///
    /// Progressive encoding chooses the cheapest scan script per component
    /// ([ProgressiveMode::Optimized]); use [set_progressive_mode](Encoder::set_progressive_mode)
    /// for another layout.
    pub fn set_progressive(&mut self, progressive: bool) {
        self.progressive = if progressive {
            Some(ProgressiveMode::Optimized)
        } else {
            None
        };
    }

    /// Set the progressive scan layout, or `None` for sequential encoding
    pub fn set_progressive_mode(&mut self, mode: Option<ProgressiveMode>) {
        self.progressive = mode;
    }

    /// Return the progressive scan layout if progressive encoding is enabled
    pub fn progressive_mode(&self) -> Option<ProgressiveMode> {
        self.progressive
    }

    /// Set number of scans per component for progressive encoding
    ///
    /// Number of scans must be between 2 and 64.
    /// There is at least one scan for the DC coefficients and one for the remaining 63 AC coefficients.
    ///
    /// # Panics
    /// If number of scans is not within valid range
    pub fn set_progressive_scans(&mut self, scans: u8) {
        assert!(
            (2..=64).contains(&scans),
            "Invalid number of scans: {}",
            scans
        );
        self.progressive = Some(ProgressiveMode::Spectral(scans));
    }

    /// Return number of spectral-selection scans if that progressive layout is enabled
    pub fn progressive_scans(&self) -> Option<u8> {
        match self.progressive {
            Some(ProgressiveMode::Spectral(scans)) => Some(scans),
            _ => None,
        }
    }

    /// Set restart interval
    ///
    /// Set numbers of MCUs between restart markers.
    pub fn set_restart_interval(&mut self, interval: u16) {
        self.restart_interval = if interval == 0 { None } else { Some(interval) };
    }

    /// Return the restart interval
    pub fn restart_interval(&self) -> Option<u16> {
        self.restart_interval
    }

    /// Set if optimized huffman table should be created
    ///
    /// Optimized tables result in slightly smaller file sizes but decrease encoding performance.
    pub fn set_optimized_huffman_tables(&mut self, optimize_huffman_table: bool) {
        self.optimize_huffman_table = optimize_huffman_table;
    }

    /// Returns if optimized huffman table should be generated
    pub fn optimized_huffman_tables(&self) -> bool {
        self.optimize_huffman_table
    }

    /// Enable rate-distortion optimized quantization
    pub fn set_rdo(&mut self, rdo: Option<RdoOptions>) {
        self.rdo = rdo.filter(|r| r.lambda > 0.0);
    }

    /// Returns the rate-distortion optimization options if enabled
    pub fn rdo(&self) -> Option<RdoOptions> {
        self.rdo
    }

    /// Set how the luma and chroma quantization steps follow the quality setting
    pub fn set_step_tuning(&mut self, luma: StepTuning, chroma: StepTuning) {
        self.step_tuning = [luma, chroma, chroma];
    }

    /// Set the step tuning of the two chroma components separately (Cb, Cr);
    /// Cr then gets its own quantization table
    pub fn set_chroma_step_tuning(&mut self, cb: StepTuning, cr: StepTuning) {
        self.step_tuning[1] = cb;
        self.step_tuning[2] = cr;
    }

    /// Set the deadzone of the plain quantizer (ignored when RDO is enabled)
    ///
    /// [Deadzone::PLAIN] gives libjpeg's round-to-nearest.
    pub fn set_deadzone(&mut self, deadzone: Deadzone) {
        self.deadzone = deadzone;
    }

    /// Refine the chroma steps on images with saturated reds (on by default)
    pub fn set_auto_chroma_scale(&mut self, enabled: bool) {
        self.auto_chroma_scale = enabled;
    }

    /// Appends a custom app segment to the JFIF file
    ///
    /// Segment numbers need to be in the range between 1 and 15<br>
    /// The maximum allowed data length is 2^16 - 2 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the segment number is invalid or data exceeds the allowed size
    pub fn add_app_segment(&mut self, segment_nr: u8, data: Vec<u8>) -> Result<(), EncodingError> {
        if segment_nr == 0 || segment_nr > 15 {
            Err(EncodingError::InvalidAppSegment(segment_nr))
        } else if data.len() > 65533 {
            Err(EncodingError::AppSegmentTooLarge(data.len()))
        } else {
            self.app_segments.push((segment_nr, data));
            Ok(())
        }
    }

    /// Add an ICC profile
    ///
    /// The maximum allowed data length is 16,707,345 bytes.
    ///
    /// # Errors
    ///
    /// Returns an Error if the data exceeds the maximum size for the ICC profile
    pub fn add_icc_profile(&mut self, data: &[u8]) -> Result<(), EncodingError> {
        // Based on https://www.color.org/ICC_Minor_Revision_for_Web.pdf
        // B.4  Embedding ICC profiles in JFIF files

        const MARKER: &[u8; 12] = b"ICC_PROFILE\0";
        const MAX_CHUNK_LENGTH: usize = 65535 - 2 - 12 - 2;

        let num_chunks = data.len().div_ceil(MAX_CHUNK_LENGTH);

        // Sequence number is stored as a byte and starts with 1
        if num_chunks >= 255 {
            return Err(EncodingError::IccTooLarge(data.len()));
        }

        for (i, data) in data.chunks(MAX_CHUNK_LENGTH).enumerate() {
            let mut chunk_data = Vec::with_capacity(MAX_CHUNK_LENGTH);
            chunk_data.extend_from_slice(MARKER);
            chunk_data.push(i as u8 + 1);
            chunk_data.push(num_chunks as u8);
            chunk_data.extend_from_slice(data);

            self.add_app_segment(2, chunk_data)?;
        }

        Ok(())
    }

    /// Embeds Exif metadata into the image
    ///
    /// The maximum allowed data length is 65,528 bytes.
    ///
    /// # Errors
    ///
    /// Returns an Error if the data exceeds the maximum size for the Exif metadata
    pub fn add_exif_metadata(&mut self, data: &[u8]) -> Result<(), EncodingError> {
        // E x i f \0 \0
        /// The header for an EXIF APP1 segment
        const EXIF_HEADER: [u8; 6] = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00];

        let mut formatted = EXIF_HEADER.to_vec();
        formatted.extend_from_slice(data);

        self.add_app_segment(1, formatted)
    }

    /// Encode an image
    ///
    /// Data format and length must conform to specified width, height and color type.
    pub fn encode(
        self,
        data: &[u8],
        width: u16,
        height: u16,
        color_type: ColorType,
    ) -> Result<(), EncodingError> {
        let required_data_len = width as usize * height as usize * color_type.get_bytes_per_pixel();

        if data.len() < required_data_len {
            return Err(EncodingError::BadImageData {
                length: data.len(),
                required: required_data_len,
            });
        }

        match color_type {
            ColorType::Luma => self.encode_image(GrayImage(data, width, height)),
            ColorType::Rgb => (self.functions.rgb)(self, data, width, height),
            ColorType::Rgba => (self.functions.rgba)(self, data, width, height),
            ColorType::Bgr => (self.functions.bgr)(self, data, width, height),
            ColorType::Bgra => (self.functions.bgra)(self, data, width, height),
            ColorType::Ycbcr => self.encode_image(YCbCrImage(data, width, height)),
            ColorType::Cmyk => self.encode_image(CmykImage(data, width, height)),
            ColorType::CmykAsYcck => self.encode_image(CmykAsYcckImage(data, width, height)),
            ColorType::Ycck => self.encode_image(YcckImage(data, width, height)),
        }
    }

    /// Encode an image
    pub fn encode_image<I: ImageBuffer>(mut self, image: I) -> Result<(), EncodingError> {
        if image.width() == 0 || image.height() == 0 {
            return Err(EncodingError::ZeroImageDimensions {
                width: image.width(),
                height: image.height(),
            });
        }

        let mut chroma_tuning = [self.step_tuning[1], self.step_tuning[2]];
        if image.get_jpeg_color_type().get_num_components() >= 3
            && self.sampling_factor.get_sampling_factors() == (1, 1)
        {
            for t in &mut chroma_tuning {
                t.ac *= FULL_RESOLUTION_CHROMA_SCALE;
                t.dc *= FULL_RESOLUTION_CHROMA_DC_SCALE;
            }
        } else if image.get_jpeg_color_type() == JpegColorType::Ycbcr {
            chroma_tuning[0].ac *= SUBSAMPLED_CB_SCALE;
        }
        if self.auto_chroma_scale
            && image.get_jpeg_color_type() == JpegColorType::Ycbcr
            && saturated_red(&image)
        {
            for t in &mut chroma_tuning {
                t.ac *= SATURATED_RED_CHROMA_SCALE;
            }
        }

        let q_tables = [
            QuantizationTable::new_with_quality_tuned(
                &self.quantization_tables[0],
                self.quality,
                true,
                &self.step_tuning[0],
            ),
            QuantizationTable::new_with_quality_tuned(
                &self.quantization_tables[1],
                self.quality,
                false,
                &chroma_tuning[0],
            ),
            QuantizationTable::new_with_quality_tuned(
                &self.quantization_tables[1],
                self.quality,
                false,
                &chroma_tuning[1],
            ),
        ];

        let jpeg_color_type = image.get_jpeg_color_type();
        self.init_components(jpeg_color_type);

        self.writer.write_marker(Marker::SOI)?;

        self.writer.write_header(&self.density)?;

        if jpeg_color_type == JpegColorType::Cmyk {
            //Set ColorTransform info to "Unknown"
            let app_14 = b"Adobe\0\0\0\0\0\0\0";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        } else if jpeg_color_type == JpegColorType::Ycck {
            //Set ColorTransform info to YCCK
            let app_14 = b"Adobe\0\0\0\0\0\0\x02";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        }

        for (nr, data) in &self.app_segments {
            self.writer.write_segment(Marker::APP(*nr), data)?;
        }

        if let Some(mode) = self.progressive {
            self.encode_image_progressive_scripted(image, mode, &q_tables)?;
        } else if self.optimize_huffman_table
            || self.rdo.is_some()
            || !self.sampling_factor.supports_interleaved()
            || (self.sample_precision != SamplePrecision::Integer
                && image.supports_precise_samples())
        {
            self.encode_image_sequential(image, &q_tables)?;
        } else {
            self.encode_image_interleaved(image, &q_tables)?;
        }

        self.writer.write_marker(Marker::EOI)?;

        Ok(())
    }

    fn init_components(&mut self, color: JpegColorType) {
        let (horizontal_sampling_factor, vertical_sampling_factor) =
            self.sampling_factor.get_sampling_factors();

        match color {
            JpegColorType::Luma => {
                add_component!(self.components, 0, 0, 1, 1);
            }
            JpegColorType::Ycbcr => {
                add_component!(
                    self.components,
                    0,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
                // Cr gets its own quantization table (the Huffman tables stay shared).
                self.components[2].quantization_table = 2;
            }
            JpegColorType::Cmyk => {
                add_component!(self.components, 0, 1, 1, 1);
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
                add_component!(
                    self.components,
                    3,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
            }
            JpegColorType::Ycck => {
                add_component!(
                    self.components,
                    0,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
                add_component!(
                    self.components,
                    3,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
            }
        }
    }

    fn get_max_sampling_size(&self) -> (usize, usize) {
        let max_h_sampling = self.components.iter().fold(1, |value, component| {
            value.max(component.horizontal_sampling_factor)
        });

        let max_v_sampling = self.components.iter().fold(1, |value, component| {
            value.max(component.vertical_sampling_factor)
        });

        (usize::from(max_h_sampling), usize::from(max_v_sampling))
    }

    fn write_frame_header<I: ImageBuffer>(
        &mut self,
        image: &I,
        q_tables: &[QuantizationTable; 3],
    ) -> Result<(), EncodingError> {
        self.writer.write_frame_header(
            image.width(),
            image.height(),
            &self.components,
            self.progressive.is_some(),
        )?;

        self.writer.write_quantization_segment(0, &q_tables[0])?;
        self.writer.write_quantization_segment(1, &q_tables[1])?;
        if self.components.iter().any(|c| c.quantization_table == 2) {
            self.writer.write_quantization_segment(2, &q_tables[2])?;
        }

        if let Some(restart_interval) = self.restart_interval {
            self.writer.write_dri(restart_interval)?;
        }

        if self.progressive.is_some() {
            // Every progressive scan brings its own Huffman table.
            return Ok(());
        }

        self.writer
            .write_huffman_segment(CodingClass::Dc, 0, &self.huffman_tables[0].0)?;

        self.writer
            .write_huffman_segment(CodingClass::Ac, 0, &self.huffman_tables[0].1)?;

        if image.get_jpeg_color_type().get_num_components() >= 3 {
            self.writer
                .write_huffman_segment(CodingClass::Dc, 1, &self.huffman_tables[1].0)?;

            self.writer
                .write_huffman_segment(CodingClass::Ac, 1, &self.huffman_tables[1].1)?;
        }

        Ok(())
    }

    /// Encode image with a scripted progressive layout (successive approximation,
    /// EOB runs, one optimized Huffman table per scan)
    fn encode_image_progressive_scripted<I: ImageBuffer>(
        &mut self,
        image: I,
        mode: ProgressiveMode,
        q_tables: &[QuantizationTable; 3],
    ) -> Result<(), EncodingError> {
        let (blocks, grids) = self.encode_blocks(&image, q_tables, None);
        let used = self.components.len();
        let blocks: Vec<Vec<AlignedBlock>> = blocks
            .into_iter()
            .take(used)
            .enumerate()
            .map(|(i, blocks)| grids[i].compact_blocks(blocks))
            .collect();
        let luma: Vec<bool> = self
            .components
            .iter()
            .map(|c| c.quantization_table == 0)
            .collect();
        let restart_interval = self.restart_interval.unwrap_or(0) as usize;
        let scripts = progressive::choose_scripts(&blocks, &luma, restart_interval, mode);

        self.write_frame_header(&image, q_tables)?;
        progressive::write_scripts(
            &mut self.writer,
            &blocks,
            &scripts,
            &self.components,
            restart_interval,
        )?;
        Ok(())
    }

    fn init_rows(&mut self, buffer_size: usize) -> [Vec<u8>; 4] {
        // To simplify the code and to give the compiler more infos to optimize stuff we always initialize 4 components
        // Resource overhead should be minimal because an empty Vec doesn't allocate

        match self.components.len() {
            1 => [
                Vec::with_capacity(buffer_size),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            3 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::new(),
            ],
            4 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
            ],
            len => unreachable!("Unsupported component length: {}", len),
        }
    }

    /// Encode all components with one scan
    ///
    /// This is only valid for sampling factors of 1 and 2
    fn encode_image_interleaved<I: ImageBuffer>(
        &mut self,
        image: I,
        q_tables: &[QuantizationTable; 3],
    ) -> Result<(), EncodingError> {
        self.write_frame_header(&image, q_tables)?;
        self.writer
            .write_scan_header(&self.components.iter().collect::<Vec<_>>(), None, (0, 0))?;

        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        let width = image.width();
        let height = image.height();

        let num_cols = usize::from(width).div_ceil(8 * max_h_sampling);
        let num_rows = usize::from(height).div_ceil(8 * max_v_sampling);

        let buffer_width = num_cols * 8 * max_h_sampling;
        let buffer_size = buffer_width * 8 * max_v_sampling;

        let mut row: [Vec<_>; 4] = self.init_rows(buffer_size);

        let mut prev_dc = [0i16; 4];

        let restart_interval = self.restart_interval.unwrap_or(0);
        let mut restarts = 0;
        let mut restarts_to_go = restart_interval;

        for block_y in 0..num_rows {
            for r in &mut row {
                r.clear();
            }

            for y in 0..(8 * max_v_sampling) {
                let y = y + block_y * 8 * max_v_sampling;
                let y = (y.min(height as usize - 1)) as u16;

                image.fill_buffers(y, &mut row);

                for _ in usize::from(width)..buffer_width {
                    for channel in &mut row {
                        if !channel.is_empty() {
                            channel.push(channel[channel.len() - 1]);
                        }
                    }
                }
            }

            for block_x in 0..num_cols {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc[0] = 0;
                    prev_dc[1] = 0;
                    prev_dc[2] = 0;
                    prev_dc[3] = 0;
                }

                for (i, component) in self.components.iter().enumerate() {
                    let h_stride = max_h_sampling / component.horizontal_sampling_factor as usize;
                    let v_stride = max_v_sampling / component.vertical_sampling_factor as usize;
                    let average = self.chroma_subsampling_method
                        == ChromaSubsamplingMethod::Average
                        && (h_stride > 1 || v_stride > 1);

                    for v_offset in 0..component.vertical_sampling_factor as usize {
                        for h_offset in 0..component.horizontal_sampling_factor as usize {
                            let bx = block_x * 8 * max_h_sampling + (h_offset * 8);
                            let by = v_offset * 8;
                            let mut block = if average {
                                get_block_averaged(
                                    &row[i],
                                    bx,
                                    by,
                                    h_stride,
                                    v_stride,
                                    buffer_width,
                                )
                            } else {
                                get_block(&row[i], bx, by, h_stride, v_stride, buffer_width)
                            };

                            (self.functions.fdct)(&mut block);

                            let q_block = self.quantize_plain(
                                &block,
                                &q_tables[component.quantization_table as usize],
                                component.quantization_table == 0,
                            );

                            self.writer.write_block(
                                &q_block,
                                prev_dc[i],
                                &self.huffman_tables[component.dc_huffman_table as usize].0,
                                &self.huffman_tables[component.ac_huffman_table as usize].1,
                            )?;

                            prev_dc[i] = q_block.data[0];
                        }
                    }
                }

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }
        }

        self.writer.finalize_bit_buffer()?;

        Ok(())
    }

    /// Encode components with one scan per component
    fn encode_image_sequential<I: ImageBuffer>(
        &mut self,
        image: I,
        q_tables: &[QuantizationTable; 3],
    ) -> Result<(), EncodingError> {
        let mut final_ac = [[0u32; 257]; 2];
        for freq in &mut final_ac {
            freq[256] = 1;
        }
        let count_ac = self.rdo.is_some();
        let (blocks, grids) =
            self.encode_blocks(&image, q_tables, count_ac.then_some(&mut final_ac));
        let orders = self.coding_orders(&grids);

        if self.optimize_huffman_table || self.rdo.is_some() {
            self.optimize_huffman_table(&blocks, &orders, count_ac.then_some(&final_ac));
        }

        self.write_frame_header(&image, q_tables)?;

        if self.sampling_factor.supports_interleaved() {
            return self.write_interleaved_scan(&blocks, &orders);
        }

        for (i, component) in self.components.iter().enumerate() {
            let restart_interval = self.restart_interval.unwrap_or(0);
            let mut restarts = 0;
            let mut restarts_to_go = restart_interval;

            self.writer.write_scan_header(&[component], None, (0, 0))?;

            let mut prev_dc = 0;

            for &index in &orders[i].indices {
                let block = &blocks[i][index as usize];
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc = 0;
                }

                self.writer.write_block(
                    block,
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                    &self.huffman_tables[component.ac_huffman_table as usize].1,
                )?;

                prev_dc = block.data[0];

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }

            self.writer.finalize_bit_buffer()?;
        }

        Ok(())
    }

    /// Block order of the sequential scan(s): MCU order over the padded grids
    /// when the scan is interleaved, else raster order inside the image
    fn coding_orders(&self, grids: &[BlockGrid; 4]) -> [CodingOrder; 4] {
        let restart_interval = self.restart_interval.unwrap_or(0) as usize;
        let mut orders: [CodingOrder; 4] = Default::default();
        if !self.sampling_factor.supports_interleaved() || self.components.len() == 1 {
            for (i, grid) in grids.iter().enumerate().take(self.components.len()) {
                orders[i] = grid.raster_order(restart_interval);
            }
            return orders;
        }
        let luma = &self.components[0];
        let mcu_cols = grids[0].padded_cols / usize::from(luma.horizontal_sampling_factor);
        let mcu_rows = grids[0].padded_rows / usize::from(luma.vertical_sampling_factor);
        for (i, component) in self.components.iter().enumerate() {
            let h = usize::from(component.horizontal_sampling_factor);
            let v = usize::from(component.vertical_sampling_factor);
            let mut indices = Vec::with_capacity(mcu_cols * mcu_rows * h * v);
            for mcu_y in 0..mcu_rows {
                for mcu_x in 0..mcu_cols {
                    for dy in 0..v {
                        for dx in 0..h {
                            indices.push(
                                ((mcu_y * v + dy) * grids[i].padded_cols + mcu_x * h + dx) as u32,
                            );
                        }
                    }
                }
            }
            orders[i] = CodingOrder {
                indices,
                restart_blocks: restart_interval * h * v,
            };
        }
        orders
    }

    /// One interleaved scan over all components in MCU order
    fn write_interleaved_scan(
        &mut self,
        blocks: &[Vec<AlignedBlock>; 4],
        orders: &[CodingOrder; 4],
    ) -> Result<(), EncodingError> {
        self.writer
            .write_scan_header(&self.components.iter().collect::<Vec<_>>(), None, (0, 0))?;
        let blocks_per_mcu: Vec<usize> = self
            .components
            .iter()
            .map(|c| {
                usize::from(c.horizontal_sampling_factor) * usize::from(c.vertical_sampling_factor)
            })
            .collect();
        let mcus = orders[0].indices.len() / blocks_per_mcu[0];

        let restart_interval = self.restart_interval.unwrap_or(0) as usize;
        let mut restarts = 0u8;
        let mut prev_dc = [0i16; 4];

        for mcu in 0..mcus {
            if restart_interval > 0 && mcu > 0 && mcu.is_multiple_of(restart_interval) {
                self.writer.finalize_bit_buffer()?;
                self.writer.write_marker(Marker::RST(restarts))?;
                restarts = (restarts + 1) & 7;
                prev_dc = [0; 4];
            }
            for (i, component) in self.components.iter().enumerate() {
                let per = blocks_per_mcu[i];
                for &index in &orders[i].indices[mcu * per..mcu * per + per] {
                    let block = &blocks[i][index as usize];
                    self.writer.write_block(
                        block,
                        prev_dc[i],
                        &self.huffman_tables[component.dc_huffman_table as usize].0,
                        &self.huffman_tables[component.ac_huffman_table as usize].1,
                    )?;
                    prev_dc[i] = block.data[0];
                }
            }
        }
        self.writer.finalize_bit_buffer()
    }

    fn encode_blocks<I: ImageBuffer>(
        &mut self,
        image: &I,
        q_tables: &[QuantizationTable; 3],
        final_ac: Option<&mut [[u32; 257]; 2]>,
    ) -> ([Vec<AlignedBlock>; 4], [BlockGrid; 4]) {
        if self.sample_precision != SamplePrecision::Integer && image.supports_precise_samples() {
            return self.encode_blocks_precise(image, q_tables, final_ac);
        }
        let width = image.width();
        let height = image.height();

        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        let num_cols = usize::from(width).div_ceil(8 * max_h_sampling) * max_h_sampling;
        let num_rows = usize::from(height).div_ceil(8 * max_v_sampling) * max_v_sampling;

        debug_assert!(num_cols > 0);
        debug_assert!(num_rows > 0);

        let buffer_width = num_cols * 8;
        // Append each component's raster blocks from one MCU row of samples.
        let stripe_height = max_v_sampling * 8;
        let mut row: [Vec<_>; 4] = self.init_rows(buffer_width * stripe_height);
        let mcu_cols = usize::from(width).div_ceil(8 * max_h_sampling);
        let mcu_rows = usize::from(height).div_ceil(8 * max_v_sampling);

        let mut blocks: [Vec<_>; 4] = self.init_block_buffers(0);
        let mut grids = [BlockGrid::default(); 4];
        // Per-block activity of the source pixels, taken before clip continuation
        // so the continuation's own energy does not count as texture.
        let mut activities: [Vec<f32>; 4] = Default::default();

        for (i, component) in self.components.iter().enumerate() {
            let h_samp = component.horizontal_sampling_factor as usize;
            let v_samp = component.vertical_sampling_factor as usize;

            // Blocks inside the image (what a non-interleaved scan codes) and the
            // MCU-aligned grid that is stored and that an interleaved scan codes.
            let component_width = (usize::from(width) * h_samp).div_ceil(max_h_sampling);
            let component_height = (usize::from(height) * v_samp).div_ceil(max_v_sampling);
            grids[i] = BlockGrid {
                cols: component_width.div_ceil(8),
                rows: component_height.div_ceil(8),
                padded_cols: mcu_cols * h_samp,
                padded_rows: mcu_rows * v_samp,
            };
            blocks[i].reserve_exact(grids[i].padded_cols * grids[i].padded_rows);
        }
        for mcu_y in 0..mcu_rows {
            for channel in &mut row {
                channel.clear();
            }
            for dy in 0..stripe_height {
                let y = (mcu_y * stripe_height + dy).min(usize::from(height) - 1) as u16;
                image.fill_buffers(y, &mut row);

                for _ in usize::from(width)..num_cols * 8 {
                    for channel in &mut row {
                        if !channel.is_empty() {
                            channel.push(channel[channel.len() - 1]);
                        }
                    }
                }
            }
            for (i, component) in self.components.iter().enumerate() {
                let h_samp = usize::from(component.horizontal_sampling_factor);
                let v_samp = usize::from(component.vertical_sampling_factor);
                let h_scale = max_h_sampling / h_samp;
                let v_scale = max_v_sampling / v_samp;
                let subsampled = h_scale > 1 || v_scale > 1;
                let average = self.chroma_subsampling_method == ChromaSubsamplingMethod::Average
                    && subsampled;

                for block_y in 0..v_samp {
                    for block_x in 0..grids[i].padded_cols {
                        let bx = block_x * 8 * h_scale;
                        let by = block_y * 8 * v_scale;
                        let mut block = if average {
                            get_block_averaged(&row[i], bx, by, h_scale, v_scale, buffer_width)
                        } else {
                            get_block(&row[i], bx, by, h_scale, v_scale, buffer_width)
                        };

                        if let Some(rdo) = self.rdo.filter(|rdo| {
                            component.quantization_table == 0 && rdo.clip.iterations > 0
                        }) {
                            activities[i].push(pixel_activity(&block));
                            crate::clip::continue_saturated(
                                &mut block,
                                &q_tables[component.quantization_table as usize],
                                &rdo.clip,
                                &self.functions.float_dct,
                            );
                        }

                        (self.functions.fdct)(&mut block);

                        let luma = component.quantization_table == 0;
                        if self.rdo.is_some() {
                            blocks[i].push(block);
                        } else {
                            blocks[i].push(self.quantize_plain(
                                &block,
                                &q_tables[component.quantization_table as usize],
                                luma,
                            ));
                        }
                    }
                }
            }
        }

        if let Some(rdo) = self.rdo {
            self.rdo_quantize(&mut blocks, q_tables, &rdo, &activities, &grids, final_ac);
        }
        (blocks, grids)
    }

    /// Convert precise RGB-family planes, retaining Q2 coefficients until
    /// quantization. The tables passed to the writer remain in JPEG units.
    fn encode_blocks_precise<I: ImageBuffer>(
        &self,
        image: &I,
        q_tables: &[QuantizationTable; 3],
        mut final_ac: Option<&mut [[u32; 257]; 2]>,
    ) -> ([Vec<AlignedBlock>; 4], [BlockGrid; 4]) {
        let width = usize::from(image.width());
        let height = usize::from(image.height());
        let (max_h, max_v) = self.get_max_sampling_size();
        let mcu_cols = width.div_ceil(8 * max_h);
        let mcu_rows = height.div_ceil(8 * max_v);
        let buffer_width = mcu_cols * max_h * 8;
        // One MCU row supplies all component blocks, preserving the Q8 samples
        // and the extraction filter's exact rounding before reusing the planes.
        let stripe_height = max_v * 8;
        let mut planes: [Vec<u16>; 4] = core::array::from_fn(|i| {
            if i < self.components.len() {
                Vec::with_capacity(buffer_width * stripe_height)
            } else {
                Vec::new()
            }
        });
        let mut grids = [BlockGrid::default(); 4];
        let mut raw: [Vec<[i32; 64]>; 4] = Default::default();
        let mut output: [Vec<AlignedBlock>; 4] = self.init_block_buffers(0);
        let mut activities: [Vec<f32>; 4] = Default::default();
        let mut frequencies = [[0u32; 257]; 2];
        for frequency in &mut frequencies {
            frequency[256] = 1;
        }
        let mut plain = core::array::from_fn::<_, 4, _>(|_| None);
        for (i, component) in self.components.iter().enumerate() {
            let h = usize::from(component.horizontal_sampling_factor);
            let v = usize::from(component.vertical_sampling_factor);
            grids[i] = BlockGrid {
                cols: (width * h).div_ceil(max_h).div_ceil(8),
                rows: (height * v).div_ceil(max_v).div_ceil(8),
                padded_cols: mcu_cols * h,
                padded_rows: mcu_rows * v,
            };
            let count = grids[i].padded_cols * grids[i].padded_rows;
            if self.rdo.is_some() {
                raw[i].reserve(count);
            } else {
                output[i].reserve(count);
            }
            let luma = component.quantization_table == 0;
            let table = &q_tables[component.quantization_table as usize];
            plain[i] = Some(
                table.plain_deadzone(
                    self.deadzone
                        .thresholds_for(luma, self.full_resolution_chroma()),
                ),
            );
        }
        for mcu_y in 0..mcu_rows {
            for plane in &mut planes {
                plane.clear();
            }
            for dy in 0..stripe_height {
                let y = mcu_y * stripe_height + dy;
                assert!(
                    image.fill_buffers_precise(
                        y.min(height - 1) as u16,
                        &mut planes,
                        self.sample_precision,
                    ),
                    "ImageBuffer advertised precise samples but did not supply them"
                );
                for channel in planes.iter_mut().take(self.components.len()) {
                    let edge = *channel.last().expect("missing precise component row");
                    channel.resize(channel.len() + buffer_width - width, edge);
                }
            }
            for (i, component) in self.components.iter().enumerate() {
                let h_scale = max_h / usize::from(component.horizontal_sampling_factor);
                let v_scale = max_v / usize::from(component.vertical_sampling_factor);
                let luma = component.quantization_table == 0;
                let table = &q_tables[component.quantization_table as usize];
                let fractional = luma || self.sample_precision == SamplePrecision::All;
                let average = self.chroma_subsampling_method == ChromaSubsamplingMethod::Average
                    && (h_scale > 1 || v_scale > 1);
                for by in 0..usize::from(component.vertical_sampling_factor) {
                    for bx in 0..grids[i].padded_cols {
                        let mut block = get_block_precise(
                            &planes[i],
                            bx * 8 * h_scale,
                            by * 8 * v_scale,
                            h_scale,
                            v_scale,
                            buffer_width,
                            average,
                            fractional,
                        );
                        if let Some(rdo) = self.rdo.filter(|r| luma && r.clip.iterations > 0) {
                            activities[i].push(pixel_activity(&block) * (1.0 / 16.0));
                            crate::clip::continue_saturated_precise(
                                &mut block,
                                table,
                                &rdo.clip,
                                &self.functions.float_dct,
                            );
                        }
                        let coefficients = (self.functions.fdct_precise)(&block);
                        if self.rdo.is_some() {
                            // Quantize contiguous natural-order data before the
                            // run scan, allowing the compiler to vectorize.
                            let rounded: [i16; 64] = core::array::from_fn(|z| {
                                table.quantize_precise(coefficients[z], z)
                            });
                            let zigzag: [i16; 64] =
                                core::array::from_fn(|k| rounded[ZIGZAG[k] as usize]);
                            count_ac_symbols(
                                &zigzag,
                                &mut frequencies[component.ac_huffman_table as usize],
                            );
                            raw[i].push(coefficients);
                        } else {
                            output[i]
                                .push(plain[i].as_ref().unwrap().quantize_precise(&coefficients));
                        }
                    }
                }
            }
        }
        drop(planes);
        if let Some(rdo) = self.rdo {
            let prices = frequencies.map(|f| AcPrices::from_frequencies(&f));
            for (i, component) in self.components.iter().enumerate() {
                let table = &q_tables[component.quantization_table as usize];
                let luma = component.quantization_table == 0;
                let (channel_weight, tiled) = if luma {
                    (1.0, true)
                } else if self.full_resolution_chroma() {
                    (rdo.chroma_weight_full, rdo.tile_chroma_full)
                } else {
                    (rdo.chroma_weight, rdo.tile_chroma)
                };
                let weights = distortion_weights(table, &rdo, channel_weight, tiled);
                let scales = if rdo.aq_strength > 0.0 && (luma || rdo.aq_chroma) {
                    let measured;
                    let activity = if activities[i].len() == raw[i].len() {
                        &activities[i]
                    } else {
                        measured = raw[i]
                            .iter()
                            .map(|b| {
                                b[1..]
                                    .iter()
                                    .map(|&c| {
                                        let c = c as f32 * 0.25;
                                        c * c
                                    })
                                    .sum()
                            })
                            .collect::<Vec<f32>>();
                        &measured
                    };
                    let mut scales = aq_lambda_scales(activity, &rdo);
                    if luma && rdo.aq_brightness != 0.0 {
                        for (scale, block) in scales.iter_mut().zip(&raw[i]) {
                            // Precise DC is 256 × (mean − 128).
                            let mean = (128.0 + block[0] as f32 / 256.0).clamp(16.0, 255.0);
                            *scale *= crate::rdo::pow_no_libm(mean / 128.0, rdo.aq_brightness)
                                .min(rdo.aq_brightness_max);
                        }
                    }
                    Some(scales)
                } else {
                    None
                };
                let source = core::mem::take(&mut raw[i]);
                output[i].reserve(source.len());
                for (b, block) in source.into_iter().enumerate() {
                    let coded = self.sampling_factor.supports_interleaved()
                        || (b / grids[i].padded_cols < grids[i].rows
                            && b % grids[i].padded_cols < grids[i].cols);
                    let counts = final_ac
                        .as_deref_mut()
                        .filter(|_| coded)
                        .map(|f| &mut f[component.ac_huffman_table as usize]);
                    output[i].push(crate::rdo::rdoq_block_precise(
                        &block,
                        table,
                        &prices[component.ac_huffman_table as usize],
                        &rdo,
                        &weights,
                        scales.as_ref().map_or(1.0, |s| s[b]),
                        counts,
                    ));
                }
            }
        }
        (output, grids)
    }

    /// Whether chroma is coded at luma resolution (4:4:4)
    fn full_resolution_chroma(&self) -> bool {
        self.components.len() >= 3 && self.sampling_factor.get_sampling_factors() == (1, 1)
    }

    /// Quantize one block of DCT coefficients into zigzag order with the plain
    /// quantizer and its deadzone
    fn quantize_plain(
        &self,
        block: &AlignedBlock,
        table: &QuantizationTable,
        luma: bool,
    ) -> AlignedBlock {
        let mut q_block = AlignedBlock::default();
        if self.deadzone == Deadzone::PLAIN {
            quantize_block(block, &mut q_block, table);
        } else {
            let thresholds = self
                .deadzone
                .thresholds_for(luma, self.full_resolution_chroma());
            let natural: [i16; 64] =
                core::array::from_fn(|z| table.quantize_deadzone(block.data[z], z, thresholds));
            for (value, &z) in q_block.data.iter_mut().zip(ZIGZAG.iter()) {
                *value = natural[z as usize & 0x3f];
            }
        }
        q_block
    }

    /// Replace the DCT coefficients in `blocks` by rate-distortion optimized levels
    ///
    /// Pass one quantizes plainly and counts the AC symbols per Huffman table
    /// destination; the trellis then prices every level with the code lengths
    /// those counts imply.
    fn rdo_quantize(
        &self,
        blocks: &mut [Vec<AlignedBlock>; 4],
        q_tables: &[QuantizationTable; 3],
        rdo: &RdoOptions,
        activities: &[Vec<f32>; 4],
        grids: &[BlockGrid; 4],
        mut final_ac: Option<&mut [[u32; 257]; 2]>,
    ) {
        let frequencies = self.rdo_ac_frequencies(blocks, q_tables);
        let max_tables = self.components.len().min(2);
        let prices: Vec<AcPrices> = frequencies[..max_tables]
            .iter()
            .map(AcPrices::from_frequencies)
            .collect();
        for (i, component) in self.components.iter().enumerate() {
            let table = &q_tables[component.quantization_table as usize];
            let prices = &prices[component.ac_huffman_table as usize];
            let luma = component.quantization_table == 0;
            let (channel_weight, tiled) = if luma {
                (1.0, true)
            } else if self.full_resolution_chroma() {
                (rdo.chroma_weight_full, rdo.tile_chroma_full)
            } else {
                (rdo.chroma_weight, rdo.tile_chroma)
            };
            let weights = distortion_weights(table, rdo, channel_weight, tiled);

            let lambda_scales = if rdo.aq_strength > 0.0 && (luma || rdo.aq_chroma) {
                let measured;
                let activity = if activities[i].len() == blocks[i].len() {
                    &activities[i]
                } else {
                    measured = blocks[i].iter().map(block_activity).collect::<Vec<_>>();
                    &measured
                };
                Some(aq_lambda_scales(activity, rdo))
            } else {
                None
            };

            for (b, block) in blocks[i].iter_mut().enumerate() {
                let scale = lambda_scales.as_ref().map_or(1.0, |s| s[b]);
                let coded = self.sampling_factor.supports_interleaved()
                    || (b / grids[i].padded_cols < grids[i].rows
                        && b % grids[i].padded_cols < grids[i].cols);
                let counts = final_ac
                    .as_deref_mut()
                    .filter(|_| coded)
                    .map(|f| &mut f[component.ac_huffman_table as usize]);
                *block = rdoq_block(block, table, prices, rdo, &weights, scale, counts);
            }
        }
    }

    /// Count initial AC symbols with one scratch block, without storing a
    /// second quantized image or computing unused DC histograms and orders.
    fn rdo_ac_frequencies(
        &self,
        blocks: &[Vec<AlignedBlock>; 4],
        q_tables: &[QuantizationTable; 3],
    ) -> [[u32; 257]; 2] {
        let mut frequencies = [[0u32; 257]; 2];
        for freq in &mut frequencies {
            freq[256] = 1;
        }
        for (i, component) in self.components.iter().enumerate() {
            let table = &q_tables[component.quantization_table as usize];
            let freq = &mut frequencies[component.ac_huffman_table as usize];
            let mut quantized = AlignedBlock::default();
            for block in &blocks[i] {
                quantize_block(block, &mut quantized, table);
                count_ac_symbols(&quantized.data, freq);
            }
        }
        frequencies
    }

    fn init_block_buffers(&self, buffer_size: usize) -> [Vec<AlignedBlock>; 4] {
        // To simplify the code and to give the compiler more infos to optimize stuff we always initialize 4 components
        // Resource overhead should be minimal because an empty Vec doesn't allocate

        match self.components.len() {
            1 => [
                Vec::with_capacity(buffer_size),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            3 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::new(),
            ],
            4 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
            ],
            len => unreachable!("Unsupported component length: {}", len),
        }
    }

    // Create new huffman tables optimized for this image
    fn optimize_huffman_table(
        &mut self,
        blocks: &[Vec<AlignedBlock>; 4],
        orders: &[CodingOrder; 4],
        final_ac: Option<&[[u32; 257]; 2]>,
    ) {
        let max_tables = self.components.len().min(2);
        let (dc_freq, ac_freq) = self.collect_frequencies(blocks, orders, final_ac);

        for table in 0..max_tables {
            self.huffman_tables[table] = (
                HuffmanTable::new_optimized(dc_freq[table]),
                HuffmanTable::new_optimized(ac_freq[table]),
            );
        }
    }

    /// Count the DC and AC symbols each huffman table will code, walking every
    /// component's blocks once in their coding order so every DC difference
    /// category the writer needs is in its table
    fn collect_frequencies(
        &self,
        blocks: &[Vec<AlignedBlock>; 4],
        orders: &[CodingOrder; 4],
        final_ac: Option<&[[u32; 257]; 2]>,
    ) -> ([[u32; 257]; 2], [[u32; 257]; 2]) {
        // TODO: Find out if it's possible to reuse some code from the writer
        let mut dc_freq = [[0u32; 257]; 2];
        let mut ac_freq = final_ac.copied().unwrap_or([[0u32; 257]; 2]);
        for freq in dc_freq.iter_mut().chain(&mut ac_freq) {
            freq[256] = 1;
        }

        let mut had_dc = [false; 2];
        let mut had_ac = [false; 2];

        for (i, component) in self.components.iter().enumerate() {
            let dc_table = component.dc_huffman_table as usize;
            let ac_table = component.ac_huffman_table as usize;
            had_dc[dc_table] = true;
            had_ac[ac_table] = true;

            debug_assert!(!blocks[i].is_empty());

            let order = &orders[i];
            let restart_blocks = match order.restart_blocks {
                0 => order.indices.len().max(1),
                n => n,
            };
            // DC prediction restarts with every restart interval
            for interval in order.indices.chunks(restart_blocks) {
                let mut prev_dc = 0;
                for &index in interval {
                    let data = &blocks[i][index as usize].data;
                    dc_freq[dc_table][get_num_bits(data[0] - prev_dc) as usize] += 1;
                    prev_dc = data[0];
                    if final_ac.is_none() {
                        count_ac_symbols(data, &mut ac_freq[ac_table]);
                    }
                }
            }
        }

        for table in 0..self.components.len().min(2) {
            assert!(had_dc[table], "Missing DC data for table {}", table);
            assert!(had_ac[table], "Missing AC data for table {}", table);
        }

        (dc_freq, ac_freq)
    }
}

impl Encoder<BufWriter<File>> {
    /// Create a new decoder that writes into a file
    ///
    /// See [new](Encoder::new) for further information.
    ///
    /// # Errors
    ///
    /// Returns an `IoError(std::io::Error)` if the file can't be created
    pub fn new_file<P: AsRef<Path>>(
        path: P,
        quality: u8,
    ) -> Result<Encoder<BufWriter<File>>, EncodingError> {
        let file = File::create(path)?;
        let buf = BufWriter::new(file);
        Ok(Self::new(buf, quality))
    }
}

fn get_block(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    col_stride: usize,
    row_stride: usize,
    width: usize,
) -> AlignedBlock {
    let mut block = [0i16; 64];

    if col_stride == 1 && row_stride == 1 {
        for (y, out) in block.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let start = (start_y + y) * width + start_x;
            for (value, &pixel) in out.iter_mut().zip(&data[start..start + 8]) {
                *value = (pixel as i16) - 128;
            }
        }
        return AlignedBlock::new(block);
    }

    for y in 0..8 {
        for x in 0..8 {
            let ix = start_x + (x * col_stride);
            let iy = start_y + (y * row_stride);

            block[y * 8 + x] = (data[iy * width + ix] as i16) - 128;
        }
    }

    AlignedBlock::new(block)
}

/// Extract a Q2 spatial block from unsigned Q8 planes.
#[allow(clippy::too_many_arguments)]
fn get_block_precise(
    data: &[u16],
    start_x: usize,
    start_y: usize,
    h_scale: usize,
    v_scale: usize,
    width: usize,
    average: bool,
    fractional: bool,
) -> AlignedBlock {
    let mut block = AlignedBlock::default();
    if h_scale == 1 && v_scale == 1 {
        for (y, out) in block.data.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let start = (start_y + y) * width + start_x;
            for (value, &sample) in out.iter_mut().zip(&data[start..start + 8]) {
                let sample = if fractional {
                    (u32::from(sample) + 31) >> 6
                } else {
                    u32::from(sample >> 8) * 4
                };
                *value = sample as i16 - 512;
            }
        }
        return block;
    }
    if average && h_scale == 2 && v_scale == 2 {
        for (y, out) in block.data.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let start = (start_y + y * 2) * width + start_x;
            let top = &data[start..start + 16];
            let bottom = &data[start + width..start + width + 16];
            for (x, value) in out.iter_mut().enumerate() {
                let i = x * 2;
                let sum = u32::from(top[i])
                    + u32::from(top[i + 1])
                    + u32::from(bottom[i])
                    + u32::from(bottom[i + 1]);
                let sample = if fractional {
                    (sum + 127) >> 8
                } else {
                    (((sum >> 8) + 1 + (x as u32 & 1)) >> 2) * 4
                };
                *value = sample as i16 - 512;
            }
        }
        return block;
    }
    let n = if average { h_scale * v_scale } else { 1 };
    let shift = 6 + n.trailing_zeros();
    for y in 0..8 {
        for x in 0..8 {
            let index = (start_y + y * v_scale) * width + start_x + x * h_scale;
            let mut sum = 0u32;
            if average {
                for dy in 0..v_scale {
                    for dx in 0..h_scale {
                        sum += u32::from(data[index + dy * width + dx]);
                    }
                }
            } else {
                sum = u32::from(data[index]);
            }
            let sample = if fractional {
                ((sum + (1 << (shift - 1)) - 1) >> shift) as i16
            } else {
                let bias = if x & 1 == 0 { (n - 1) / 2 } else { n / 2 };
                ((((sum >> 8) + bias as u32) / n as u32) * 4) as i16
            };
            block.data[y * 8 + x] = sample - 512;
        }
    }
    block
}

fn get_block_averaged(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    col_stride: usize,
    row_stride: usize,
    width: usize,
) -> AlignedBlock {
    let mut block = [0i16; 64];
    // The common 4:2:0 case uses fixed row slices so the compiler can
    // eliminate inner bounds checks and vectorize the pairwise sums.
    if col_stride == 2 && row_stride == 2 {
        for (y, out) in block.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let start = (start_y + y * 2) * width + start_x;
            let top = &data[start..start + 16];
            let bottom = &data[start + width..start + width + 16];
            for (x, value) in out.iter_mut().enumerate() {
                let i = x * 2;
                let sum = u16::from(top[i])
                    + u16::from(top[i + 1])
                    + u16::from(bottom[i])
                    + u16::from(bottom[i + 1]);
                let bias = 1 + (x as u16 & 1);
                *value = ((sum + bias) >> 2) as i16 - 128;
            }
        }
        return AlignedBlock::new(block);
    }
    let n = col_stride * row_stride;
    // Alternate the rounding bias per column as libjpeg does (see jcsample.c)
    let bias_even = (n - 1) / 2;
    let bias_odd = n / 2;

    for y in 0..8 {
        for x in 0..8 {
            let ix = start_x + (x * col_stride);
            let iy = start_y + (y * row_stride);

            let mut sum = 0usize;
            for dy in 0..row_stride {
                for dx in 0..col_stride {
                    sum += data[(iy + dy) * width + (ix + dx)] as usize;
                }
            }

            let bias = if x & 1 == 0 { bias_even } else { bias_odd };
            block[y * 8 + x] = ((sum + bias) / n) as i16 - 128;
        }
    }

    AlignedBlock::new(block)
}

/// Luma AC step multipliers by radial band (u + v 1–2, 3–4, 5–6, 7–9, ≥ 10) on top of
/// the table.
pub static DEFAULT_LUMA_BANDS: [f32; 5] = [1.0, 1.25, 1.25, 0.9, 0.8];

/// Cb AC step multiplier at 4:2:0 (Cr keeps the table): blue-yellow detail
/// is cheaper to drop than red-green at the same perceptual cost.
pub const SUBSAMPLED_CB_SCALE: f32 = 1.25;

/// Chroma AC step multiplier when chroma is coded at full resolution: the
/// tables are balanced for 4:2:0, where each chroma coefficient serves four
/// luma pixels. Kodak holdout, 4:4:4, RDO, vs cjpegli 4:4:4: ×1 +5.5 % SS2 /
/// +17 % BA; ×2 with chroma weight 1 −2.0 / +3.4; ×2 and chroma DC ×0.7
/// −2.5 / +2.3; ×2.3 and DC ×0.7 −2.4 / +1.5; ×2.5 and DC ×0.7 −2.3 / +1.1.
pub const FULL_RESOLUTION_CHROMA_SCALE: f32 = 2.3;

/// Chroma DC step multiplier at full resolution, on top of [StepTuning::dc]
/// (0.5 × 1.4 = 0.7): at 4:2:0 a chroma DC serves four luma blocks, at 4:4:4 one
pub const FULL_RESOLUTION_CHROMA_DC_SCALE: f32 = 1.4;

/// 99th-percentile |Cr − 128| from which an image counts as saturated red
pub const SATURATED_RED_CR: u8 = 90;

/// Chroma AC step multiplier applied on saturated-red images
pub const SATURATED_RED_CHROMA_SCALE: f32 = 0.8;

/// Whether the 99th percentile of |Cr − 128| over every fourth row reaches
/// [SATURATED_RED_CR]
fn saturated_red<I: ImageBuffer>(image: &I) -> bool {
    let mut lanes = [[0u32; 129]; 4];
    let mut rows: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for y in (0..image.height()).step_by(4) {
        for row in &mut rows {
            row.clear();
        }
        image.fill_buffers(y, &mut rows);
        count_cr_magnitudes(&rows[2], &mut lanes);
    }
    let mut histogram = [0u64; 129];
    for lane in &lanes {
        for (total, &count) in histogram.iter_mut().zip(lane) {
            *total += u64::from(count);
        }
    }
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return false;
    }
    let mut below = 0u64;
    for (magnitude, &count) in histogram.iter().enumerate() {
        below += count;
        if below * 100 > total * 99 {
            return magnitude >= SATURATED_RED_CR as usize;
        }
    }
    false
}

/// Add `|Cr − 128|` of one row to four interleaved histograms. Neighboring
/// pixels mostly share a bin; separate lanes keep their increments from
/// waiting on each other. A lane holds at most a quarter of a quarter of
/// 65,535² samples, below `u32::MAX`.
#[inline]
fn count_cr_magnitudes(row: &[u8], lanes: &mut [[u32; 129]; 4]) {
    let (groups, rest) = row.as_chunks::<4>();
    for group in groups {
        for (lane, &cr) in lanes.iter_mut().zip(group) {
            lane[(cr as i32 - 128).unsigned_abs() as usize] += 1;
        }
    }
    for &cr in rest {
        lanes[0][(cr as i32 - 128).unsigned_abs() as usize] += 1;
    }
}

/// AC energy of a block of level-shifted pixels in the scale of the DCT output
fn pixel_activity(block: &AlignedBlock) -> f32 {
    let mean = block.data.iter().map(|&p| p as f32).sum::<f32>() / 64.0;
    64.0 * block
        .data
        .iter()
        .map(|&p| {
            let d = p as f32 - mean;
            d * d
        })
        .sum::<f32>()
}

/// Bit length of `|value|`. `i16::MIN` gives 0, as the former shift loop did
/// after its wrapping negation.
#[inline]
fn get_num_bits(value: i16) -> u8 {
    (15 - ((value.wrapping_abs() << 1) | 1).leading_zeros()) as u8
}

/// Count the AC symbols of one zigzag-ordered block: a run/size symbol per
/// nonzero level, a ZRL per 16 skipped zeros and an EOB unless position 63 is
/// nonzero. Visits only the nonzero positions.
#[inline]
fn count_ac_symbols(data: &[i16; 64], freq: &mut [u32; 257]) {
    let mut mask = crate::writer::nonzero_mask(data) & !1;
    let mut next = 1;
    // ZRLs are summed in a register: adding to freq[0xF0] for every symbol
    // would chain each increment on the previous one's store.
    let mut zrl = 0;
    while mask != 0 {
        let k = mask.trailing_zeros() as usize;
        let run = k - next;
        zrl += run >> 4;
        freq[((run & 15) << 4) | get_num_bits(data[k]) as usize] += 1;
        next = k + 1;
        mask &= mask - 1;
    }
    freq[0xF0] += zrl as u32;
    if next < 64 {
        freq[0] += 1;
    }
}

/// The original coefficient-by-coefficient AC symbol count.
#[cfg(test)]
fn reference_count_ac_symbols(data: &[i16; 64], freq: &mut [u32; 257]) {
    let mut zero_run = 0;
    for &value in &data[1..] {
        if value == 0 {
            zero_run += 1;
        } else {
            while zero_run > 15 {
                freq[0xF0] += 1;
                zero_run -= 16;
            }
            freq[(zero_run << 4) | get_num_bits(value) as usize] += 1;
            zero_run = 0;
        }
    }
    if zero_run > 0 {
        freq[0] += 1;
    }
}

type EncodeRgb<W> = fn(Encoder<W>, &[u8], u16, u16) -> Result<(), EncodingError>;

/// CPU-dependent entry points, resolved once when the encoder is created.
/// RGB entry points retain concrete ImageBuffer types so row conversion stays
/// statically dispatched inside the encoding loops.
struct EncoderFunctions<W: JfifWrite> {
    fdct: fn(&mut AlignedBlock),
    fdct_precise: fn(&AlignedBlock) -> [i32; 64],
    float_dct: Transforms,
    rgb: EncodeRgb<W>,
    rgba: EncodeRgb<W>,
    bgr: EncodeRgb<W>,
    bgra: EncodeRgb<W>,
}

impl<W: JfifWrite> EncoderFunctions<W> {
    fn new() -> Self {
        let functions = Self::scalar(Transforms::new());
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx2") {
            use crate::avx2::*;
            return Self {
                fdct: fdct_avx2,
                rgb: |e, data, w, h| e.encode_image(RgbImageAVX2(data, w, h)),
                rgba: |e, data, w, h| e.encode_image(RgbaImageAVX2(data, w, h)),
                bgr: |e, data, w, h| e.encode_image(BgrImageAVX2(data, w, h)),
                bgra: |e, data, w, h| e.encode_image(BgraImageAVX2(data, w, h)),
                ..functions
            };
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            use crate::neon::*;
            Self {
                rgb: |e, data, w, h| e.encode_image(RgbImageNEON(data, w, h)),
                rgba: |e, data, w, h| e.encode_image(RgbaImageNEON(data, w, h)),
                bgr: |e, data, w, h| e.encode_image(BgrImageNEON(data, w, h)),
                bgra: |e, data, w, h| e.encode_image(BgraImageNEON(data, w, h)),
                ..functions
            }
        }
        #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
        functions
    }

    fn scalar(float_dct: Transforms) -> Self {
        Self {
            fdct: crate::fdct::fdct,
            fdct_precise: crate::fdct::fdct_precise,
            float_dct,
            rgb: |e, data, w, h| e.encode_image(RgbImage(data, w, h)),
            rgba: |e, data, w, h| e.encode_image(RgbaImage(data, w, h)),
            bgr: |e, data, w, h| e.encode_image(BgrImage(data, w, h)),
            bgra: |e, data, w, h| e.encode_image(BgraImage(data, w, h)),
        }
    }
}

#[inline(always)]
fn quantize_block(block: &AlignedBlock, q_block: &mut AlignedBlock, table: &QuantizationTable) {
    let natural: [i16; 64] = core::array::from_fn(|z| table.quantize(block.data[z], z));
    for (value, &z) in q_block.data.iter_mut().zip(ZIGZAG.iter()) {
        *value = natural[z as usize & 0x3f];
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::encoder::{
        count_ac_symbols, count_cr_magnitudes, get_block, get_block_averaged, get_num_bits,
        reference_count_ac_symbols,
    };
    use crate::writer::get_code;
    use crate::{Encoder, SamplingFactor};

    #[test]
    fn test_get_block_averaged_2x2() {
        // Every 2x2 block is {0, 252, 0, 252}; averages to 126.
        let width = 16;
        let mut data = vec![0u8; width * 16];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if (i % width) % 2 == 0 { 0 } else { 252 };
        }

        let nearest = get_block(&data, 0, 0, 2, 2, width);
        let averaged = get_block_averaged(&data, 0, 0, 2, 2, width);

        assert!(nearest.data.iter().all(|&v| v == -128));
        assert!(averaged.data.iter().all(|&v| v == 126 - 128));
    }

    #[test]
    fn test_get_block_averaged_dithers_bias() {
        // Every 2x2 block averages to 1.5; bias dither rounds to 1 on even cols, 2 on odd.
        let width = 16;
        let mut data = vec![0u8; width * 16];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if (i % width) % 2 == 0 { 1 } else { 2 };
        }

        let averaged = get_block_averaged(&data, 0, 0, 2, 2, width);
        assert_eq!(averaged.data[0], 1 - 128);
        assert_eq!(averaged.data[1], 2 - 128);
        assert_eq!(averaged.data[8], 1 - 128);
    }

    #[test]
    fn averaged_2x2_matches_generic_sums_at_offsets() {
        for width in [16, 32, 48, 64] {
            let height = 40;
            let data: alloc::vec::Vec<u8> = (0..width * height)
                .map(|i| ((i * 37 + i / width * 19) % 256) as u8)
                .collect();
            for start_y in [0, 3, height - 16] {
                for start_x in [0, width - 16] {
                    let got = get_block_averaged(&data, start_x, start_y, 2, 2, width);
                    for y in 0..8 {
                        for x in 0..8 {
                            let i = (start_y + y * 2) * width + start_x + x * 2;
                            let sum = usize::from(data[i])
                                + usize::from(data[i + 1])
                                + usize::from(data[i + width])
                                + usize::from(data[i + width + 1]);
                            let expected = ((sum + 1 + (x & 1)) / 4) as i16 - 128;
                            assert_eq!(got.data[y * 8 + x], expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn precise_extraction_matches_independent_filter_sums() {
        let width = 64;
        let height = 64;
        for fractional in [false, true] {
            let data: alloc::vec::Vec<u16> = (0..width * height)
                .map(|i| {
                    let value = ((i * 113 + i / width * 71) % 65536) as u16;
                    if fractional { value } else { value & 0xff00 }
                })
                .collect();
            for (h, v) in [
                (1, 1),
                (2, 1),
                (1, 2),
                (2, 2),
                (4, 1),
                (1, 4),
                (4, 2),
                (2, 4),
            ] {
                for average in [false, true] {
                    for (sx, sy) in [(0, 0), (width - 8 * h, height - 8 * v)] {
                        let got = super::get_block_precise(
                            &data, sx, sy, h, v, width, average, fractional,
                        );
                        for y in 0..8 {
                            for x in 0..8 {
                                let mut sum = 0u32;
                                let (nx, ny) = if average { (h, v) } else { (1, 1) };
                                for dy in 0..ny {
                                    for dx in 0..nx {
                                        let value =
                                            data[(sy + y * v + dy) * width + sx + x * h + dx];
                                        sum += if fractional {
                                            u32::from(value)
                                        } else {
                                            u32::from(value >> 8)
                                        };
                                    }
                                }
                                let n = (nx * ny) as u32;
                                let sample = if fractional {
                                    (sum + n * 32 - 1) / (n * 64)
                                } else {
                                    let bias = if x & 1 == 0 { (n - 1) / 2 } else { n / 2 };
                                    ((sum + bias) / n) * 4
                                };
                                assert_eq!(
                                    got.data[y * 8 + x],
                                    sample as i16 - 512,
                                    "h={h},v={v},average={average},fractional={fractional}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn split_cr_histograms_sum_to_the_single_histogram() {
        let mut seed = 0x5eed_1234u32;
        for length in [0, 1, 3, 4, 5, 7, 8, 63, 64, 65, 768, 1001] {
            for pattern in 0..4 {
                let row: alloc::vec::Vec<u8> = (0..length)
                    .map(|i| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        match pattern {
                            0 => (seed >> 24) as u8,
                            1 => 128,
                            2 => [0, 255, 127, 129][i % 4],
                            _ => (128 + (i % 9) as i32 - 4) as u8,
                        }
                    })
                    .collect();
                let mut expected = [0u64; 129];
                for &cr in &row {
                    expected[(cr as i32 - 128).unsigned_abs() as usize] += 1;
                }
                let mut lanes = [[0u32; 129]; 4];
                count_cr_magnitudes(&row, &mut lanes);
                count_cr_magnitudes(&row, &mut lanes);
                let summed: alloc::vec::Vec<u64> = (0..129)
                    .map(|bin| lanes.iter().map(|lane| u64::from(lane[bin])).sum())
                    .collect();
                let doubled: alloc::vec::Vec<u64> = expected.iter().map(|&c| 2 * c).collect();
                assert_eq!(summed, doubled, "length={length}, pattern={pattern}");
            }
        }
    }

    #[test]
    fn get_num_bits_matches_shift_loop_for_every_i16() {
        fn shift_loop(mut value: i16) -> u8 {
            if value < 0 {
                value = value.wrapping_neg();
            }
            let mut num_bits = 0;
            while value > 0 {
                num_bits += 1;
                value >>= 1;
            }
            num_bits
        }
        for value in i16::MIN..=i16::MAX {
            assert_eq!(get_num_bits(value), shift_loop(value), "value {value}");
        }
    }

    #[test]
    fn sparse_ac_symbol_count_matches_original_loop() {
        let mut fast = [0u32; 257];
        let mut reference = [0u32; 257];
        for data in crate::writer::sparse_scan_blocks(true) {
            let mut one_fast = [0u32; 257];
            let mut one_reference = [0u32; 257];
            count_ac_symbols(&data, &mut one_fast);
            reference_count_ac_symbols(&data, &mut one_reference);
            assert_eq!(one_fast, one_reference, "{data:?}");
            count_ac_symbols(&data, &mut fast);
            reference_count_ac_symbols(&data, &mut reference);
        }
        assert_eq!(fast, reference);
    }

    #[test]
    fn test_get_num_bits() {
        let min_max = 2i16.pow(13);

        for value in -min_max..=min_max {
            let num_bits1 = get_num_bits(value);
            let (num_bits2, _) = get_code(value);

            assert_eq!(
                num_bits1, num_bits2,
                "Difference in num bits for value {}: {} vs {}",
                value, num_bits1, num_bits2
            );
        }
    }

    #[test]
    fn sampling_factors() {
        assert_eq!(SamplingFactor::F_1_1.get_sampling_factors(), (1, 1));
        assert_eq!(SamplingFactor::F_2_1.get_sampling_factors(), (2, 1));
        assert_eq!(SamplingFactor::F_1_2.get_sampling_factors(), (1, 2));
        assert_eq!(SamplingFactor::F_2_2.get_sampling_factors(), (2, 2));
        assert_eq!(SamplingFactor::F_4_1.get_sampling_factors(), (4, 1));
        assert_eq!(SamplingFactor::F_4_2.get_sampling_factors(), (4, 2));
        assert_eq!(SamplingFactor::F_1_4.get_sampling_factors(), (1, 4));
        assert_eq!(SamplingFactor::F_2_4.get_sampling_factors(), (2, 4));

        assert_eq!(SamplingFactor::R_4_4_4.get_sampling_factors(), (1, 1));
        assert_eq!(SamplingFactor::R_4_4_0.get_sampling_factors(), (1, 2));
        assert_eq!(SamplingFactor::R_4_4_1.get_sampling_factors(), (1, 4));
        assert_eq!(SamplingFactor::R_4_2_2.get_sampling_factors(), (2, 1));
        assert_eq!(SamplingFactor::R_4_2_0.get_sampling_factors(), (2, 2));
        assert_eq!(SamplingFactor::R_4_2_1.get_sampling_factors(), (2, 4));
        assert_eq!(SamplingFactor::R_4_1_1.get_sampling_factors(), (4, 1));
        assert_eq!(SamplingFactor::R_4_1_0.get_sampling_factors(), (4, 2));
    }

    #[test]
    fn test_set_progressive() {
        let mut encoder = Encoder::new(vec![], 100);
        encoder.set_progressive(true);
        assert_eq!(
            encoder.progressive_mode(),
            Some(crate::ProgressiveMode::Optimized)
        );
        assert_eq!(encoder.progressive_scans(), None);
        encoder.set_progressive_scans(4);
        assert_eq!(encoder.progressive_scans(), Some(4));

        encoder.set_progressive(false);
        assert_eq!(encoder.progressive_scans(), None);
    }
}

#[cfg(test)]
mod rdo_tests {
    use crate::{ColorType, Encoder, RdoOptions};
    use alloc::vec::Vec;

    #[test]
    fn unpadding_preserves_visible_blocks_and_allocation() {
        use super::{AlignedBlock, BlockGrid};
        for cols in 1..=5 {
            for rows in 1..=5 {
                for extra_cols in 0..=3 {
                    for extra_rows in 0..=3 {
                        let grid = BlockGrid {
                            cols,
                            rows,
                            padded_cols: cols + extra_cols,
                            padded_rows: rows + extra_rows,
                        };
                        let blocks: Vec<_> = (0..grid.padded_cols * grid.padded_rows)
                            .map(|i| AlignedBlock::new([i as i16; 64]))
                            .collect();
                        let allocation = blocks.as_ptr();
                        let expected: Vec<_> = (0..rows)
                            .flat_map(|y| {
                                (0..cols).map(move |x| [(y * grid.padded_cols + x) as i16; 64])
                            })
                            .collect();
                        let actual = grid.compact_blocks(blocks);
                        assert_eq!(actual.as_ptr(), allocation);
                        assert_eq!(actual.iter().map(|b| b.data).collect::<Vec<_>>(), expected);
                    }
                }
            }
        }
    }

    fn image() -> (Vec<u8>, u16, u16) {
        let (w, h) = (64u16, 64u16);
        let mut data = Vec::new();
        for y in 0..h as u32 {
            for x in 0..w as u32 {
                let v = ((x * 7 + y * 3) % 23 * 11) as u8;
                data.extend_from_slice(&[v, (x * y % 255) as u8, 255 - v]);
            }
        }
        (data, w, h)
    }

    #[test]
    fn streamed_initial_ac_counts_match_stored_image() {
        use super::{AlignedBlock, CodingOrder, JpegColorType, quantize_block};
        use crate::quantization::{QuantizationTable, QuantizationTableType};
        for color in [
            JpegColorType::Luma,
            JpegColorType::Ycbcr,
            JpegColorType::Cmyk,
            JpegColorType::Ycck,
        ] {
            for quality in [25, 75, 95] {
                let mut output = Vec::new();
                let mut encoder = Encoder::new(&mut output, quality);
                encoder.init_components(color);
                let tables = core::array::from_fn(|i| {
                    QuantizationTable::new_with_quality_tuned(
                        &QuantizationTableType::Default,
                        quality,
                        i == 0,
                        &crate::StepTuning::PLAIN,
                    )
                });
                let mut raw: [Vec<AlignedBlock>; 4] = Default::default();
                let mut plain: [Vec<AlignedBlock>; 4] = Default::default();
                let mut orders: [CodingOrder; 4] = Default::default();
                for (i, component) in encoder.components.iter().enumerate() {
                    for b in 0..128 {
                        let mut block = AlignedBlock::default();
                        for (z, coefficient) in block.data.iter_mut().enumerate() {
                            let value = ((b * 131 + z * 61 + i * 17) % 8191) as i16 - 4095;
                            *coefficient = if b % 3 == 0 && z % 19 != 0 { 0 } else { value };
                        }
                        let mut q = AlignedBlock::default();
                        quantize_block(
                            &block,
                            &mut q,
                            &tables[component.quantization_table as usize],
                        );
                        raw[i].push(block);
                        plain[i].push(q);
                    }
                    orders[i] = CodingOrder {
                        indices: (0..plain[i].len() as u32).collect(),
                        restart_blocks: 0,
                    };
                }
                let streamed = encoder.rdo_ac_frequencies(&raw, &tables);
                for (table, freq) in streamed
                    .iter()
                    .enumerate()
                    .take(encoder.components.len().min(2))
                {
                    assert_eq!(
                        *freq,
                        encoder.collect_frequencies(&plain, &orders, None).1[table]
                    );
                }
            }
        }
    }

    #[test]
    fn trellis_changes_the_stream_and_decodes() {
        let (data, w, h) = image();
        let mut plain = Vec::new();
        let mut e = Encoder::new(&mut plain, 75);
        e.set_optimized_huffman_tables(true);
        e.encode(&data, w, h, ColorType::Rgb).unwrap();

        let mut rdo = Vec::new();
        let mut e = Encoder::new(&mut rdo, 75);
        e.set_rdo(Some(RdoOptions {
            lambda: 0.3,
            ..RdoOptions::default()
        }));
        assert!(e.rdo().is_some());
        e.encode(&data, w, h, ColorType::Rgb).unwrap();

        assert!(
            rdo.len() < plain.len(),
            "rdo {} vs plain {}",
            rdo.len(),
            plain.len()
        );
        let mut d = jpeg_decoder::Decoder::new(rdo.as_slice());
        d.decode().unwrap();
    }
}

#[cfg(test)]
mod progressive_tests {
    use crate::{ColorType, Encoder, ProgressiveMode, RdoOptions};
    use alloc::vec::Vec;

    fn image() -> (Vec<u8>, u16, u16) {
        let (w, h) = (72u16, 56u16);
        let mut data = Vec::new();
        for y in 0..h as u32 {
            for x in 0..w as u32 {
                let v = ((x * 7 + y * 3) % 23 * 11) as u8;
                data.extend_from_slice(&[v, (x * y % 255) as u8, 255 - v]);
            }
        }
        (data, w, h)
    }

    #[test]
    fn progressive_scripts_decode_to_the_sequential_pixels() {
        let (data, w, h) = image();
        let decode = |bytes: &[u8]| jpeg_decoder::Decoder::new(bytes).decode().unwrap();
        let mut sequential = Vec::new();
        let mut e = Encoder::new(&mut sequential, 60);
        e.set_restart_interval(5);
        e.set_rdo(Some(RdoOptions::default()));
        e.encode(&data, w, h, ColorType::Rgb).unwrap();
        let reference = decode(&sequential);

        for mode in [
            ProgressiveMode::Standard,
            ProgressiveMode::Optimized,
            ProgressiveMode::Spectral(3),
        ] {
            let mut out = Vec::new();
            let mut e = Encoder::new(&mut out, 60);
            e.set_restart_interval(5);
            e.set_rdo(Some(RdoOptions::default()));
            e.set_progressive_mode(Some(mode));
            e.encode(&data, w, h, ColorType::Rgb).unwrap();
            assert_eq!(decode(&out), reference, "{mode:?}");
        }
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    #[test]
    fn resolved_functions_preserve_scalar_bitstreams() {
        for width in [7u16, 33] {
            let height = 9;
            for color in [
                ColorType::Luma,
                ColorType::Rgb,
                ColorType::Rgba,
                ColorType::Bgr,
                ColorType::Bgra,
                ColorType::Ycbcr,
                ColorType::Cmyk,
                ColorType::CmykAsYcck,
                ColorType::Ycck,
            ] {
                let channels = color.get_bytes_per_pixel();
                let data: Vec<u8> = (0..usize::from(width) * usize::from(height) * channels)
                    .map(|i| {
                        // Mix white regions and texture to exercise continuation
                        // as well as conversion tails and image-edge padding.
                        if i / channels % usize::from(width) % 8 < 5 {
                            255
                        } else {
                            ((i * 37 + i / channels * 17) % 256) as u8
                        }
                    })
                    .collect();
                for precision in [
                    SamplePrecision::Integer,
                    SamplePrecision::Luma,
                    SamplePrecision::All,
                ] {
                    for sampling in [SamplingFactor::F_1_1, SamplingFactor::F_2_2] {
                        for mode in 0..4 {
                            let mut outputs = [Vec::new(), Vec::new()];
                            for (scalar, output) in outputs.iter_mut().enumerate() {
                                let mut encoder = Encoder::new(output, 75);
                                if scalar == 1 {
                                    encoder.functions =
                                        EncoderFunctions::scalar(Transforms::scalar());
                                    encoder.functions.fdct = crate::fdct::fdct_scalar;
                                    encoder.functions.fdct_precise = |block| {
                                        crate::fdct::scalar_dct_i64(block).map(|v| v as i32)
                                    };
                                }
                                encoder.set_sample_precision(precision);
                                encoder.set_sampling_factor(sampling);
                                encoder.set_restart_interval(2);
                                match mode {
                                    0 => {} // Streaming interleaved for byte samples.
                                    1 => encoder.set_optimized_huffman_tables(true),
                                    2 => encoder.set_progressive(true),
                                    3 => encoder.set_rdo(Some(RdoOptions::default())),
                                    _ => unreachable!(),
                                }
                                encoder.encode(&data, width, height, color).unwrap();
                            }
                            assert_eq!(
                                outputs[0], outputs[1],
                                "{width} {color:?} {precision:?} {sampling:?} mode={mode}"
                            );
                        }
                    }
                }
            }
        }
    }
}
