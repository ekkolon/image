use std::io::{self, BufRead, Cursor, Read, Seek, Write};
use std::marker::PhantomData;
use std::mem;

use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use crate::color::{ColorType, ExtendedColorType};
use crate::error::{
    DecodingError, ImageError, ImageResult, LimitError, LimitErrorKind, UnsupportedError,
    UnsupportedErrorKind,
};
use crate::metadata::Orientation;
use crate::{utils, ImageDecoder, ImageFormat};

const TAG_YCBCR_COEFFICIENTS: Tag = Tag::Unknown(529);
const TAG_YCBCR_REFERENCE_BLACK_WHITE: Tag = Tag::Unknown(532);
const TAG_XML_PACKET: Tag = Tag::Unknown(700);

/// Decoder for TIFF images.
pub struct TiffDecoder<R>
where
    R: BufRead + Seek,
{
    dimensions: (u32, u32),
    color_type: ColorType,
    original_color_type: ExtendedColorType,

    // We only use an Option here so we can call with_limits on the decoder without moving.
    inner: Option<Decoder<R>>,
}

impl<R> TiffDecoder<R>
where
    R: BufRead + Seek,
{
    /// Create a new `TiffDecoder`.
    pub fn new(r: R) -> Result<TiffDecoder<R>, ImageError> {
        let mut inner = Decoder::new(r).map_err(ImageError::from_tiff_decode)?;

        let dimensions = inner.dimensions().map_err(ImageError::from_tiff_decode)?;
        let tiff_color_type = inner.colortype().map_err(ImageError::from_tiff_decode)?;

        match inner.find_tag_unsigned_vec::<u16>(Tag::SampleFormat) {
            Ok(Some(sample_formats)) => {
                for format in sample_formats {
                    check_sample_format(format, tiff_color_type)?;
                }
            }
            Ok(None) => { /* assume UInt format */ }
            Err(other) => return Err(ImageError::from_tiff_decode(other)),
        }

        let planar_config = inner
            .find_tag(Tag::PlanarConfiguration)
            .map(|res| res.and_then(|r| r.into_u16().ok()).unwrap_or_default())
            .unwrap_or_default();

        // Decode not supported for non Chunky Planar Configuration
        if planar_config > 1 {
            Err(ImageError::Unsupported(
                UnsupportedError::from_format_and_kind(
                    ImageFormat::Tiff.into(),
                    UnsupportedErrorKind::GenericFeature(String::from("PlanarConfiguration = 2")),
                ),
            ))?;
        }

        let color_type = match tiff_color_type {
            tiff::ColorType::Gray(1) => ColorType::L8,
            tiff::ColorType::Gray(8) => ColorType::L8,
            tiff::ColorType::Gray(16) => ColorType::L16,
            tiff::ColorType::GrayA(8) => ColorType::La8,
            tiff::ColorType::GrayA(16) => ColorType::La16,
            tiff::ColorType::RGB(8) => ColorType::Rgb8,
            tiff::ColorType::RGB(16) => ColorType::Rgb16,
            tiff::ColorType::RGBA(8) => ColorType::Rgba8,
            tiff::ColorType::RGBA(16) => ColorType::Rgba16,
            tiff::ColorType::CMYK(8) => ColorType::Rgb8,
            tiff::ColorType::CMYK(16) => ColorType::Rgb16,
            tiff::ColorType::RGB(32) => ColorType::Rgb32F,
            tiff::ColorType::RGBA(32) => ColorType::Rgba32F,

            tiff::ColorType::Palette(n) | tiff::ColorType::Gray(n) => {
                return Err(err_unknown_color_type(n))
            }
            tiff::ColorType::GrayA(n) => return Err(err_unknown_color_type(n.saturating_mul(2))),
            tiff::ColorType::RGB(n) => return Err(err_unknown_color_type(n.saturating_mul(3))),
            tiff::ColorType::YCbCr(n) => match n {
                8 => ColorType::Rgb8,
                16 => ColorType::Rgb16,
                _ => return Err(err_unknown_color_type(n.saturating_mul(3))),
            },
            tiff::ColorType::RGBA(n) | tiff::ColorType::CMYK(n) => {
                return Err(err_unknown_color_type(n.saturating_mul(4)))
            }
            tiff::ColorType::Multiband {
                bit_depth,
                num_samples,
            } => {
                return Err(err_unknown_color_type(
                    bit_depth.saturating_mul(num_samples.min(255) as u8),
                ))
            }
            _ => return Err(err_unknown_color_type(0)),
        };

        let original_color_type = match tiff_color_type {
            tiff::ColorType::Gray(1) => ExtendedColorType::L1,
            tiff::ColorType::CMYK(8) => ExtendedColorType::Cmyk8,
            tiff::ColorType::CMYK(16) => ExtendedColorType::Cmyk16,
            _ => color_type.into(),
        };

        Ok(TiffDecoder {
            dimensions,
            color_type,
            original_color_type,
            inner: Some(inner),
        })
    }

    // The buffer can be larger for CMYK than the RGB output
    fn total_bytes_buffer(&self) -> u64 {
        let dimensions = self.dimensions();
        let total_pixels = u64::from(dimensions.0) * u64::from(dimensions.1);

        let bytes_per_pixel = match self.original_color_type {
            ExtendedColorType::Cmyk8 => 4,
            ExtendedColorType::Cmyk16 => 8,
            _ => u64::from(self.color_type().bytes_per_pixel()),
        };
        total_pixels.saturating_mul(bytes_per_pixel)
    }
}

fn check_sample_format(sample_format: u16, color_type: tiff::ColorType) -> Result<(), ImageError> {
    use tiff::{tags::SampleFormat, ColorType};
    let num_bits = match color_type {
        ColorType::CMYK(k) => k,
        ColorType::Gray(k) => k,
        ColorType::RGB(k) => k,
        ColorType::RGBA(k) => k,
        ColorType::GrayA(k) => k,
        ColorType::Palette(k) | ColorType::YCbCr(k) => {
            return Err(ImageError::Unsupported(
                UnsupportedError::from_format_and_kind(
                    ImageFormat::Tiff.into(),
                    UnsupportedErrorKind::GenericFeature(format!(
                        "Unhandled TIFF color type {color_type:?} for {k} bits",
                    )),
                ),
            ))
        }
        _ => {
            return Err(ImageError::Unsupported(
                UnsupportedError::from_format_and_kind(
                    ImageFormat::Tiff.into(),
                    UnsupportedErrorKind::GenericFeature(format!(
                        "Unhandled TIFF color type {color_type:?}",
                    )),
                ),
            ))
        }
    };

    match SampleFormat::from_u16(sample_format) {
        Some(SampleFormat::Uint) if num_bits <= 16 => Ok(()),
        Some(SampleFormat::IEEEFP) if num_bits == 32 => Ok(()),
        _ => Err(ImageError::Unsupported(
            UnsupportedError::from_format_and_kind(
                ImageFormat::Tiff.into(),
                UnsupportedErrorKind::GenericFeature(format!(
                    "Unhandled TIFF sample format {sample_format:?} for {num_bits} bits",
                )),
            ),
        )),
    }
}

fn err_unknown_color_type(value: u8) -> ImageError {
    ImageError::Unsupported(UnsupportedError::from_format_and_kind(
        ImageFormat::Tiff.into(),
        UnsupportedErrorKind::Color(ExtendedColorType::Unknown(value)),
    ))
}

impl ImageError {
    fn from_tiff_decode(err: tiff::TiffError) -> ImageError {
        match err {
            tiff::TiffError::IoError(err) => ImageError::IoError(err),
            err @ (tiff::TiffError::FormatError(_)
            | tiff::TiffError::IntSizeError
            | tiff::TiffError::UsageError(_)) => {
                ImageError::Decoding(DecodingError::new(ImageFormat::Tiff.into(), err))
            }
            tiff::TiffError::UnsupportedError(desc) => {
                ImageError::Unsupported(UnsupportedError::from_format_and_kind(
                    ImageFormat::Tiff.into(),
                    UnsupportedErrorKind::GenericFeature(desc.to_string()),
                ))
            }
            tiff::TiffError::LimitsExceeded => {
                ImageError::Limits(LimitError::from_kind(LimitErrorKind::InsufficientMemory))
            }
        }
    }
}

/// Wrapper struct around a `Cursor<Vec<u8>>`
#[allow(dead_code)]
#[deprecated]
pub struct TiffReader<R>(Cursor<Vec<u8>>, PhantomData<R>);
#[allow(deprecated)]
impl<R> Read for TiffReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        if self.0.position() == 0 && buf.is_empty() {
            mem::swap(buf, self.0.get_mut());
            Ok(buf.len())
        } else {
            self.0.read_to_end(buf)
        }
    }
}

const YCBCR_LIMITED_RANGE_U8_DEFAULTS: [f32; 6] = [16.0, 235.0, 16.0, 240.0, 16.0, 240.0];
const YCBCR_LIMITED_RANGE_U16_DEFAULTS: [f32; 6] =
    [4112.0, 60495.0, 4112.0, 61680.0, 4112.0, 61680.0];

impl<R: BufRead + Seek> ImageDecoder for TiffDecoder<R> {
    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn color_type(&self) -> ColorType {
        self.color_type
    }

    fn original_color_type(&self) -> ExtendedColorType {
        self.original_color_type
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        if let Some(decoder) = &mut self.inner {
            Ok(decoder.get_tag_u8_vec(Tag::Unknown(34675)).ok())
        } else {
            Ok(None)
        }
    }

    fn xmp_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        let Some(decoder) = &mut self.inner else {
            return Ok(None);
        };

        let value = match decoder.get_tag(TAG_XML_PACKET) {
            Ok(value) => value,
            Err(tiff::TiffError::FormatError(tiff::TiffFormatError::RequiredTagNotFound(_))) => {
                return Ok(None);
            }
            Err(err) => return Err(ImageError::from_tiff_decode(err)),
        };
        value
            .into_u8_vec()
            .map(Some)
            .map_err(ImageError::from_tiff_decode)
    }

    fn orientation(&mut self) -> ImageResult<Orientation> {
        if let Some(decoder) = &mut self.inner {
            Ok(decoder
                .find_tag(Tag::Orientation)
                .map_err(ImageError::from_tiff_decode)?
                .and_then(|v| Orientation::from_exif(v.into_u16().ok()?.min(255) as u8))
                .unwrap_or(Orientation::NoTransforms))
        } else {
            Ok(Orientation::NoTransforms)
        }
    }

    fn set_limits(&mut self, limits: crate::Limits) -> ImageResult<()> {
        limits.check_support(&crate::LimitSupport::default())?;

        let (width, height) = self.dimensions();
        limits.check_dimensions(width, height)?;

        let max_alloc = limits.max_alloc.unwrap_or(u64::MAX);
        let max_intermediate_alloc = max_alloc.saturating_sub(self.total_bytes_buffer());

        let mut tiff_limits: tiff::decoder::Limits = Default::default();
        tiff_limits.decoding_buffer_size =
            usize::try_from(max_alloc - max_intermediate_alloc).unwrap_or(usize::MAX);
        tiff_limits.intermediate_buffer_size =
            usize::try_from(max_intermediate_alloc).unwrap_or(usize::MAX);
        tiff_limits.ifd_value_size = tiff_limits.intermediate_buffer_size;
        self.inner = Some(self.inner.take().unwrap().with_limits(tiff_limits));

        Ok(())
    }

    fn read_image(self, buf: &mut [u8]) -> ImageResult<()> {
        assert_eq!(u64::try_from(buf.len()), Ok(self.total_bytes()));

        let mut inner = self.inner.unwrap();

        match inner.read_image().map_err(ImageError::from_tiff_decode)? {
            DecodingResult::U8(v) if self.original_color_type == ExtendedColorType::Cmyk8 => {
                let mut out_cur = Cursor::new(buf);
                for cmyk in v.chunks_exact(4) {
                    out_cur.write_all(&cmyk_to_rgb(cmyk))?;
                }
            }
            DecodingResult::U16(v) if self.original_color_type == ExtendedColorType::Cmyk16 => {
                let mut out_cur = Cursor::new(buf);
                for cmyk in v.chunks_exact(4) {
                    out_cur.write_all(bytemuck::cast_slice(&cmyk_to_rgb16(cmyk)))?;
                }
            }
            DecodingResult::U8(v) if self.original_color_type == ExtendedColorType::L1 => {
                let width = self.dimensions.0;
                let row_bytes = width.div_ceil(8);

                for (in_row, out_row) in v
                    .chunks_exact(row_bytes as usize)
                    .zip(buf.chunks_exact_mut(width as usize))
                {
                    out_row.copy_from_slice(&utils::expand_bits(1, width, in_row));
                }
            }
            result @ (DecodingResult::U8(_) | DecodingResult::U16(_))
                if matches!(
                    (self.color_type, &result),
                    (ColorType::Rgb8, DecodingResult::U8(_))
                        | (ColorType::Rgb16, DecodingResult::U16(_))
                ) =>
            {
                let luma = inner
                    .find_tag(TAG_YCBCR_COEFFICIENTS)
                    .ok()
                    .flatten()
                    .and_then(|val| val.into_f32_vec().ok());

                let ref_bw_vec = inner
                    .find_tag(TAG_YCBCR_REFERENCE_BLACK_WHITE)
                    .ok()
                    .flatten()
                    .and_then(|val| val.into_f32_vec().ok());

                let coeffs = luma
                    .as_ref()
                    .map(YCbCrCoefficients::from_luma)
                    .unwrap_or_default();

                let default_ref_bw = match &result {
                    DecodingResult::U8(_) => YCBCR_LIMITED_RANGE_U8_DEFAULTS,
                    DecodingResult::U16(_) => YCBCR_LIMITED_RANGE_U16_DEFAULTS,
                    _ => unreachable!(),
                };

                let ref_bw = ref_bw_vec
                    .as_ref()
                    .and_then(|rbw| {
                        (rbw.len() >= 6).then(|| [rbw[0], rbw[1], rbw[2], rbw[3], rbw[4], rbw[5]])
                    })
                    .unwrap_or(default_ref_bw);

                let mut out_cur = Cursor::new(buf);

                match result {
                    DecodingResult::U8(v) => {
                        for ycbcr_pixel in v.chunks_exact(3) {
                            let rgb = ycbcr_to_rgb(ycbcr_pixel, &coeffs, &ref_bw);
                            out_cur.write_all(&rgb)?;
                        }
                    }
                    DecodingResult::U16(v) => {
                        for ycbcr_pixel in v.chunks_exact(3) {
                            let rgb = ycbcr_to_rgb16(ycbcr_pixel, &coeffs, &ref_bw);
                            out_cur.write_all(bytemuck::cast_slice(&rgb))?;
                        }
                    }
                    _ => unreachable!(),
                }
            }
            DecodingResult::U8(v) => {
                buf.copy_from_slice(&v);
            }
            DecodingResult::U16(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::U32(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::U64(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::I8(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::I16(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::I32(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::I64(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::F32(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::F64(v) => {
                buf.copy_from_slice(bytemuck::cast_slice(&v));
            }
            DecodingResult::F16(_) => unreachable!(),
        }
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }
}

fn cmyk_to_rgb(cmyk: &[u8]) -> [u8; 3] {
    let c = f32::from(cmyk[0]);
    let m = f32::from(cmyk[1]);
    let y = f32::from(cmyk[2]);
    let kf = 1. - f32::from(cmyk[3]) / 255.;
    [
        ((255. - c) * kf) as u8,
        ((255. - m) * kf) as u8,
        ((255. - y) * kf) as u8,
    ]
}

fn cmyk_to_rgb16(cmyk: &[u16]) -> [u16; 3] {
    let c = f32::from(cmyk[0]);
    let m = f32::from(cmyk[1]);
    let y = f32::from(cmyk[2]);
    let kf = 1. - f32::from(cmyk[3]) / 65535.;
    [
        ((65535. - c) * kf) as u16,
        ((65535. - m) * kf) as u16,
        ((65535. - y) * kf) as u16,
    ]
}

/// See: https://libtiff.gitlab.io/libtiff/_sources/functions/TIFFcolor.rst.txt
struct YCbCrCoefficients {
    cr_r: f32,
    cb_b: f32,
    cr_g: f32,
    cb_g: f32,
}

const TIFF_BT_601_COEFF: [f32; 3] = [0.299, 0.587, 0.114];

impl YCbCrCoefficients {
    fn from_luma<L: AsRef<[f32]>>(luma: L) -> Self {
        let luma = luma.as_ref();
        let (kr, kg, kb) = if luma.len() >= 3 {
            (luma[0], luma[1], luma[2])
        } else {
            (
                TIFF_BT_601_COEFF[0],
                TIFF_BT_601_COEFF[1],
                TIFF_BT_601_COEFF[2],
            )
        };

        Self {
            cr_r: 2.0 * (1.0 - kr),
            cb_b: 2.0 * (1.0 - kb),
            cr_g: 2.0 * kr * (1.0 - kr) / kg,
            cb_g: 2.0 * kb * (1.0 - kb) / kg,
        }
    }
}

impl Default for YCbCrCoefficients {
    fn default() -> Self {
        Self::from_luma(&TIFF_BT_601_COEFF)
    }
}

fn ycbcr_to_rgb(ycbcr: &[u8], coeffs: &YCbCrCoefficients, ref_bw: &[f32; 6]) -> [u8; 3] {
    let y = f32::from(ycbcr[0]);
    let cb = f32::from(ycbcr[1]);
    let cr = f32::from(ycbcr[2]);

    let y_norm = (y - ref_bw[0]) / (ref_bw[1] - ref_bw[0]);

    let cb_norm = (cb - ref_bw[2]) / (ref_bw[3] - ref_bw[2]) - 0.5;
    let cr_norm = (cr - ref_bw[4]) / (ref_bw[5] - ref_bw[4]) - 0.5;

    let r = y_norm + coeffs.cr_r * cr_norm;
    let g = y_norm - coeffs.cb_g * cb_norm - coeffs.cr_g * cr_norm;
    let b = y_norm + coeffs.cb_b * cb_norm;

    [
        (r.clamp(0., 1.) * 255.) as u8,
        (g.clamp(0., 1.) * 255.) as u8,
        (b.clamp(0., 1.) * 255.) as u8,
    ]
}

fn ycbcr_to_rgb16(ycbcr: &[u16], coeffs: &YCbCrCoefficients, ref_bw: &[f32; 6]) -> [u16; 3] {
    let y = f32::from(ycbcr[0]);
    let cb = f32::from(ycbcr[1]);
    let cr = f32::from(ycbcr[2]);

    let y_norm = (y - ref_bw[0]) / (ref_bw[1] - ref_bw[0]);

    let cb_norm = (cb - ref_bw[2]) / (ref_bw[3] - ref_bw[2]) - 0.5;
    let cr_norm = (cr - ref_bw[4]) / (ref_bw[5] - ref_bw[4]) - 0.5;

    let r = y_norm + coeffs.cr_r * cr_norm;
    let g = y_norm - coeffs.cb_g * cb_norm - coeffs.cr_g * cr_norm;
    let b = y_norm + coeffs.cb_b * cb_norm;

    [
        (r.clamp(0., 1.) * 65535.) as u16,
        (g.clamp(0., 1.) * 65535.) as u16,
        (b.clamp(0., 1.) * 65535.) as u16,
    ]
}
