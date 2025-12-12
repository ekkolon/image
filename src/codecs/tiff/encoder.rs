use std::io::{Seek, Write};

use crate::color::ExtendedColorType;
use crate::error::{
    EncodingError, ImageError, ImageResult, LimitError, LimitErrorKind, ParameterError,
    ParameterErrorKind, UnsupportedError, UnsupportedErrorKind,
};
use crate::{ImageEncoder, ImageFormat};

impl ImageError {
    fn from_tiff_encode(err: tiff::TiffError) -> ImageError {
        match err {
            tiff::TiffError::IoError(err) => ImageError::IoError(err),
            err @ (tiff::TiffError::FormatError(_)
            | tiff::TiffError::IntSizeError
            | tiff::TiffError::UsageError(_)) => {
                ImageError::Encoding(EncodingError::new(ImageFormat::Tiff.into(), err))
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

/// Encoder for tiff images
pub struct TiffEncoder<W> {
    w: W,
}

/// Convert a slice of sample bytes to its semantic type, being a `Pod`.
fn u8_slice_as_pod<P: bytemuck::Pod>(buf: &[u8]) -> ImageResult<std::borrow::Cow<'_, [P]>> {
    bytemuck::try_cast_slice(buf)
        .map(std::borrow::Cow::Borrowed)
        .or_else(|err| {
            match err {
                bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                    // If the buffer is not aligned for a native slice, copy the buffer into a Vec,
                    // aligning it in the process. This is only done if the element count can be
                    // represented exactly.
                    let vec = bytemuck::allocation::pod_collect_to_vec(buf);
                    Ok(std::borrow::Cow::Owned(vec))
                }
                /* only expecting: bytemuck::PodCastError::OutputSliceWouldHaveSlop */
                _ => {
                    // `bytemuck::PodCastError` of bytemuck-1.2.0 does not implement `Error` and
                    // `Display` trait.
                    // See <https://github.com/Lokathor/bytemuck/issues/22>.
                    Err(ImageError::Parameter(ParameterError::from_kind(
                        ParameterErrorKind::Generic(format!(
                            "Casting samples to their representation failed: {err:?}",
                        )),
                    )))
                }
            }
        })
}

impl<W: Write + Seek> TiffEncoder<W> {
    /// Create a new encoder that writes its output to `w`
    pub fn new(w: W) -> TiffEncoder<W> {
        TiffEncoder { w }
    }

    /// Encodes the image `image` that has dimensions `width` and `height` and `ColorType` `c`.
    ///
    /// 16-bit types assume the buffer is native endian.
    ///
    /// # Panics
    ///
    /// Panics if `width * height * color_type.bytes_per_pixel() != data.len()`.
    #[track_caller]
    pub fn encode(
        self,
        buf: &[u8],
        width: u32,
        height: u32,
        color_type: ExtendedColorType,
    ) -> ImageResult<()> {
        use tiff::encoder::colortype::{
            Gray16, Gray8, RGB32Float, RGBA32Float, RGB16, RGB8, RGBA16, RGBA8,
        };
        let expected_buffer_len = color_type.buffer_size(width, height);
        assert_eq!(
            expected_buffer_len,
            buf.len() as u64,
            "Invalid buffer length: expected {expected_buffer_len} got {} for {width}x{height} image",
            buf.len(),
        );
        let mut encoder =
            tiff::encoder::TiffEncoder::new(self.w).map_err(ImageError::from_tiff_encode)?;
        match color_type {
            ExtendedColorType::L8 => encoder.write_image::<Gray8>(width, height, buf),
            ExtendedColorType::Rgb8 => encoder.write_image::<RGB8>(width, height, buf),
            ExtendedColorType::Rgba8 => encoder.write_image::<RGBA8>(width, height, buf),
            ExtendedColorType::L16 => {
                encoder.write_image::<Gray16>(width, height, u8_slice_as_pod::<u16>(buf)?.as_ref())
            }
            ExtendedColorType::Rgb16 => {
                encoder.write_image::<RGB16>(width, height, u8_slice_as_pod::<u16>(buf)?.as_ref())
            }
            ExtendedColorType::Rgba16 => {
                encoder.write_image::<RGBA16>(width, height, u8_slice_as_pod::<u16>(buf)?.as_ref())
            }
            ExtendedColorType::Rgb32F => encoder.write_image::<RGB32Float>(
                width,
                height,
                u8_slice_as_pod::<f32>(buf)?.as_ref(),
            ),
            ExtendedColorType::Rgba32F => encoder.write_image::<RGBA32Float>(
                width,
                height,
                u8_slice_as_pod::<f32>(buf)?.as_ref(),
            ),
            _ => {
                return Err(ImageError::Unsupported(
                    UnsupportedError::from_format_and_kind(
                        ImageFormat::Tiff.into(),
                        UnsupportedErrorKind::Color(color_type),
                    ),
                ))
            }
        }
        .map_err(ImageError::from_tiff_encode)?;

        Ok(())
    }
}

impl<W: Write + Seek> ImageEncoder for TiffEncoder<W> {
    #[track_caller]
    fn write_image(
        self,
        buf: &[u8],
        width: u32,
        height: u32,
        color_type: ExtendedColorType,
    ) -> ImageResult<()> {
        self.encode(buf, width, height, color_type)
    }
}
