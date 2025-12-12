//! Decoding and Encoding of TIFF Images
//!
//! TIFF (Tagged Image File Format) is a versatile image format that supports
//! lossless and lossy compression.
//!
//! # Related Links
//! * <http://partners.adobe.com/public/developer/tiff/index.html> - The TIFF specification

mod decoder;
mod encoder;

#[allow(deprecated)]
pub use decoder::{TiffDecoder, TiffReader};
pub use encoder::TiffEncoder;
