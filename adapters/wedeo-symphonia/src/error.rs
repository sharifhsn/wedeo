use wedeo_core::Error;

/// Convert a symphonia error to a `wedeo_core::Error`.
pub fn from_symphonia(component: &'static str, e: symphonia::core::errors::Error) -> Error {
    match e {
        symphonia::core::errors::Error::IoError(io_err) => Error::io(component, &io_err),
        symphonia::core::errors::Error::DecodeError(msg) => {
            Error::invalid_data(component, format!("symphonia decode error: {msg}"))
        }
        symphonia::core::errors::Error::SeekError(kind) => match kind {
            symphonia::core::errors::SeekErrorKind::Unseekable => {
                Error::not_seekable(component, "symphonia stream is not seekable")
            }
            symphonia::core::errors::SeekErrorKind::OutOfRange => {
                Error::invalid_argument(component, "symphonia seek target out of range")
            }
            symphonia::core::errors::SeekErrorKind::ForwardOnly => {
                Error::not_seekable(component, "symphonia stream supports forward seek only")
            }
            symphonia::core::errors::SeekErrorKind::InvalidTrack => {
                Error::not_found(component, "symphonia seek track not found")
            }
        },
        symphonia::core::errors::Error::Unsupported(msg) => {
            Error::unsupported(component, format!("symphonia unsupported: {msg}"))
        }
        symphonia::core::errors::Error::LimitError(msg) => {
            Error::invalid_data(component, format!("symphonia limit error: {msg}"))
        }
        symphonia::core::errors::Error::ResetRequired => {
            Error::invalid_data(component, "symphonia: reset required")
        }
    }
}
