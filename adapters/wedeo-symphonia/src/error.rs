use wedeo_core::error::Error;

/// Convert a symphonia error to a wedeo error.
pub fn from_symphonia(e: symphonia::core::errors::Error) -> Error {
    match e {
        symphonia::core::errors::Error::IoError(io_err) => Error::from(io_err),
        symphonia::core::errors::Error::DecodeError(msg) => {
            Error::Other(format!("symphonia decode error: {msg}"))
        }
        symphonia::core::errors::Error::SeekError(kind) => {
            Error::Other(format!("symphonia seek error: {kind:?}"))
        }
        symphonia::core::errors::Error::Unsupported(msg) => {
            Error::Other(format!("symphonia unsupported: {msg}"))
        }
        symphonia::core::errors::Error::LimitError(msg) => {
            Error::Other(format!("symphonia limit error: {msg}"))
        }
        symphonia::core::errors::Error::ResetRequired => {
            Error::Other("symphonia: reset required".into())
        }
    }
}
