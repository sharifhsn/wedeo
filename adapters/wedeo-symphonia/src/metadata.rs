use symphonia::core::meta::{MetadataRevision, StandardTagKey};
use wedeo_core::metadata::Metadata;

/// Convert symphonia metadata to wedeo Metadata.
pub fn convert_metadata(revision: &MetadataRevision) -> Metadata {
    let mut metadata = Metadata::new();

    for tag in revision.tags() {
        let key = if let Some(std_key) = tag.std_key {
            standard_tag_to_ffmpeg_key(std_key)
        } else {
            // Use the raw tag key
            None
        };

        if let Some(key) = key {
            metadata.set(key, tag.value.to_string());
        } else if !tag.key.is_empty() {
            metadata.set(&tag.key, tag.value.to_string());
        }
    }

    metadata
}

/// Map symphonia StandardTagKey to FFmpeg metadata key names.
fn standard_tag_to_ffmpeg_key(key: StandardTagKey) -> Option<&'static str> {
    match key {
        StandardTagKey::Album => Some("album"),
        StandardTagKey::AlbumArtist => Some("album_artist"),
        StandardTagKey::Artist => Some("artist"),
        StandardTagKey::Comment => Some("comment"),
        StandardTagKey::Composer => Some("composer"),
        StandardTagKey::Copyright => Some("copyright"),
        StandardTagKey::Date => Some("date"),
        StandardTagKey::DiscNumber => Some("disc"),
        StandardTagKey::DiscTotal => Some("disc"),
        StandardTagKey::Genre => Some("genre"),
        StandardTagKey::Language => Some("language"),
        StandardTagKey::Performer => Some("performer"),
        StandardTagKey::TrackNumber => Some("track"),
        StandardTagKey::TrackTitle => Some("title"),
        StandardTagKey::TrackTotal => Some("track"),
        StandardTagKey::Encoder => Some("encoder"),
        StandardTagKey::EncodedBy => Some("encoded_by"),
        StandardTagKey::Lyrics => Some("lyrics"),
        StandardTagKey::Bpm => Some("TBPM"),
        StandardTagKey::Description => Some("description"),
        _ => None,
    }
}
