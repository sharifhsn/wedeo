//! Unified Timeline Algebra (`CodedTime ↔ PresentationTime`), edit-list compilation,
//! and GOP-boundary lossless smart-cut planning (`ReencodeHead`, `CopyInterior`, `ReencodeTail`).

use std::ops::Range;

use wedeo_core::{EditSegment, Error, Rational, Result, Rounding, Timestamp, rescale_q};

/// Timestamp in a track's coded media stream coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CodedTime(pub Timestamp);

/// Timestamp in the output linear presentation timeline coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PresentationTime(pub Timestamp);

/// Half-open presentation/coded time interval `[start, end)` in an explicit `time_base`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClipRange {
    pub start: Timestamp,
    pub end: Timestamp,
    pub time_base: Rational,
}

impl ClipRange {
    pub const fn new(start: Timestamp, end: Timestamp, time_base: Rational) -> Self {
        Self {
            start,
            end,
            time_base,
        }
    }

    /// Construct a [`ClipRange`] from raw tick bounds `[start_ticks, end_ticks)` in `time_base`.
    pub const fn from_ticks(start_ticks: i64, end_ticks: i64, time_base: Rational) -> Self {
        Self {
            start: Timestamp::new(start_ticks),
            end: Timestamp::new(end_ticks),
            time_base,
        }
    }

    /// Validate that `start` and `end` are known, `start < end`, and `time_base` is valid.
    pub fn validate(&self) -> Result<()> {
        if !self.time_base.is_valid() {
            return Err(Error::invalid_argument(
                "timeline",
                "ClipRange time_base must have positive denominator and non-zero numerator",
            ));
        }
        let (Some(s), Some(e)) = (self.start.get(), self.end.get()) else {
            return Err(Error::invalid_argument(
                "timeline",
                "ClipRange start and end must be known timestamps",
            ));
        };
        if s >= e {
            return Err(Error::invalid_argument(
                "timeline",
                "ClipRange start must be strictly less than end",
            ));
        }
        Ok(())
    }

    /// Exact duration `end - start` in `self.time_base` ticks.
    #[inline]
    pub const fn duration(&self) -> Option<i64> {
        self.end.diff(self.start)
    }

    /// Rescale this range to `target_tb`.
    pub fn rescale(&self, target_tb: Rational) -> Self {
        if !self.time_base.is_valid() || !target_tb.is_valid() || self.time_base == target_tb {
            return Self {
                start: self.start,
                end: self.end,
                time_base: target_tb,
            };
        }
        Self {
            start: self
                .start
                .rescale(self.time_base, target_tb, Rounding::NearInf),
            end: self
                .end
                .rescale(self.time_base, target_tb, Rounding::NearInf),
            time_base: target_tb,
        }
    }

    /// Returns `true` if `pts` (expressed in `pts_tb`) falls within `[self.start, self.end)`.
    pub fn contains(&self, pts: Timestamp, pts_tb: Rational) -> bool {
        let (Some(p_raw), Some(s_raw), Some(e_raw)) = (pts.get(), self.start.get(), self.end.get())
        else {
            return false;
        };
        let p_in_self =
            if pts_tb.is_valid() && self.time_base.is_valid() && pts_tb != self.time_base {
                rescale_q(p_raw, pts_tb, self.time_base)
            } else {
                p_raw
            };
        p_in_self >= s_raw && p_in_self < e_raw
    }

    /// Intersection of two ranges, returned in `self.time_base`.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let other_r = other.rescale(self.time_base);
        let s = self.start.get()?.max(other_r.start.get()?);
        let e = self.end.get()?.min(other_r.end.get()?);
        if s < e {
            Some(Self::from_ticks(s, e, self.time_base))
        } else {
            None
        }
    }
}

/// Ordered sequence of [`ClipRange`] segments defining a bidirectional mapping between
/// coded media timestamps (`CodedTime`) and linear output timestamps (`PresentationTime`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Timeline {
    pub ranges: Vec<ClipRange>,
}

impl Timeline {
    pub fn new(ranges: Vec<ClipRange>) -> Self {
        Self { ranges }
    }

    pub fn single(range: ClipRange) -> Self {
        Self {
            ranges: vec![range],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Build a [`Timeline`] from container [`EditSegment`] entries (e.g. ISOBMFF `elst`).
    ///
    /// Empty dwell edits (`media_time == Timestamp::NONE`) are skipped in the coded range list.
    pub fn from_edit_list(segments: &[EditSegment]) -> Self {
        let mut ranges = Vec::with_capacity(segments.len());
        for seg in segments {
            if let Some(start) = seg.media_time.get()
                && seg.duration > 0
                && seg.time_base.is_valid()
                && let Some(end) = seg.media_time.checked_add(seg.duration)
            {
                ranges.push(ClipRange::new(Timestamp::new(start), end, seg.time_base));
            }
        }
        Self { ranges }
    }

    /// Export this [`Timeline`] into container [`EditSegment`] entries.
    pub fn to_edit_list(&self) -> Vec<EditSegment> {
        self.ranges
            .iter()
            .filter_map(|r| {
                let dur = r.duration()?;
                if dur <= 0 {
                    return None;
                }
                Some(EditSegment {
                    media_time: r.start,
                    duration: dur,
                    time_base: r.time_base,
                    rate: Rational::new(1, 1),
                })
            })
            .collect()
    }

    /// Total linear presentation duration across all segments, rescaled to `target_tb`.
    pub fn total_duration(&self, target_tb: Rational) -> i64 {
        let mut total = 0i64;
        for r in &self.ranges {
            let rr = r.rescale(target_tb);
            if let Some(d) = rr.duration()
                && d > 0
            {
                total = total.saturating_add(d);
            }
        }
        total
    }

    /// Map a coded timestamp (`CodedTime`) in `tb` into linear presentation time (`PresentationTime`) in `tb`.
    ///
    /// Returns `None` if `coded` lies outside all active [`ClipRange`] segments (trimmed out).
    pub fn coded_to_presentation(
        &self,
        coded: CodedTime,
        tb: Rational,
    ) -> Option<PresentationTime> {
        let coded_raw = coded.0.get()?;
        let mut pres_accum = 0i64;

        for r in &self.ranges {
            let rr = r.rescale(tb);
            let s = rr.start.get()?;
            let e = rr.end.get()?;
            if coded_raw >= s && coded_raw < e {
                let offset = coded_raw - s;
                return Timestamp::new(pres_accum)
                    .checked_add(offset)
                    .map(PresentationTime);
            }
            pres_accum = pres_accum.saturating_add(e.saturating_sub(s));
        }
        None
    }

    /// Map a linear output presentation timestamp (`PresentationTime`) in `tb` back to
    /// `(segment_index, CodedTime)` in `tb`.
    pub fn presentation_to_coded(
        &self,
        pres: PresentationTime,
        tb: Rational,
    ) -> Option<(usize, CodedTime)> {
        let pres_raw = pres.0.get()?;
        if pres_raw < 0 {
            return None;
        }
        let mut cursor = 0i64;

        for (idx, r) in self.ranges.iter().enumerate() {
            let rr = r.rescale(tb);
            let s = rr.start.get()?;
            let e = rr.end.get()?;
            let dur = e.saturating_sub(s);
            if pres_raw >= cursor && pres_raw < cursor.saturating_add(dur) {
                let offset = pres_raw - cursor;
                let coded = Timestamp::new(s).checked_add(offset)?;
                return Some((idx, CodedTime(coded)));
            }
            cursor = cursor.saturating_add(dur);
        }
        None
    }

    /// Compute a [`SmartCutPlan`] for every [`ClipRange`] in this [`Timeline`].
    pub fn plan_smart_cuts(
        &self,
        keyframes: &[KeyframeEntry],
        stream_end: Timestamp,
    ) -> Result<Vec<SmartCutPlan>> {
        let mut plans = Vec::with_capacity(self.ranges.len());
        for &clip in &self.ranges {
            plans.push(SmartCutPlan::compute(clip, keyframes, stream_end)?);
        }
        Ok(plans)
    }
}

impl From<Vec<ClipRange>> for Timeline {
    fn from(ranges: Vec<ClipRange>) -> Self {
        Self::new(ranges)
    }
}

/// Keyframe index entry marking the start of a closed Group of Pictures (GOP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyframeEntry {
    pub pts: Timestamp,
    pub dts: Timestamp,
    pub byte_offset: Option<u64>,
    pub time_base: Rational,
}

impl KeyframeEntry {
    pub const fn new(
        pts: Timestamp,
        dts: Timestamp,
        byte_offset: Option<u64>,
        time_base: Rational,
    ) -> Self {
        Self {
            pts,
            dts,
            byte_offset,
            time_base,
        }
    }

    pub const fn from_pts(pts_ticks: i64, time_base: Rational) -> Self {
        Self {
            pts: Timestamp::new(pts_ticks),
            dts: Timestamp::new(pts_ticks),
            byte_offset: None,
            time_base,
        }
    }
}

/// Actionable GOP slice in a [`SmartCutPlan`]:
///
/// - [`GopSegment::ReencodeHead`]: Cut starts mid-GOP; decode from `decode_from` keyframe and
///   re-encode frames in `[pts_start, pts_end)`.
/// - [`GopSegment::CopyInterior`]: One or more complete GOPs inside `[pts_start, pts_end)` that
///   can be losslessly stream-copied without decoding.
/// - [`GopSegment::ReencodeTail`]: Cut ends mid-GOP; decode from `decode_from` keyframe and
///   re-encode frames in `[pts_start, pts_end)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GopSegment {
    ReencodeHead {
        decode_from: Timestamp,
        pts_start: Timestamp,
        pts_end: Timestamp,
        gop_index: usize,
        time_base: Rational,
    },
    CopyInterior {
        pts_start: Timestamp,
        pts_end: Timestamp,
        gop_range: Range<usize>,
        time_base: Rational,
    },
    ReencodeTail {
        decode_from: Timestamp,
        pts_start: Timestamp,
        pts_end: Timestamp,
        gop_index: usize,
        time_base: Rational,
    },
}

impl GopSegment {
    pub const fn pts_start(&self) -> Timestamp {
        match self {
            Self::ReencodeHead { pts_start, .. }
            | Self::CopyInterior { pts_start, .. }
            | Self::ReencodeTail { pts_start, .. } => *pts_start,
        }
    }

    pub const fn pts_end(&self) -> Timestamp {
        match self {
            Self::ReencodeHead { pts_end, .. }
            | Self::CopyInterior { pts_end, .. }
            | Self::ReencodeTail { pts_end, .. } => *pts_end,
        }
    }

    pub const fn time_base(&self) -> Rational {
        match self {
            Self::ReencodeHead { time_base, .. }
            | Self::CopyInterior { time_base, .. }
            | Self::ReencodeTail { time_base, .. } => *time_base,
        }
    }

    pub const fn is_copy(&self) -> bool {
        matches!(self, Self::CopyInterior { .. })
    }

    pub const fn is_reencode(&self) -> bool {
        matches!(self, Self::ReencodeHead { .. } | Self::ReencodeTail { .. })
    }
}

/// Lossless GOP smart-cut execution plan decomposing a [`ClipRange`] into at most three regions:
/// `ReencodeHead` (optional), `CopyInterior` (optional, coalesced across contiguous interior GOPs),
/// and `ReencodeTail` (optional).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartCutPlan {
    pub clip: ClipRange,
    pub segments: Vec<GopSegment>,
}

impl SmartCutPlan {
    /// Compute the [`SmartCutPlan`] for `clip` given a sorted list of keyframe GOP boundaries
    /// and the total `stream_end` timestamp.
    pub fn compute(
        clip: ClipRange,
        keyframes: &[KeyframeEntry],
        stream_end: Timestamp,
    ) -> Result<Self> {
        clip.validate()?;
        if keyframes.is_empty() {
            return Err(Error::invalid_argument(
                "smart_cut",
                "keyframes list must not be empty",
            ));
        }

        let tb = clip.time_base;
        let cut_start = clip.start.raw();
        let cut_end = clip.end.raw();

        // Rescale all keyframe PTS values into `clip.time_base`.
        let mut kf_pts: Vec<i64> = Vec::with_capacity(keyframes.len());
        for kf in keyframes {
            let Some(raw) = kf.pts.get() else {
                return Err(Error::invalid_argument(
                    "smart_cut",
                    "keyframe PTS must be known",
                ));
            };
            let rescaled = if kf.time_base.is_valid() && kf.time_base != tb {
                rescale_q(raw, kf.time_base, tb)
            } else {
                raw
            };
            kf_pts.push(rescaled);
        }

        for pair in kf_pts.windows(2) {
            if pair[1] <= pair[0] {
                return Err(Error::invalid_argument(
                    "smart_cut",
                    "keyframes must be strictly monotonically increasing",
                ));
            }
        }

        let end_stream_raw = stream_end
            .get()
            .unwrap_or_else(|| cut_end.max(*kf_pts.last().unwrap() + 1));

        let mut segments: Vec<GopSegment> = Vec::new();
        let mut copy_start_pts: Option<i64> = None;
        let mut copy_end_pts: i64 = 0;
        let mut copy_gop_start: usize = 0;
        let mut copy_gop_end: usize = 0;

        for gop_idx in 0..kf_pts.len() {
            let gop_s = kf_pts[gop_idx];
            let gop_e = if gop_idx + 1 < kf_pts.len() {
                kf_pts[gop_idx + 1]
            } else {
                end_stream_raw.max(gop_s + 1)
            };

            // Skip GOPs completely outside `[cut_start, cut_end)`
            if gop_e <= cut_start || gop_s >= cut_end {
                continue;
            }

            let starts_inside_gop = cut_start > gop_s;
            let ends_inside_gop = cut_end < gop_e;

            if starts_inside_gop {
                // Partial head GOP (or single-GOP cut where both start and end lie inside this GOP)
                let seg_end = cut_end.min(gop_e);
                segments.push(GopSegment::ReencodeHead {
                    decode_from: Timestamp::new(gop_s),
                    pts_start: Timestamp::new(cut_start),
                    pts_end: Timestamp::new(seg_end),
                    gop_index: gop_idx,
                    time_base: tb,
                });
            } else if !ends_inside_gop {
                // Full interior GOP `[gop_s, gop_e)` completely inside `[cut_start, cut_end)`
                if copy_start_pts.is_none() {
                    copy_start_pts = Some(gop_s);
                    copy_gop_start = gop_idx;
                }
                copy_end_pts = gop_e;
                copy_gop_end = gop_idx + 1;
            } else {
                // Flush any accumulated CopyInterior before emitting ReencodeTail
                if let Some(cs) = copy_start_pts.take() {
                    segments.push(GopSegment::CopyInterior {
                        pts_start: Timestamp::new(cs),
                        pts_end: Timestamp::new(copy_end_pts),
                        gop_range: copy_gop_start..copy_gop_end,
                        time_base: tb,
                    });
                }
                segments.push(GopSegment::ReencodeTail {
                    decode_from: Timestamp::new(gop_s),
                    pts_start: Timestamp::new(gop_s),
                    pts_end: Timestamp::new(cut_end),
                    gop_index: gop_idx,
                    time_base: tb,
                });
            }
        }

        if let Some(cs) = copy_start_pts.take() {
            segments.push(GopSegment::CopyInterior {
                pts_start: Timestamp::new(cs),
                pts_end: Timestamp::new(copy_end_pts),
                gop_range: copy_gop_start..copy_gop_end,
                time_base: tb,
            });
        }

        Ok(Self { clip, segments })
    }

    pub fn head(&self) -> Option<&GopSegment> {
        self.segments
            .iter()
            .find(|s| matches!(s, GopSegment::ReencodeHead { .. }))
    }

    pub fn interior(&self) -> Option<&GopSegment> {
        self.segments
            .iter()
            .find(|s| matches!(s, GopSegment::CopyInterior { .. }))
    }

    pub fn tail(&self) -> Option<&GopSegment> {
        self.segments
            .iter()
            .find(|s| matches!(s, GopSegment::ReencodeTail { .. }))
    }
}
