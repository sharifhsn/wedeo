//! Cost-optimal `AudioOpList` optimizer.
//!
//! - Multiplies all linear channel/pan/downmix/gain matrices into a single `M x N` `ChannelMatrix`.
//! - Places `Rematrix` *before* `Resample` when downmixing (`out_ch < in_ch`) and
//!   *after* `Resample` when upmixing (`out_ch > in_ch`).
//! - Elides identity rematrix, identity resample (`in_rate == out_rate`), zero trim,
//!   `AudioDither::None`, and detects pure zero-copy pass-through.

use wedeo_core::SampleTrim;

use crate::ops::{AudioContract, AudioDither, AudioOp};

/// Optimize an `AudioOp` pipeline in place.
pub fn optimize_ops(ops: &mut Vec<AudioOp>) {
    loop {
        let mut changed = false;

        // 1. Elide identity / no-op operations.
        let prev_len = ops.len();
        ops.retain(|op| match op {
            AudioOp::Trim(t) => t.skip_start != 0 || t.skip_end != 0,
            AudioOp::Rematrix(m) => !m.is_identity(),
            AudioOp::Resample(cfg) => cfg.in_rate != cfg.out_rate,
            AudioOp::Dither(d) => *d != AudioDither::None,
            AudioOp::Reframe { frame_size } => *frame_size != 0,
            AudioOp::Read(_) | AudioOp::Write(_) => true,
        });
        if ops.len() != prev_len {
            changed = true;
        }

        // 2. Fuse adjacent ops and commute Rematrix across Resample for cost-optimal ordering.
        let mut i = 0;
        while i + 1 < ops.len() {
            match (&ops[i], &ops[i + 1]) {
                // Fuse consecutive Trim ops.
                (AudioOp::Trim(t1), AudioOp::Trim(t2)) => {
                    let merged = SampleTrim {
                        skip_start: t1.skip_start.saturating_add(t2.skip_start),
                        skip_end: t1.skip_end.saturating_add(t2.skip_end),
                    };
                    ops[i] = AudioOp::Trim(merged);
                    ops.remove(i + 1);
                    changed = true;
                    continue;
                }
                // Fuse consecutive Rematrix ops: M_combined = M_next * M_first.
                (AudioOp::Rematrix(first), AudioOp::Rematrix(next))
                    if first.out_channels == next.in_channels =>
                {
                    if let Ok(fused) = first.then(next) {
                        ops[i] = AudioOp::Rematrix(fused);
                        ops.remove(i + 1);
                        changed = true;
                        continue;
                    }
                }
                // Downmix (or equal-channel gain/swap) after Resample -> move BEFORE Resample.
                (AudioOp::Resample(_), AudioOp::Rematrix(m)) if m.out_channels <= m.in_channels => {
                    ops.swap(i, i + 1);
                    changed = true;
                    continue;
                }
                // Upmix before Resample -> move AFTER Resample.
                (AudioOp::Rematrix(m), AudioOp::Resample(_)) if m.out_channels > m.in_channels => {
                    ops.swap(i, i + 1);
                    changed = true;
                    continue;
                }
                _ => {}
            }

            // Fuse Rematrix(m1) -> Resample(r) -> Rematrix(m2) by commuting across Resample!
            if i + 2 < ops.len()
                && let (AudioOp::Rematrix(m1), AudioOp::Resample(_), AudioOp::Rematrix(m2)) =
                    (&ops[i], &ops[i + 1], &ops[i + 2])
                && m1.out_channels == m2.in_channels
                && let Ok(fused) = m1.then(m2)
            {
                if fused.out_channels > fused.in_channels {
                    // Upmix overall: place fused Rematrix AFTER Resample.
                    ops[i + 2] = AudioOp::Rematrix(fused);
                    ops.remove(i);
                } else {
                    // Downmix or equal-channel overall: place fused Rematrix BEFORE Resample.
                    ops[i] = AudioOp::Rematrix(fused);
                    ops.remove(i + 2);
                }
                changed = true;
                continue;
            }

            i += 1;
        }

        if !changed {
            break;
        }
    }
}

/// Returns `true` if `ops` represents a pure zero-copy pass-through between `src` and `dst`.
pub fn is_noop(src: &AudioContract, dst: &AudioContract, ops: &[AudioOp]) -> bool {
    if src.format != dst.format
        || src.sample_rate != dst.sample_rate
        || src.channel_layout.nb_channels != dst.channel_layout.nb_channels
        || dst.frame_size.is_some()
    {
        return false;
    }

    let mut read_fmt = None;
    let mut write_fmt = None;

    for op in ops {
        match op {
            AudioOp::Read(f) => read_fmt = Some(*f),
            AudioOp::Write(f) => write_fmt = Some(*f),
            AudioOp::Trim(t) if t.skip_start == 0 && t.skip_end == 0 => {}
            AudioOp::Rematrix(m) if m.is_identity() => {}
            AudioOp::Resample(r) if r.in_rate == r.out_rate => {}
            AudioOp::Dither(AudioDither::None) => {}
            AudioOp::Reframe { frame_size: 0 } => {}
            _ => return false,
        }
    }

    match (read_fmt, write_fmt) {
        (Some(r), Some(w)) => r == w,
        _ => true,
    }
}
