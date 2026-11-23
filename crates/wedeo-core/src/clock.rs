/// Monotonic seek epoch identifier for discarding stale queued frames/packets
/// across asynchronous playback or seek boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SeekEpoch(pub u64);

impl SeekEpoch {
    #[inline]
    pub const fn new(epoch: u64) -> Self {
        Self(epoch)
    }

    /// Return the next epoch value (wrapping on `u64::MAX`).
    #[inline]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

/// Headless media playback clock modeled on `ffplay`'s `pts_drift` clock.
///
/// Tracks `pts_drift = pts - now * speed` so `get(now)` smoothly interpolates
/// presentation time between audio callback updates, supports variable playback
/// speed, and freezes cleanly when paused.
#[derive(Debug, Clone, Copy)]
pub struct AudioClock {
    pts: f64,
    pts_drift: f64,
    last_updated: f64,
    speed: f64,
    paused: bool,
    epoch: SeekEpoch,
}

impl AudioClock {
    /// Create a new clock initialized at `0.0` seconds, playing at `1.0x`.
    pub const fn new() -> Self {
        Self {
            pts: 0.0,
            pts_drift: 0.0,
            last_updated: 0.0,
            speed: 1.0,
            paused: false,
            epoch: SeekEpoch(0),
        }
    }

    /// Anchor the clock to presentation time `pts` (seconds) at monotonic wall time `now` (seconds).
    pub fn set_at(&mut self, pts: f64, now: f64) {
        self.pts = pts;
        self.last_updated = now;
        self.pts_drift = pts - now * self.speed;
    }

    /// Compute the current interpolated presentation time (seconds) at wall time `now`.
    pub fn get(&self, now: f64) -> f64 {
        if self.paused {
            self.pts
        } else {
            self.pts_drift + now * self.speed
        }
    }

    /// Pause or resume the clock at wall time `now`, preserving continuity.
    pub fn set_paused(&mut self, paused: bool, now: f64) {
        if self.paused == paused {
            return;
        }
        let current = self.get(now);
        self.paused = paused;
        self.set_at(current, now);
    }

    /// Change playback speed (clamped to strictly positive values) at wall time `now`
    /// without causing a clock jump.
    pub fn set_speed(&mut self, speed: f64, now: f64) {
        let speed = if speed.is_finite() && speed > 0.0 {
            speed
        } else {
            1.0
        };
        let current = self.get(now);
        self.speed = speed;
        self.set_at(current, now);
    }

    /// Reset the clock for a seek to `pts` at wall time `now` under a new `epoch`.
    pub fn seek(&mut self, pts: f64, now: f64, epoch: SeekEpoch) {
        self.epoch = epoch;
        self.set_at(pts, now);
    }

    #[inline]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    #[inline]
    pub const fn speed(&self) -> f64 {
        self.speed
    }

    #[inline]
    pub const fn epoch(&self) -> SeekEpoch {
        self.epoch
    }

    #[inline]
    pub const fn last_updated(&self) -> f64 {
        self.last_updated
    }
}

impl Default for AudioClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_clock_interpolation_pause_speed_and_seek() {
        let mut clk = AudioClock::new();
        clk.set_at(10.0, 100.0);
        assert!((clk.get(102.5) - 12.5).abs() < 1e-9);

        // Pause freezes clock at 12.5
        clk.set_paused(true, 102.5);
        assert!(clk.is_paused());
        assert!((clk.get(200.0) - 12.5).abs() < 1e-9);

        // Resume continues smoothly from 12.5
        clk.set_paused(false, 200.0);
        assert!((clk.get(201.0) - 13.5).abs() < 1e-9);

        // Speed change to 2.0x has no discontinuity at t = 201.0
        clk.set_speed(2.0, 201.0);
        assert!((clk.get(201.0) - 13.5).abs() < 1e-9);
        assert!((clk.get(203.0) - 17.5).abs() < 1e-9);

        // Seek advances epoch and anchors cleanly
        let next_epoch = clk.epoch().next();
        clk.seek(50.0, 203.0, next_epoch);
        assert_eq!(clk.epoch(), SeekEpoch(1));
        assert!((clk.get(203.5) - 51.0).abs() < 1e-9);
    }
}
