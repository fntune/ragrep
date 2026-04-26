//! Adaptive rate-limit throttle.
//!
//! Starts conservative (30s for Voyage free tier with 16-chunk batches),
//! decays delay after consecutive successes, backs off on rate-limit, remembers
//! the boundary so it doesn't immediately probe below the failed delay again.
//! The floor decays over time so probing eventually resumes.

use std::thread;
use std::time::Duration;

const DEFAULT_INITIAL: f32 = 30.0;
const DEFAULT_MIN: f32 = 0.2;
const DEFAULT_MAX: f32 = 90.0;

/// Number of consecutive successes that triggers a delay decay.
const SUCCESS_STREAK_FOR_DECAY: u32 = 3;
/// Multiplier applied to delay on success-streak (clamped to floor).
const DECAY_FACTOR: f32 = 0.75;
/// Multiplier applied to delay on rate-limit (when no `retry-after`).
const BACKOFF_FACTOR: f32 = 1.5;
/// Window around floor that counts as "near floor" for erosion (1.1× floor).
const NEAR_FLOOR_RATIO: f32 = 1.1;
/// How many successes near the floor erode it by `FLOOR_DECAY`.
const SUCCESSES_TO_ERODE: u32 = 10;
/// Multiplier applied to floor when eroding.
const FLOOR_DECAY: f32 = 0.85;
/// Seconds added to a server-supplied `retry-after` header value.
const RETRY_AFTER_PAD: f32 = 2.0;

#[derive(Debug)]
pub struct Throttle {
    pub delay: f32,
    min_delay: f32,
    max_delay: f32,
    consecutive_ok: u32,
    floor: f32,
    floor_successes: u32,
}

impl Default for Throttle {
    fn default() -> Self {
        Self::new(DEFAULT_INITIAL, DEFAULT_MIN, DEFAULT_MAX)
    }
}

impl Throttle {
    pub fn new(initial: f32, min_delay: f32, max_delay: f32) -> Self {
        Self {
            delay: initial,
            min_delay,
            max_delay,
            consecutive_ok: 0,
            floor: 0.0,
            floor_successes: 0,
        }
    }

    /// For tests / introspection.
    pub fn floor(&self) -> f32 {
        self.floor
    }

    pub fn on_success(&mut self) {
        self.consecutive_ok += 1;

        // Erode the floor: SUCCESSES_TO_ERODE successes near floor → 0.85× decay
        if self.floor > 0.0 && self.delay <= self.floor * NEAR_FLOOR_RATIO {
            self.floor_successes += 1;
            if self.floor_successes >= SUCCESSES_TO_ERODE {
                self.floor = (self.floor * FLOOR_DECAY).max(self.min_delay);
                self.floor_successes = 0;
            }
        }

        if self.consecutive_ok >= SUCCESS_STREAK_FOR_DECAY {
            let new = (self.delay * DECAY_FACTOR).max(self.min_delay);
            self.delay = new.max(self.floor);
            self.consecutive_ok = 0;
        }
    }

    pub fn on_rate_limit(&mut self, retry_after: Option<f32>) {
        self.consecutive_ok = 0;
        self.floor_successes = 0;
        self.delay = match retry_after {
            Some(s) if s > 0.0 => (s + RETRY_AFTER_PAD).min(self.max_delay),
            _ => (self.delay * BACKOFF_FACTOR).min(self.max_delay),
        };
        // Remember this delay as the new floor.
        self.floor = self.delay;
    }

    pub fn wait(&self) {
        thread::sleep(Duration::from_secs_f32(self.delay));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-streak success at default → delay 30 → 22.5 (0.75x), no floor set.
    #[test]
    fn success_streak_decays_delay() {
        let mut t = Throttle::default();
        let start = t.delay;
        for _ in 0..3 {
            t.on_success();
        }
        assert!((t.delay - start * DECAY_FACTOR).abs() < 1e-3);
        assert_eq!(t.floor, 0.0);
    }

    /// rate-limit with no retry-after → 30 → 45 (1.5x), floor = 45.
    #[test]
    fn rate_limit_backs_off_and_sets_floor() {
        let mut t = Throttle::default();
        t.on_rate_limit(None);
        assert!((t.delay - 30.0 * BACKOFF_FACTOR).abs() < 1e-3);
        assert!((t.floor - t.delay).abs() < 1e-3);
    }

    /// rate-limit with retry-after=10 → delay = 12 (10 + 2 pad), floor = 12.
    #[test]
    fn rate_limit_honors_retry_after() {
        let mut t = Throttle::default();
        t.on_rate_limit(Some(10.0));
        assert!((t.delay - 12.0).abs() < 1e-3);
        assert!((t.floor - 12.0).abs() < 1e-3);
    }

    /// After back-off sets floor, decay can't drop below it.
    #[test]
    fn decay_clamps_to_floor() {
        let mut t = Throttle::default();
        t.on_rate_limit(None); // delay 45, floor 45
                               // 3 streaks of 3 successes each → would be 45 → 33.75 → 25.3 → 19, but clamped at 45
        for _ in 0..9 {
            t.on_success();
        }
        assert!(t.delay >= t.floor - 1e-3);
        assert!((t.delay - 45.0).abs() < 1e-3);
    }

    /// 10 successes near floor → floor decays by 0.85x.
    #[test]
    fn floor_erodes_after_streak_at_floor() {
        let mut t = Throttle::default();
        t.on_rate_limit(None); // floor = 45, delay = 45
        let initial_floor = t.floor;
        // 10 successes near floor → floor erodes
        for _ in 0..10 {
            t.on_success();
        }
        assert!(t.floor < initial_floor);
        assert!((t.floor - initial_floor * FLOOR_DECAY).abs() < 1e-3);
    }
}
