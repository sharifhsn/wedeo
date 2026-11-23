use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Internal time base in ticks per second, matching FFmpeg's `AV_TIME_BASE` (1 MHz).
pub const TIME_BASE: i32 = 1_000_000;

/// Internal time base as a `Rational`, matching FFmpeg's `AV_TIME_BASE_Q`.
pub const TIME_BASE_Q: Rational = Rational::new(1, TIME_BASE);

/// Flag telling rescaling functions to pass `i64::MIN`/`i64::MAX` through unchanged,
/// matching FFmpeg's `AV_ROUND_PASS_MINMAX`.
pub const ROUND_PASS_MINMAX: u32 = 8192;

/// Rounding methods, matching FFmpeg's `AVRounding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Rounding {
    /// Round toward zero.
    Zero = 0,
    /// Round away from zero.
    Inf = 1,
    /// Round toward -infinity.
    Down = 2,
    /// Round toward +infinity.
    Up = 3,
    /// Round to nearest and halfway cases away from zero.
    NearInf = 5,
}

/// Rational number (numerator / denominator), matching FFmpeg's `AVRational`.
#[derive(Clone, Copy, Default)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

impl Rational {
    /// Create a new `Rational`.
    #[inline]
    pub const fn new(num: i32, den: i32) -> Self {
        Self { num, den }
    }

    /// Returns `true` when the denominator is positive (a valid time base or rate).
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.den > 0 && self.num != 0
    }

    /// Convert to `f64`.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Invert the rational (swap `num` and `den`).
    #[inline]
    pub const fn invert(self) -> Self {
        Self {
            num: self.den,
            den: self.num,
        }
    }

    /// Reduce the rational to lowest terms with components bounded by `max`.
    /// Returns `true` if the reduction is exact. Matches FFmpeg's `av_reduce`.
    pub fn reduce(&mut self, max: i64) -> bool {
        let (num, den, exact) = reduce(self.num as i64, self.den as i64, max);
        self.num = num;
        self.den = den;
        exact
    }

    /// Multiply two rationals (FFmpeg `av_mul_q`).
    #[inline]
    pub fn mul_q(self, rhs: Self) -> Self {
        self * rhs
    }

    /// Divide two rationals (FFmpeg `av_div_q`).
    #[inline]
    pub fn div_q(self, rhs: Self) -> Self {
        self / rhs
    }

    /// Add two rationals (FFmpeg `av_add_q`).
    #[inline]
    pub fn add_q(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        cmp_q(*self, *other) == Some(Ordering::Equal)
    }
}

impl Eq for Rational {}

impl std::hash::Hash for Rational {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.den == 0 {
            self.num.signum().hash(state);
            0i32.hash(state);
        } else if self.num == 0 {
            0i32.hash(state);
            1i32.hash(state);
        } else {
            let (n, d, _) = reduce(self.num as i64, self.den as i64, i32::MAX as i64);
            n.hash(state);
            d.hash(state);
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        cmp_q(*self, *other)
    }
}

/// Compare two rationals, matching FFmpeg's `av_cmp_q` behavior.
/// Returns `None` for indeterminate forms (`0/0`).
pub fn cmp_q(a: Rational, b: Rational) -> Option<Ordering> {
    let tmp = a.num as i64 * b.den as i64 - b.num as i64 * a.den as i64;

    if tmp != 0 {
        // FFmpeg: return (int)((tmp ^ a.den ^ b.den) >> 63) | 1;
        let v = ((tmp ^ a.den as i64 ^ b.den as i64) >> 63) | 1;
        if v > 0 {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Less)
        }
    } else if b.den != 0 && a.den != 0 {
        Some(Ordering::Equal)
    } else if a.num != 0 && b.num != 0 {
        let cmp = (a.num >> 31) - (b.num >> 31);
        Some(cmp.cmp(&0))
    } else {
        None
    }
}

/// Reduce a fraction. Returns `(num, den, exact)`. Matches FFmpeg's `av_reduce`.
#[allow(unknown_lints, clippy::manual_checked_ops)]
pub fn reduce(num: i64, den: i64, max: i64) -> (i32, i32, bool) {
    let (mut a0n, mut a0d): (u64, u64) = (0, 1);
    let (mut a1n, mut a1d): (u64, u64) = (1, 0);
    let sign = (num < 0) ^ (den < 0);
    let g = gcd(num.unsigned_abs(), den.unsigned_abs());
    let max = max as u64;

    let mut num = num.unsigned_abs();
    let mut den = den.unsigned_abs();

    if g != 0 {
        num /= g;
        den /= g;
    }

    if num <= max && den <= max {
        a1n = num;
        a1d = den;
        den = 0;
    }

    while den != 0 {
        let x = num / den;
        let next_den = num - den * x;
        let a2n = x * a1n + a0n;
        let a2d = x * a1d + a0d;

        if a2n > max || a2d > max {
            let mut x = x;
            if a1n != 0 {
                x = (max - a0n) / a1n;
            }
            if a1d != 0 {
                x = x.min((max - a0d) / a1d);
            }

            if den * (2 * x * a1d + a0d) > num * a1d {
                a1n = x * a1n + a0n;
                a1d = x * a1d + a0d;
            }
            break;
        }

        a0n = a1n;
        a0d = a1d;
        a1n = a2n;
        a1d = a2d;
        num = den;
        den = next_den;
    }

    let dst_num = if sign {
        (a1n as i32).wrapping_neg()
    } else {
        a1n as i32
    };
    let dst_den = a1d as i32;
    (dst_num, dst_den, den == 0)
}

/// Greatest common divisor (Stein's algorithm), matching FFmpeg's `av_gcd`.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let za = a.trailing_zeros();
    let zb = b.trailing_zeros();
    let k = za.min(zb);
    a >>= za;
    b >>= zb;
    while a != b {
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        b >>= b.trailing_zeros();
    }
    a << k
}

/// Rescale a 64-bit integer with specified rounding (`a * b / c`).
/// Matches FFmpeg's `av_rescale_rnd`.
pub fn rescale_rnd(a: i64, b: i64, c: i64, rnd: Rounding, pass_minmax: bool) -> i64 {
    if c <= 0 || b < 0 {
        return i64::MIN;
    }

    if pass_minmax && (a == i64::MIN || a == i64::MAX) {
        return a;
    }

    if a < 0 {
        let neg_rnd = match rnd {
            Rounding::Down => Rounding::Up,
            Rounding::Up => Rounding::Down,
            Rounding::Inf => Rounding::Inf,
            other => other,
        };
        let abs_a = if a == i64::MIN { i64::MAX } else { -a };
        return (rescale_rnd(abs_a, b, c, neg_rnd, false) as u64).wrapping_neg() as i64;
    }

    let r: i64 = match rnd {
        Rounding::NearInf => c / 2,
        Rounding::Inf | Rounding::Up => c - 1,
        _ => 0,
    };

    if b <= i32::MAX as i64 && c <= i32::MAX as i64 {
        if a <= i32::MAX as i64 {
            return (a * b + r) / c;
        }
        let ad = a / c;
        let a2 = (a % c * b + r) / c;
        if ad >= i32::MAX as i64 && b != 0 && ad > (i64::MAX - a2) / b {
            return i64::MIN;
        }
        return ad * b + a2;
    }

    let product = a as u128 * b as u128 + r as u128;
    let result = product / c as u128;
    if result > i64::MAX as u128 {
        return i64::MIN;
    }
    result as i64
}

/// Rescale a 64-bit integer with rounding to nearest. Matches FFmpeg's `av_rescale`.
#[inline]
pub fn rescale(a: i64, b: i64, c: i64) -> i64 {
    rescale_rnd(a, b, c, Rounding::NearInf, false)
}

/// Rescale a 64-bit integer by two rational numbers with specified rounding.
/// Matches FFmpeg's `av_rescale_q_rnd`.
#[inline]
pub fn rescale_q_rnd(a: i64, bq: Rational, cq: Rational, rnd: Rounding, pass_minmax: bool) -> i64 {
    let b = bq.num as i64 * cq.den as i64;
    let c = cq.num as i64 * bq.den as i64;
    rescale_rnd(a, b, c, rnd, pass_minmax)
}

/// Rescale a 64-bit integer by two rational numbers. Matches FFmpeg's `av_rescale_q`.
#[inline]
pub fn rescale_q(a: i64, bq: Rational, cq: Rational) -> i64 {
    rescale_q_rnd(a, bq, cq, Rounding::NearInf, false)
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            num: self.num.wrapping_neg(),
            den: self.den,
        }
    }
}

impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let num = self.num as i64 * rhs.den as i64 + rhs.num as i64 * self.den as i64;
        let den = self.den as i64 * rhs.den as i64;
        let (n, d, _) = reduce(num, den, i32::MAX as i64);
        Self { num: n, den: d }
    }
}

impl Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let num = self.num as i64 * rhs.num as i64;
        let den = self.den as i64 * rhs.den as i64;
        let (n, d, _) = reduce(num, den, i32::MAX as i64);
        Self { num: n, den: d }
    }
}

impl Div for Rational {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)] // Matches FFmpeg av_div_q(b, c) = av_mul_q(b, av_inv_q(c))
    fn div(self, rhs: Self) -> Self {
        self * rhs.invert()
    }
}

/// Zero-cost transparent wrapper matching FFmpeg's `int64_t` + `AV_NOPTS_VALUE`,
/// while preventing accidental arithmetic on unknown timestamps.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Undefined / unknown timestamp (`i64::MIN`, exact FFmpeg `AV_NOPTS_VALUE`).
    pub const NONE: Self = Self(i64::MIN);

    /// Construct a `Timestamp` from raw ticks (`i64::MIN` represents `NONE`).
    #[inline]
    pub const fn new(ticks: i64) -> Self {
        Self(ticks)
    }

    /// Construct a `Timestamp` from an `Option<i64>`.
    #[inline]
    pub const fn from_option(v: Option<i64>) -> Self {
        match v {
            Some(ticks) => Self(ticks),
            None => Self::NONE,
        }
    }

    /// Returns `true` if the timestamp is known (not `NONE`).
    #[inline]
    pub const fn is_known(self) -> bool {
        self.0 != i64::MIN
    }

    /// Returns `true` if the timestamp is `NONE`.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == i64::MIN
    }

    /// Extract the tick count if known, or `None` if `Timestamp::NONE`.
    #[inline]
    pub const fn get(self) -> Option<i64> {
        if self.0 == i64::MIN {
            None
        } else {
            Some(self.0)
        }
    }

    /// Access the raw `i64` representation (with `i64::MIN` as `NONE`).
    #[inline]
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Checked addition of a signed delta. Returns `None` if `self` is `NONE`,
    /// if addition overflows, or if the result equals `i64::MIN` (`NONE`).
    #[inline]
    pub const fn checked_add(self, delta: i64) -> Option<Self> {
        let Some(v) = self.get() else {
            return None;
        };
        match v.checked_add(delta) {
            Some(res) if res != i64::MIN => Some(Self(res)),
            _ => None,
        }
    }

    /// Checked subtraction of a signed delta. Returns `None` if `self` is `NONE`,
    /// if subtraction overflows, or if the result equals `i64::MIN` (`NONE`).
    #[inline]
    pub const fn checked_sub(self, delta: i64) -> Option<Self> {
        let Some(v) = self.get() else {
            return None;
        };
        match v.checked_sub(delta) {
            Some(res) if res != i64::MIN => Some(Self(res)),
            _ => None,
        }
    }

    /// Signed difference `self - other` if both timestamps are known and do not overflow.
    #[inline]
    pub const fn diff(self, other: Self) -> Option<i64> {
        match (self.get(), other.get()) {
            (Some(a), Some(b)) => a.checked_sub(b),
            _ => None,
        }
    }

    /// Rescale this timestamp from `from` time base to `to` time base using exact
    /// FFmpeg `av_rescale_q_rnd` with `AV_ROUND_PASS_MINMAX`.
    #[inline]
    pub fn rescale(self, from: Rational, to: Rational, rnd: Rounding) -> Self {
        if self.is_none() {
            return Self::NONE;
        }
        Self(rescale_q_rnd(self.0, from, to, rnd, true))
    }

    /// Convert to floating-point seconds given its time base.
    #[inline]
    pub fn to_seconds(self, time_base: Rational) -> Option<f64> {
        if time_base.den == 0 {
            return None;
        }
        self.get().map(|t| t as f64 * time_base.to_f64())
    }
}

impl Default for Timestamp {
    #[inline]
    fn default() -> Self {
        Self::NONE
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "Timestamp({v})"),
            None => f.write_str("Timestamp(NONE)"),
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "{v}"),
            None => f.write_str("NOPTS"),
        }
    }
}

/// Concise wrapping integer arithmetic helpers for exact C `-fwrapv` parity
/// inside ported FFmpeg algorithms.
pub trait WrappingExt: Copy {
    fn wadd(self, rhs: Self) -> Self;
    fn wsub(self, rhs: Self) -> Self;
    fn wmul(self, rhs: Self) -> Self;
    fn wneg(self) -> Self;
}

macro_rules! impl_wrapping_ext {
    ($($t:ty),* $(,)?) => {
        $(
            impl WrappingExt for $t {
                #[inline(always)]
                fn wadd(self, rhs: Self) -> Self {
                    self.wrapping_add(rhs)
                }
                #[inline(always)]
                fn wsub(self, rhs: Self) -> Self {
                    self.wrapping_sub(rhs)
                }
                #[inline(always)]
                fn wmul(self, rhs: Self) -> Self {
                    self.wrapping_mul(rhs)
                }
                #[inline(always)]
                fn wneg(self) -> Self {
                    self.wrapping_neg()
                }
            }
        )*
    };
}

impl_wrapping_ext!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_arithmetic_and_reduce() {
        let a = Rational::new(1, 3);
        let b = Rational::new(1, 6);
        assert_eq!(a + b, Rational::new(1, 2));
        assert_eq!(a - b, Rational::new(1, 6));
        assert_eq!(
            Rational::new(2, 3) * Rational::new(3, 4),
            Rational::new(1, 2)
        );
        assert_eq!(
            Rational::new(2, 3) / Rational::new(4, 3),
            Rational::new(1, 2)
        );

        // Exact continued-fraction bounded reduction matching FFmpeg av_reduce
        let (n, d, exact) = reduce(3_141_592_653, 1_000_000_000, 1000);
        assert!(!exact);
        assert_eq!((n, d), (355, 113));

        // Negation of i32::MIN wraps matching C
        assert_eq!((-Rational::new(i32::MIN, 1)).num, i32::MIN);
    }

    #[test]
    fn rational_cmp_q_edge_cases() {
        assert_eq!(cmp_q(Rational::new(0, 0), Rational::new(0, 0)), None);
        assert_eq!(
            cmp_q(Rational::new(1, 0), Rational::new(-1, 0)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            cmp_q(Rational::new(2, 4), Rational::new(1, 2)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            cmp_q(Rational::new(-1, 2), Rational::new(1, -2)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn rescale_rnd_edge_cases() {
        assert_eq!(rescale(3, 1, 2), 2);
        assert_eq!(rescale_rnd(3, 1, 2, Rounding::Zero, false), 1);
        assert_eq!(rescale_rnd(3, 1, 2, Rounding::Up, false), 2);
        assert_eq!(rescale_rnd(3, 1, 2, Rounding::Down, false), 1);
        // Negative rounding swaps Up/Down
        assert_eq!(rescale_rnd(-3, 1, 2, Rounding::Down, false), -2);
        assert_eq!(rescale_rnd(-3, 1, 2, Rounding::Up, false), -1);
        assert_eq!(rescale_rnd(-3, 1, 2, Rounding::Zero, false), -1);
        assert_eq!(rescale_rnd(-3, 1, 2, Rounding::Inf, false), -2);
        // 128-bit large intermediate path
        assert_eq!(
            rescale_rnd(
                1_000_000_000_000,
                90_000,
                1_000_000,
                Rounding::NearInf,
                false
            ),
            90_000_000_000
        );
        // Invalid divisor/multiplier
        assert_eq!(rescale_rnd(10, 1, 0, Rounding::NearInf, false), i64::MIN);
        assert_eq!(rescale_rnd(10, -1, 2, Rounding::NearInf, false), i64::MIN);
        // Pass minmax
        assert_eq!(
            rescale_rnd(i64::MIN, 1, 2, Rounding::NearInf, true),
            i64::MIN
        );
        assert_eq!(
            rescale_rnd(i64::MAX, 1, 2, Rounding::NearInf, true),
            i64::MAX
        );
    }

    #[test]
    fn timestamp_nopts_and_rescale() {
        assert!(Timestamp::NONE.is_none());
        assert!(!Timestamp::NONE.is_known());
        assert_eq!(Timestamp::NONE.get(), None);
        assert_eq!(Timestamp::NONE.checked_add(10), None);
        assert_eq!(Timestamp::NONE.checked_sub(10), None);
        assert_eq!(
            Timestamp::NONE.rescale(Rational::new(1, 90_000), TIME_BASE_Q, Rounding::NearInf),
            Timestamp::NONE
        );

        let ts = Timestamp::new(90_000);
        assert_eq!(ts.get(), Some(90_000));
        assert_eq!(ts.checked_add(100), Some(Timestamp::new(90_100)));
        assert_eq!(ts.checked_sub(100_000), Some(Timestamp::new(-10_000)));
        // Landing exactly on i64::MIN must return None (reserved for NOPTS)
        assert_eq!(Timestamp::new(i64::MIN + 1).checked_sub(1), None);
        assert_eq!(
            ts.rescale(
                Rational::new(1, 90_000),
                Rational::new(1, 1000),
                Rounding::NearInf
            ),
            Timestamp::new(1000)
        );
        assert_eq!(ts.to_seconds(Rational::new(1, 90_000)), Some(1.0));
    }

    #[test]
    fn rational_hash_matches_eq() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |r: Rational| {
            let mut h = DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        };

        assert_eq!(Rational::new(2, 4), Rational::new(1, 2));
        assert_eq!(hash_of(Rational::new(2, 4)), hash_of(Rational::new(1, 2)));
        assert_eq!(Rational::new(-1, 2), Rational::new(1, -2));
        assert_eq!(hash_of(Rational::new(-1, 2)), hash_of(Rational::new(1, -2)));
        assert_eq!(Rational::new(0, 5), Rational::new(0, 99));
        assert_eq!(hash_of(Rational::new(0, 5)), hash_of(Rational::new(0, 99)));
    }

    #[test]
    fn wrapping_ext_matches_c_fwrapv() {
        assert_eq!(i32::MAX.wadd(1), i32::MIN);
        assert_eq!(i32::MIN.wsub(1), i32::MAX);
        assert_eq!(i32::MIN.wneg(), i32::MIN);
        assert_eq!(200u8.wmul(2), 144u8);
    }
}
