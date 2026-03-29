use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Rounding methods, matching FFmpeg's AVRounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Flag telling rescaling functions to pass `i64::MIN`/`i64::MAX` through unchanged.
pub const ROUND_PASS_MINMAX: u32 = 8192;

/// Rational number (pair of numerator and denominator).
///
/// Matches FFmpeg's AVRational. The denominator should not be zero
/// except for special values (infinity).
#[derive(Clone, Copy, Default)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

impl Rational {
    /// Create a new Rational.
    pub const fn new(num: i32, den: i32) -> Self {
        Self { num, den }
    }

    /// Convert to f64.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Invert the rational (swap num and den).
    pub const fn invert(self) -> Self {
        Self {
            num: self.den,
            den: self.num,
        }
    }

    /// Reduce the rational to lowest terms with components bounded by `max`.
    /// Returns `true` if the reduction is exact.
    pub fn reduce(&mut self, max: i64) -> bool {
        let (num, den, exact) = reduce(self.num as i64, self.den as i64, max);
        self.num = num;
        self.den = den;
        exact
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

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        cmp_q(*self, *other)
    }
}

/// Compare two rationals, matching FFmpeg's `av_cmp_q` behavior.
/// Returns `None` for indeterminate forms (0/0).
fn cmp_q(a: Rational, b: Rational) -> Option<Ordering> {
    let tmp = a.num as i64 * b.den as i64 - b.num as i64 * a.den as i64;

    if tmp != 0 {
        // FFmpeg: return (int)((tmp ^ a.den ^ b.den)>>63)|1;
        // This yields 1 or -1 depending on the combined sign.
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
        match cmp.cmp(&0) {
            Ordering::Less => Some(Ordering::Less),
            Ordering::Greater => Some(Ordering::Greater),
            Ordering::Equal => Some(Ordering::Equal),
        }
    } else {
        None // indeterminate (0/0)
    }
}

/// Reduce a fraction. Returns (num, den, exact).
/// Matches FFmpeg's `av_reduce`.
#[allow(unknown_lints, clippy::manual_checked_ops)]
pub fn reduce(num: i64, den: i64, max: i64) -> (i32, i32, bool) {
    // Convergents: a0 and a1, each with .num and .den components.
    // These track the numerator and denominator of successive convergents
    // of the continued fraction expansion of num/den.
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

    // If already within bounds, set a1 = {num, den} and skip the loop
    // by zeroing den (FFmpeg: den = 0).
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
            // Last step refinement: find largest k such that
            // k*a1n + a0n <= max AND k*a1d + a0d <= max
            let mut x = x;
            if a1n != 0 {
                x = (max - a0n) / a1n;
            }
            if a1d != 0 {
                x = x.min((max - a0d) / a1d);
            }

            // Quality check: is the refined convergent closer to the true value
            // than the previous convergent a1? If so, accept it.
            // FFmpeg: if (den * (2*x*a1.den + a0.den) > num * a1.den)
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

    // FFmpeg: exactness is determined by whether den reached 0 naturally
    // (i.e., the continued fraction terminated without exceeding max).
    let dst_num = if sign { -(a1n as i32) } else { a1n as i32 };
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

/// Rescale a 64-bit integer with specified rounding.
/// Mathematically equivalent to `a * b / c`.
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

    // 128-bit arithmetic path
    let product = a as u128 * b as u128 + r as u128;
    let result = product / c as u128;
    if result > i64::MAX as u128 {
        return i64::MIN;
    }
    result as i64
}

/// Rescale a 64-bit integer with rounding to nearest.
/// Matches FFmpeg's `av_rescale`.
pub fn rescale(a: i64, b: i64, c: i64) -> i64 {
    rescale_rnd(a, b, c, Rounding::NearInf, false)
}

/// Rescale a 64-bit integer by 2 rational numbers with specified rounding.
/// Matches FFmpeg's `av_rescale_q_rnd`.
pub fn rescale_q_rnd(a: i64, bq: Rational, cq: Rational, rnd: Rounding, pass_minmax: bool) -> i64 {
    let b = bq.num as i64 * cq.den as i64;
    let c = cq.num as i64 * bq.den as i64;
    rescale_rnd(a, b, c, rnd, pass_minmax)
}

/// Rescale a 64-bit integer by 2 rational numbers.
/// Matches FFmpeg's `av_rescale_q`.
pub fn rescale_q(a: i64, bq: Rational, cq: Rational) -> i64 {
    rescale_q_rnd(a, bq, cq, Rounding::NearInf, false)
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self {
        // Use wrapping_neg to match FFmpeg's C behavior (wraps on i32::MIN).
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
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.invert()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_basics() {
        let r = Rational::new(1, 2);
        assert_eq!(r.to_f64(), 0.5);
        assert_eq!(r.invert(), Rational::new(2, 1));
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = Rational::new(1, 3);
        let b = Rational::new(1, 6);
        let sum = a + b;
        assert_eq!(sum, Rational::new(1, 2));

        let product = Rational::new(2, 3) * Rational::new(3, 4);
        assert_eq!(product, Rational::new(1, 2));
    }

    #[test]
    fn test_rescale_rnd() {
        // Simple cases
        assert_eq!(rescale(3, 1, 2), 2); // 3*1/2 = 1.5, round to nearest = 2
        assert_eq!(rescale_rnd(3, 1, 2, Rounding::Zero, false), 1);
        assert_eq!(rescale_rnd(3, 1, 2, Rounding::Up, false), 2);
        assert_eq!(rescale_rnd(3, 1, 2, Rounding::Down, false), 1);
    }

    #[test]
    fn test_rescale_q() {
        let tb1 = Rational::new(1, 48000);
        let tb2 = Rational::new(1, 1000);
        // 48000 samples at 48kHz = 1000ms
        let result = rescale_q(48000, tb1, tb2);
        assert_eq!(result, 1000);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(7, 13), 1);
    }

    #[test]
    fn test_reduce() {
        let (n, d, exact) = reduce(6, 4, i32::MAX as i64);
        assert!(exact);
        assert_eq!(n, 3);
        assert_eq!(d, 2);
    }

    #[test]
    fn test_pass_minmax() {
        assert_eq!(
            rescale_rnd(i64::MIN, 1, 2, Rounding::NearInf, true),
            i64::MIN
        );
        assert_eq!(
            rescale_rnd(i64::MAX, 1, 2, Rounding::NearInf, true),
            i64::MAX
        );
    }
}
