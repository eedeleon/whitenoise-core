use whitenoise_validator::errors::*;
// use probability::distribution::{Gaussian, Laplace, Inverse, Distribution};
// use ieee754::Ieee754;
use std::{cmp, f64::consts, f64::MAX, char};
use rug::rand::{ThreadRandGen, ThreadRandState};
use rug::{Float, ops::Pow};
use gmp_mpfr_sys::mpfr;
use math::round;

use crate::utilities::utilities;

/// Calculate base from `eta` values.
///
/// # Arguments
/// * `eta_x` - Privacy parameter.
/// * `eta_y` - Privacy parameter.
/// * `eta_z` - Privacy parameter.
/// * `precision` - Bits of precision with which you want the output generated.
///
/// # Return
/// Functional base for base2 exponential mechanism.
pub fn get_base(eta_x: i64, eta_y: i64, eta_z: i64, precision: u32) -> Float {
    let base_p1 = Float::with_val(precision, eta_x.pow(eta_z as u32));
    let base_p2 = Float::with_val(precision, Float::with_val(precision, -(eta_y * eta_z)).exp2());
    let base = Float::with_val(precision, base_p1 * base_p2);
    return base;
}

/// Check for sufficient precision to exactly carry out necessary operations.
///
/// Differs slightly from Ilv19 in that utility can be negative.
///
/// Returns 0 if yes, non-zero otherwise (depending on the flag that is raised).
///
/// # Arguments
/// * `eta_x` - Privacy parameter.
/// * `eta_y` - Privacy parameter.
/// * `eta_z` - Privacy parameter.
/// * `precision` - Bits of precision with which you want the output generated.
/// * `min_utility` - Minimum possible utility value.
/// * `max_utility` - Maximum possible utility value.
/// * `max_size_outcome_space` - Maximum size of the outcome space.
///
/// # Return
/// 0 if precision is sufficient to exactly carry out necessary operations, non-zero `u32` otherwise.
pub unsafe fn check_precision(eta_x: &i64, eta_y: &i64, eta_z: &i64, precision: &u32,
                              min_utility: &f64, max_utility: &f64, max_size_outcome_space: &u32) -> u32 {
    // reset flags
    mpfr::clear_flags();

    // compute base
    let base = &get_base(*eta_x, *eta_y, *eta_z, *precision);

    // compute base^({min/max return})
    let min_weight = Float::with_val(*precision, base.pow(*min_utility));
    let max_weight = Float::with_val(*precision, base.pow(*max_utility));
    let mm = Float::with_val(*precision, &min_weight + &max_weight);

    // compute max/min total utility
    let min_total = Float::with_val(*precision, &min_weight * &Float::with_val(*precision, *max_size_outcome_space));
    let max_total = Float::with_val(*precision, &max_weight * &Float::with_val(*precision, *max_size_outcome_space));

    // add min/max total utilities
    let max_min_total = Float::with_val(*precision, max_total + min_weight);
    let min_max_total = Float::with_val(*precision, min_total + max_weight);

    // get raised flags and return whether or not computations were exact
    let flags = mpfr::flags_save();
    return mpfr::flags_test(flags);
}

/// Calculate upper bound on sufficient precision for base2 exponential mechanism.
///
/// Differs slightly from Ilv19 in that utility can be negative.
///
/// # Arguments
/// * `eta_x` - Privacy parameter.
/// * `eta_y` - Privacy parameter.
/// * `eta_z` - Privacy parameter.
/// * `precision` - Bits of precision with which you want the output generated.
/// * `min_utility` - Minimum possible utility value.
/// * `max_utility` - Maximum possible utility value.
/// * `max_size_outcome_space` - Maximum size of the outcome space.
///
/// # Return
/// Upper bound on sufficient precision.
pub unsafe fn get_sufficient_precision(eta_x: &i64, eta_y: &i64, eta_z: &i64,
                                       min_utility: &f64, max_utility: &f64, max_size_outcome_space: &u32) -> u32 {
    let mut precision = 16_u32;
    let sufficient_precision = false;
    let mut flag;
    while sufficient_precision == false {
        // ensure that desired precision is supported
        assert!(precision <= rug::float::prec_max());

        // check if precision is sufficient for exact operations
        flag = check_precision(eta_x, eta_y, eta_z, &precision, min_utility, max_utility, max_size_outcome_space);
        if flag == 0 {
            break;
        } else {
            precision = 2 * precision;
        }
    }
    return precision;
}

/// Implementation of the `get_random_value` function from Ilv19.
///
/// Generate uniform numbers from the interval `[0, 2^(pow_2))`.
/// `pow_2` and `precision` are equivalent to `start_pow`+1 and `p` from the original code.
///
/// # Arguments
/// * `pow_2` - `k` such that `2^k` will be an upper bound on the generated values.
/// * `precision` - Bits of precision with which you want the output generated.
///
/// # Example
/// ```
/// use whitenoise_runtime::utilities::base2_exponential::sample_uniform_bounded_pow_2;
/// let unif = sample_uniform_bounded_pow_2(3, 53).to_f64();
/// assert!(unif >= 0. && unif < 8.);
/// ```
pub fn sample_uniform_bounded_pow_2(pow_2: i64, precision: u32) -> Float {
    // get random bits from OpenSSL and convert them into a vector of ints
    let n_bytes = round::ceil((precision as f64 / 8.) as f64, 0) as usize;
    let bits = utilities::get_bytes(n_bytes);
    let bit_vec: Vec<u32> = bits.chars().map(|x| x.to_digit(10).unwrap()).collect();

    // loop over first `precision` bits and store the number it represents (bit * 2^k) for some k
    let result_vec: Vec<Float> = (0..precision).map(|i|
                                                       Float::with_val(precision,
                                                            Float::with_val(precision, bit_vec[i as usize]) *
                                                            Float::with_val(precision, 2_f64.powi((pow_2 - 1 - (i as i64)) as i32))
                                                       ) ).collect();

    // sum over vector to get final uniform output
    let unif = Float::with_val(precision, Float::sum(result_vec.iter()));

    return unif;
}

/// Slightly altered implementation of the `randomized_round` function from Ilv19.
///
/// Rounds the input x to an adjacent integer value.
/// x is rounded up with probability `x - floor(x)` and
/// rounding randomness is sampled at the level of `precision`
///
/// This implementation differs from the original in that `u_min` and `u_max` are
/// replaced by explicit arguments `min_return` and `max_return`. Because `min_return`
/// and `max_return` might be `f64` for some other use case, we allow the to be so here,
/// even `u_min` and `u_max` are `i64` in the original code.
///
/// Additionally, the original code assumes that lower utilities are better than higher ones, so
/// `u_min` > `u_max`. We take the approach of higher utilities being better.
///
/// # Arguments
/// * `x` - Element to be rounded.
/// * `precision` - Bits of precision with which you want the output generated.
/// * `min_return` - Minimum allowable return value.
/// * `max_return` - Maximum allowable return value.
///
/// # Example
/// use whitenoise_runtime::utilities::base2_exponential::randomized_round;
/// let rounded = randomized_round(3.5, 53, 2., 5.);
/// assert!(rounded == 3. || rounded == 4.)
pub fn randomized_round(x: f64, precision: u32, min_return: f64, max_return: f64) -> f64 {
    let unif = sample_uniform_bounded_pow_2(0, precision);

    let lower = round::floor(x, 0) as f64;
    let upper = round::ceil(x, 0) as f64;

    if unif > x - (lower as f64) {
        return lower.max(min_return).min(max_return);
    } else {
        return upper.min(max_return).max(min_return);
    }
}
/// Implementation of `normalized sample` from the original code.
///
/// Returns an index based on weights.
///
/// # Arguments
/// * `weights` - Weights for each index.
/// * `precision` - Bits of precision with which you want the output generated.
///
/// # Return
/// Index based on sampling weights.

pub fn normalized_sample(weights: Vec<Float>, precision: u32) -> usize {
    // get total weight
    let total_weight = Float::with_val(53, Float::sum(weights.iter()));

    // generate cumulative weights
    let mut cumulative_weight_vec: Vec<rug::Float> = Vec::with_capacity(weights.len() as usize);
    for i in 0..weights.len() {
        cumulative_weight_vec.push( Float::with_val(53, Float::sum(weights[0..(i+1)].iter())) );
    }

    // get maximum power of two needed for sampling
    let mut pow_2: i64 = 0;
    while (Float::with_val(53, pow_2)).exp2() > total_weight.to_f64() {
        pow_2 = pow_2 - 1;
    }
    while (Float::with_val(53, pow_2)).exp2() <= total_weight.to_f64() {
        pow_2 = pow_2 + 1;
    }

    // sample a random number from [0, 2^pow_2)
    let mut s = Float::with_val(53, std::f64::MAX);
    while s > total_weight {
        s = sample_uniform_bounded_pow_2(pow_2, precision);
    }

    // return the index of an element based on where it falls in the cumulative distribution of weights
    let mut index = 0;
    for i in 0..weights.len() {
        if cumulative_weight_vec[i].to_f64() >= s {
            index = i;
            break;
        }
    }
    return index;
}

