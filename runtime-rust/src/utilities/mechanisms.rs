use whitenoise_validator::errors::*;


use ndarray::prelude::*;

use rug::{float::Constant, Float, ops::Pow};

use crate::utilities::noise;
use crate::utilities::utilities;
use crate::utilities::base2_exponential;

/// Returns noise drawn according to the Laplace mechanism
///
/// Noise is drawn with scale sensitivity/epsilon and centered about 0.
/// For more information, see the Laplace mechanism in
/// C. Dwork, A. Roth The Algorithmic Foundations of Differential Privacy, Chapter 3.3 The Laplace Mechanism p.30-37. August 2014.
///
/// NOTE: this implementation of Laplace draws is likely non-private due to floating-point attacks
/// See [Mironov (2012)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.5957&rep=rep1&type=pdf)
/// for more information
///
/// # Arguments
///
/// * `epsilon` - Multiplicative privacy loss parameter.
/// * `sensitivity` - Upper bound on the L1 sensitivity of the function you want to privatize.
///
/// # Return
/// Array of a single value drawn from the Laplace distribution with scale sensitivity/epsilon centered about 0.
///
/// # Examples
/// ```
/// use whitenoise_runtime::utilities::mechanisms::laplace_mechanism;
/// let n = laplace_mechanism(&0.1, &2.0);
/// ```
pub fn laplace_mechanism(epsilon: &f64, sensitivity: &f64) -> f64 {
    let scale: f64 = sensitivity / epsilon;
    let noise: f64 = noise::sample_laplace(0., scale);

    noise
}

/// Returns noise drawn according to the Gaussian mechanism.
///
/// Let c = sqrt(2*ln(1.25/delta)). Noise is drawn from a Gaussian distribution with scale
/// sensitivity*c/epsilon and centered about 0.
///
/// For more information, see the Gaussian mechanism in
/// C. Dwork, A. Roth The Algorithmic Foundations of Differential Privacy, Chapter 3.5.3 Laplace versus Gauss p.53. August 2014.
///
/// NOTE: this implementation of Gaussian draws in likely non-private due to floating-point attacks
/// See [Mironov (2012)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.5957&rep=rep1&type=pdf)
/// for more information on a similar attack of the Laplace mechanism.
///
/// # Arguments
///
/// * `epsilon` - Multiplicative privacy loss parameter.
/// * `delta` - Additive privacy loss parameter.
/// * `sensitivity` - Upper bound on the L2 sensitivity of the function you want to privatize.
///
/// # Return
/// A draw from Gaussian distribution with scale defined as above.
///
/// # Examples
/// ```
/// use whitenoise_runtime::utilities::mechanisms::gaussian_mechanism;
/// let n = gaussian_mechanism(&0.1, &0.0001, &2.0);
/// ```
pub fn gaussian_mechanism(epsilon: &f64, delta: &f64, sensitivity: &f64) -> f64 {
    let scale: f64 = sensitivity * (2. * (1.25 / delta).ln()).sqrt() / epsilon;
    let noise: f64 = noise::sample_gaussian(&0., &scale);
    noise
}

/// Returns noise drawn according to the Geometric mechanism.
///
/// Uses the Geometric mechanism as originally proposed in
/// [Ghosh, Roughgarden, & Sundarajan (2012)](https://theory.stanford.edu/~tim/papers/priv.pdf).
/// We are calling this the `simple_geometric_mechanism` because there is some hope that we will later
/// add other versions, such as those developed in [Balcer & Vadhan (2019)](https://arxiv.org/pdf/1709.05396.pdf)
///
/// # Arguments
///
/// * `epsilon` - Multiplicative privacy loss parameter
/// * `sensitivity` - L1 sensitivity of function you want to privatize. The Geometric is typically used for counting queries, where sensitivity = 1.
/// * `count_min` - The minimum count you think possible, typically 0.
/// * `count_max` - The maximum count you think possible, typically the size of your data.
/// * `enforce_constant_time` - Whether or not to run the noise generation algorithm in constant time.
///                             If true, will run count_max-count_min number of times.
/// # Return
/// A draw according to the Geometric mechanism.
///
/// # Examples
/// ```
/// use whitenoise_runtime::utilities::mechanisms::simple_geometric_mechanism;
/// let n = simple_geometric_mechanism(&0.1, &1., &0, &10, &true);
/// ```
pub fn simple_geometric_mechanism(epsilon: &f64, sensitivity: &f64, count_min: &i64, count_max: &i64, enforce_constant_time: &bool) -> i64 {
    let scale: f64 = sensitivity / epsilon;
    let noise: i64 = noise::sample_simple_geometric_mechanism(&scale, &count_min, &count_max, &enforce_constant_time);
    noise
}

/// Returns data element according to the Exponential mechanism.
///
/// NOTE: This implementation is likely non-private because of the difference between theory on
///       the real numbers and floating-point numbers. See [Ilvento 2019](https://arxiv.org/abs/1912.04222) for
///       more information on the problem and a proposed fix.
///
/// # Arguments
///
/// * `epsilon` - Multiplicative privacy loss parameter.
/// * `sensitivity` - L1 sensitivity of utility function.
/// * `candidate_set` - Data from which user wants an element returned.
/// * `utility` - Utility function used within the exponential mechanism.
///
/// # Return
/// An element from `candidate_set`, chosen with probability proportional to its utility.
///
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use whitenoise_runtime::utilities::mechanisms::exponential_mechanism;
/// // create utility function
/// pub fn utility(x:&f64) -> f64 {
///     let util = *x as f64;
///     return util;
/// }
///
/// // create sample data
/// let xs: ArrayD<f64> = arr1(&[1., 2., 3., 4., 5.]).into_dyn();
/// let ans = exponential_mechanism(&1.0, &1.0, xs, &utility).unwrap();
/// assert!(ans == 1. || ans == 2. || ans == 3. || ans == 4. || ans == 5.);
/// ```
pub fn exponential_mechanism<T>(
                         epsilon: &f64,
                         sensitivity: &f64,
                         candidate_set: ArrayD<T>,
                         utility: &dyn Fn(&T) -> f64
                         ) -> Result<T> where T: Copy, {

    // get vector of e^(util), then use to find probabilities
    let rug_e = Float::with_val(53, Constant::Euler);
    let rug_eps = Float::with_val(53, epsilon);
    let rug_sens = Float::with_val(53, sensitivity);
    let e_util_vec: Vec<rug::Float> = candidate_set.iter()
        .map(|x| rug_e.clone().pow(rug_eps.clone() * Float::with_val(53, utility(x)) / (2.0 * rug_sens.clone()))).collect();
    let sum_e_util_vec: rug::Float = Float::with_val(53, Float::sum(e_util_vec.iter()));
    let probability_vec: Vec<f64> = e_util_vec.iter().map(|x| (x / sum_e_util_vec.clone()).to_f64()).collect();

    // sample element relative to probability
    let candidate_vec: Vec<T> = candidate_set.clone().into_dimensionality::<Ix1>().unwrap().to_vec();
    let elem: T = utilities::sample_from_set(&candidate_vec, &probability_vec)?;

    Ok(elem)
}

/// Returns data element according to base2 Exponential mechanism.
///
/// # Arguments
/// * `eta_x` - Privacy parameter.
/// * `eta_y` - Privacy parameter.
/// * `eta_z` - Privacy parameter.
/// * `min_utility` - Minimum possible utility value.
/// * `max_utility` - Maximum possible utility value.
/// * `candidate_set` - Data from which user wants an element returned.
/// * `utility` - Utility function used within the exponential mechanism.
///
/// # Return
/// An element from `candidate_set`, chosen with probability proportional to its utility.
///
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use whitenoise_runtime::utilities::mechanisms::base2_exponential_mechanism;
/// // create utility function
/// pub fn utility(x:&f64) -> f64 {
///     let util = *x as f64;
///     return util;
/// }
///
/// // create sample data
/// let xs: ArrayD<f64> = arr1(&[1., 2., 3., 4., 5.]).into_dyn();
/// let ans = base2_exponential_mechanism(&1, &1, &1, &1.0, &5.0, xs, &utility).unwrap();
/// assert!(ans == 1. || ans == 2. || ans == 3. || ans == 4. || ans == 5.);
/// ```
pub fn base2_exponential_mechanism<T>(
                        eta_x: &i64,
                        eta_y: &i64,
                        eta_z: &i64,
                        min_utility: &f64,
                        max_utility: &f64,
                        candidate_set: ArrayD<T>,
                        utility: &dyn Fn(&T) -> f64
                         ) -> Result<T> where T: Copy, {
    unsafe {
        // get max size of the outcome space
        let max_size_outcome_space = candidate_set.len() as u32;

        // calculate necessary precision
        let precision = base2_exponential::get_sufficient_precision(eta_x, eta_y, eta_z, min_utility, max_utility, &max_size_outcome_space);

        // calculate base
        let base: Float = base2_exponential::get_base(*eta_x, *eta_y, *eta_z, precision.clone());

        // get utilities
        let utilities: Vec<Float> = candidate_set.iter().map(|x| Float::with_val(precision.clone(), utility(x))).collect();

        // get weights
        let weights: Vec<Float> = utilities.iter().map(|u| base.clone().pow(u)).collect();

        // sample from set based on weights
        let sampling_index = base2_exponential::normalized_sample(weights, precision.clone());
        Ok(candidate_set[sampling_index])
    }
}