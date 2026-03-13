// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Marchenko-Pastur analytical checks: verify λ± bounds against known formulas.

use approx::assert_relative_eq;
use nalgebra::DVector;
use regit_covariance::math::marchenko_pastur::fit_sigma_sq;

/// For a pure-noise correlation matrix (σ² ≈ 1), the MP bounds should
/// match the theoretical formula: λ± = σ²(1 ± √(1/q))².
#[test]
fn mp_bounds_match_formula() {
    let n = 50;
    let q = 5.0;
    let eigenvalues = DVector::from_element(n, 1.0);

    let fit = fit_sigma_sq(&eigenvalues, q).unwrap();

    assert_relative_eq!(fit.sigma_sq, 1.0, epsilon = 0.1);

    let expected_plus = fit.sigma_sq * (1.0 + (1.0 / q).sqrt()).powi(2);
    let expected_minus = fit.sigma_sq * (1.0 - (1.0 / q).sqrt()).powi(2);

    assert_relative_eq!(fit.lambda_plus, expected_plus, epsilon = 1e-10);
    assert_relative_eq!(fit.lambda_minus, expected_minus, epsilon = 1e-10);
}

/// When all eigenvalues are in the MP bulk (pure noise), signal_count should be 0.
#[test]
fn pure_noise_zero_signals() {
    let n = 20;
    let eigenvalues = DVector::from_element(n, 1.0);
    let fit = fit_sigma_sq(&eigenvalues, 10.0).unwrap();
    assert_eq!(fit.signal_count, 0);
    assert_eq!(fit.noise_count, n);
}

/// When one eigenvalue is very large, it should be classified as signal.
#[test]
fn single_large_eigenvalue_is_signal() {
    let n = 20;
    let mut vals = vec![1.0; n];
    vals[0] = 10.0;
    let eigenvalues = DVector::from_vec(vals);

    let fit = fit_sigma_sq(&eigenvalues, 10.0).unwrap();
    assert!(fit.signal_count >= 1, "large eigenvalue should be signal");
}

/// λ_plus > λ_minus always.
#[test]
fn lambda_plus_exceeds_minus() {
    for q in [2.0, 5.0, 10.0, 50.0] {
        let eigenvalues = DVector::from_element(10, 1.0);
        let fit = fit_sigma_sq(&eigenvalues, q).unwrap();
        assert!(
            fit.lambda_plus > fit.lambda_minus,
            "q={q}: λ+={} should exceed λ-={}",
            fit.lambda_plus,
            fit.lambda_minus
        );
    }
}

/// signal_count + noise_count = N.
#[test]
fn signal_noise_partition_sums_to_n() {
    let n = 30;
    let mut vals: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let eigenvalues = DVector::from_vec(vals);

    let fit = fit_sigma_sq(&eigenvalues, 5.0).unwrap();
    assert_eq!(
        fit.signal_count + fit.noise_count,
        n,
        "partition must sum to N"
    );
}
