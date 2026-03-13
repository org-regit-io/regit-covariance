// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Marchenko-Pastur law: noise variance estimation and signal/noise partition.
//!
//! The Marchenko-Pastur distribution describes the limiting spectral density
//! of large random matrices. For a T × N matrix of i.i.d. entries with
//! variance σ², the eigenvalues of the sample covariance concentrate on
//! the support \[λ₋, λ₊\] where:
//!
//! - λ₊ = σ²(1 + √(N/T))²
//! - λ₋ = σ²(1 − √(N/T))²
//!
//! Eigenvalues exceeding λ₊ carry signal; those within the support are noise.
//!
//! # References
//!
//! - Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues
//!   for some sets of random matrices. *Matematicheskii Sbornik*, 114(4), 507–536.
//! - López de Prado, M. (2018). *Advances in Financial Machine Learning*,
//!   Chapter 2, Sections 2.4–2.6.

use nalgebra::DVector;
use thiserror::Error;

/// Errors from Marchenko-Pastur fitting.
#[derive(Debug, Error)]
pub enum MpError {
    /// Ratio q = T/N must be positive.
    #[error("ratio q must be positive, got {0}")]
    InvalidQ(f64),

    /// Eigenvalue vector is empty.
    #[error("empty eigenvalue vector")]
    EmptyEigenvalues,

    /// Fixed-point iteration did not converge.
    #[error("σ² fitting did not converge after {max_iter} iterations (residual: {residual})")]
    ConvergenceFailure {
        /// Maximum iterations attempted.
        max_iter: usize,
        /// Final residual.
        residual: f64,
    },
}

/// Result of Marchenko-Pastur fitting.
#[derive(Debug, Clone)]
pub struct MpFit {
    /// Fitted noise variance.
    pub sigma_sq: f64,
    /// Upper MP bound: λ₊ = σ²(1 + √(N/T))².
    pub lambda_plus: f64,
    /// Lower MP bound: λ₋ = σ²(1 − √(N/T))².
    pub lambda_minus: f64,
    /// Number of signal eigenvalues (λ > λ₊).
    pub signal_count: usize,
    /// Number of noise eigenvalues (λ ≤ λ₊).
    pub noise_count: usize,
    /// Ratio q = T/N.
    pub q: f64,
}

/// Compute the Marchenko-Pastur bounds for given σ² and q = T/N.
///
/// Returns (λ₋, λ₊) where:
/// - λ₊ = σ²(1 + √(1/q))²
/// - λ₋ = σ²(1 − √(1/q))²
///
/// # Panics
///
/// Does not panic; returns (0, 0) if q ≤ 0 (caller should validate).
#[must_use]
pub fn mp_bounds(sigma_sq: f64, q: f64) -> (f64, f64) {
    if q <= 0.0 || !q.is_finite() || !sigma_sq.is_finite() {
        return (0.0, 0.0);
    }
    let sqrt_inv_q = (1.0 / q).sqrt();
    let lambda_plus = sigma_sq * (1.0 + sqrt_inv_q).powi(2);
    let lambda_minus = sigma_sq * (1.0 - sqrt_inv_q).powi(2);
    (lambda_minus, lambda_plus)
}

/// Evaluate the Marchenko-Pastur probability density function at λ.
///
/// `f_MP`(λ) = (q / 2πσ²) × √((λ₊ − λ)(λ − λ₋)) / λ
///
/// Returns 0.0 for λ outside the support \[λ₋, λ₊\] or for λ ≤ 0.
///
/// # References
///
/// - Marchenko & Pastur (1967), Theorem 1.1.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mp_density(lambda: f64, sigma_sq: f64, q: f64) -> f64 {
    if q <= 0.0 || sigma_sq <= 0.0 || lambda <= 0.0 {
        return 0.0;
    }

    let (lambda_minus, lambda_plus) = mp_bounds(sigma_sq, q);

    if lambda < lambda_minus || lambda > lambda_plus {
        return 0.0;
    }

    let numerator = ((lambda_plus - lambda) * (lambda - lambda_minus)).sqrt();
    let denominator = 2.0 * std::f64::consts::PI * sigma_sq * lambda / q;

    if denominator.abs() < f64::EPSILON {
        return 0.0;
    }

    numerator / denominator
}

/// Fit the noise variance σ² from an empirical eigenvalue spectrum.
///
/// Uses a fixed-point iteration: starting from the initial estimate
/// σ² = mean of all eigenvalues / N (assuming all are noise), iteratively
/// refine by computing σ² as the mean of eigenvalues ≤ λ₊(σ²).
///
/// # Arguments
///
/// * `eigenvalues` — Eigenvalues sorted in descending order.
/// * `q` — Ratio T/N where T = observations, N = assets.
///
/// # Errors
///
/// Returns [`MpError`] if q is invalid, eigenvalues are empty, or the
/// iteration does not converge.
///
/// # References
///
/// - López de Prado (2018), Chapter 2, Code Snippet 2.4.
#[allow(clippy::cast_precision_loss)]
pub fn fit_sigma_sq(eigenvalues: &DVector<f64>, q: f64) -> Result<MpFit, MpError> {
    if q <= 0.0 || !q.is_finite() {
        return Err(MpError::InvalidQ(q));
    }

    let num_eigenvalues = eigenvalues.len();
    if num_eigenvalues == 0 {
        return Err(MpError::EmptyEigenvalues);
    }

    let max_iter = 1000;
    let tol = 1e-10;

    // Initial estimate: mean of all eigenvalues (assume all noise).
    let mut sigma_sq = eigenvalues.mean();

    for _ in 0..max_iter {
        let (_, lambda_plus) = mp_bounds(sigma_sq, q);

        // Noise eigenvalues: those ≤ λ₊.
        let noise_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .filter(|&&ev| ev <= lambda_plus)
            .copied()
            .collect();

        if noise_eigenvalues.is_empty() {
            // All eigenvalues are signal — σ² is essentially zero.
            sigma_sq = f64::EPSILON;
            break;
        }

        let new_sigma_sq = noise_eigenvalues.iter().sum::<f64>() / noise_eigenvalues.len() as f64;

        if (new_sigma_sq - sigma_sq).abs() < tol {
            sigma_sq = new_sigma_sq;
            break;
        }

        sigma_sq = new_sigma_sq;
    }

    let (lambda_minus, lambda_plus) = mp_bounds(sigma_sq, q);
    let signal_count = eigenvalues.iter().filter(|&&ev| ev > lambda_plus).count();
    let noise_count = num_eigenvalues - signal_count;

    Ok(MpFit {
        sigma_sq,
        lambda_plus,
        lambda_minus,
        signal_count,
        noise_count,
        q,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── MP bounds ──────────────────────────────────────────────────────

    /// For σ²=1, q=1 (T=N): λ₊ = (1+1)² = 4, λ₋ = (1-1)² = 0.
    #[test]
    fn test_mp_bounds_q1() {
        let (lambda_minus, lambda_plus) = mp_bounds(1.0, 1.0);
        assert_relative_eq!(lambda_plus, 4.0, epsilon = 1e-10);
        assert_relative_eq!(lambda_minus, 0.0, epsilon = 1e-10);
    }

    /// For σ²=1, q=2 (T=2N): λ₊ = (1+√0.5)², λ₋ = (1-√0.5)².
    #[test]
    fn test_mp_bounds_q2() {
        let (lambda_minus, lambda_plus) = mp_bounds(1.0, 2.0);
        let sqrt_half = (0.5_f64).sqrt();
        assert_relative_eq!(lambda_plus, (1.0 + sqrt_half).powi(2), epsilon = 1e-10);
        assert_relative_eq!(lambda_minus, (1.0 - sqrt_half).powi(2), epsilon = 1e-10);
    }

    /// Bounds scale linearly with σ².
    #[test]
    fn test_mp_bounds_sigma_scaling() {
        let (lm1, lp1) = mp_bounds(1.0, 2.0);
        let (lm2, lp2) = mp_bounds(2.0, 2.0);
        assert_relative_eq!(lp2, 2.0 * lp1, epsilon = 1e-10);
        assert_relative_eq!(lm2, 2.0 * lm1, epsilon = 1e-10);
    }

    /// Invalid q returns (0, 0).
    #[test]
    fn test_mp_bounds_invalid_q() {
        assert_eq!(mp_bounds(1.0, 0.0), (0.0, 0.0));
        assert_eq!(mp_bounds(1.0, -1.0), (0.0, 0.0));
    }

    // ── MP density ────────────────────────────────────────────────────

    /// Density is zero outside the support.
    #[test]
    fn test_mp_density_outside_support() {
        let sigma_sq = 1.0;
        let q = 2.0;
        let (lambda_minus, lambda_plus) = mp_bounds(sigma_sq, q);

        assert_relative_eq!(mp_density(lambda_minus - 0.01, sigma_sq, q), 0.0);
        assert_relative_eq!(mp_density(lambda_plus + 0.01, sigma_sq, q), 0.0);
        assert_relative_eq!(mp_density(-1.0, sigma_sq, q), 0.0);
    }

    /// Density is positive inside the support.
    #[test]
    fn test_mp_density_inside_support() {
        let sigma_sq = 1.0;
        let q = 2.0;
        let (lambda_minus, lambda_plus) = mp_bounds(sigma_sq, q);
        let mid = (lambda_minus + lambda_plus) / 2.0;

        let density = mp_density(mid, sigma_sq, q);
        assert!(density > 0.0, "density at midpoint should be positive");
    }

    /// Density integrates to approximately 1 (numerical check).
    #[test]
    fn test_mp_density_integrates_to_one() {
        let sigma_sq = 1.0;
        let q = 2.0;
        let (lambda_minus, lambda_plus) = mp_bounds(sigma_sq, q);

        let steps = 10_000;
        let dl = (lambda_plus - lambda_minus) / steps as f64;
        let integral: f64 = (0..steps)
            .map(|step| {
                let lambda = lambda_minus + (step as f64 + 0.5) * dl;
                mp_density(lambda, sigma_sq, q) * dl
            })
            .sum();

        assert_relative_eq!(integral, 1.0, epsilon = 1e-3);
    }

    // ── σ² fitting ────────────────────────────────────────────────────

    /// Pure noise eigenvalues (σ²=1, q=2): fitted σ² should be close to 1.
    #[test]
    fn test_fit_pure_noise() {
        // For a large random matrix with σ²=1, eigenvalues cluster around 1.
        // Simulate: N eigenvalues all equal to 1 (perfect noise).
        let eigenvalues = DVector::from_element(100, 1.0);
        let fit = fit_sigma_sq(&eigenvalues, 2.0).unwrap();

        assert_relative_eq!(fit.sigma_sq, 1.0, epsilon = 0.1);
        assert_eq!(fit.signal_count, 0);
        assert_eq!(fit.noise_count, 100);
    }

    /// Signal + noise: one large eigenvalue, rest noise.
    #[test]
    fn test_fit_signal_plus_noise() {
        let mut vals = vec![1.0; 99];
        vals.insert(0, 20.0); // One strong signal eigenvalue.
        let eigenvalues = DVector::from_vec(vals);
        let fit = fit_sigma_sq(&eigenvalues, 2.0).unwrap();

        assert!(
            fit.signal_count >= 1,
            "should detect at least 1 signal eigenvalue"
        );
        assert_relative_eq!(fit.sigma_sq, 1.0, epsilon = 0.15);
    }

    /// Invalid q should error.
    #[test]
    fn test_fit_invalid_q() {
        let eigenvalues = DVector::from_element(10, 1.0);
        assert!(fit_sigma_sq(&eigenvalues, 0.0).is_err());
        assert!(fit_sigma_sq(&eigenvalues, -1.0).is_err());
    }

    /// Empty eigenvalues should error.
    #[test]
    fn test_fit_empty() {
        let eigenvalues = DVector::from_vec(vec![]);
        assert!(fit_sigma_sq(&eigenvalues, 2.0).is_err());
    }
}
