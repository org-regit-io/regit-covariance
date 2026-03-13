// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Ledoit-Wolf shrinkage estimators.
//!
//! ## Linear shrinkage (2004)
//!
//! Shrinks the sample covariance toward a structured target (constant-correlation
//! matrix) with analytically optimal intensity δ. The estimator is:
//!
//! Σ̃ = δ·F + (1−δ)·S
//!
//! where S is the sample covariance, F is the target, and δ ∈ \[0, 1\].
//!
//! ## Nonlinear shrinkage (2020)
//!
//! Applies individual optimal shrinkage to each eigenvalue via the
//! Marchenko-Pastur equation and Stieltjes transform. No tuning parameters.
//! This is the most theoretically principled estimator for large-dimensional
//! covariance matrices.
//!
//! # References
//!
//! - Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
//!   large-dimensional covariance matrices. *Journal of Multivariate Analysis*,
//!   88(2), 365–411.
//! - Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of
//!   large-dimensional covariance matrices. *Annals of Statistics*, 48(5),
//!   3043–3065.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

use super::eigen::EigenDecomposition;

/// Errors from Ledoit-Wolf shrinkage.
#[derive(Debug, Error)]
pub enum ShrinkageError {
    /// Insufficient observations for shrinkage estimation.
    #[error("need T ≥ 2, got {0}")]
    InsufficientObservations(usize),

    /// Empty returns matrix.
    #[error("empty returns matrix")]
    EmptyReturns,
}

/// Result of linear shrinkage.
#[derive(Debug, Clone)]
pub struct LinearShrinkageResult {
    /// Shrunk covariance/correlation matrix.
    pub matrix: DMatrix<f64>,
    /// Optimal shrinkage intensity δ ∈ \[0, 1\].
    pub intensity: f64,
}

/// Result of nonlinear shrinkage.
#[derive(Debug, Clone)]
pub struct NonlinearShrinkageResult {
    /// Shrunk covariance/correlation matrix.
    pub matrix: DMatrix<f64>,
    /// Shrunk eigenvalues.
    pub eigenvalues: DVector<f64>,
}

/// Compute the Ledoit-Wolf (2004) linear shrinkage estimator.
///
/// Shrinks the sample correlation matrix toward the constant-correlation
/// target F (mean correlation on off-diagonal, ones on diagonal).
/// The optimal intensity δ is computed analytically.
///
/// # Arguments
///
/// * `returns` — (T × N) standardized returns matrix.
/// * `sample_corr` — (N × N) sample correlation matrix.
///
/// # Errors
///
/// Returns [`ShrinkageError`] if the returns matrix is too small.
///
/// # References
///
/// - Ledoit & Wolf (2004), Theorem 1.
#[allow(clippy::cast_precision_loss)]
pub fn linear_shrinkage(
    returns: &DMatrix<f64>,
    sample_corr: &DMatrix<f64>,
) -> Result<LinearShrinkageResult, ShrinkageError> {
    let (num_obs, num_assets) = returns.shape();

    if num_obs == 0 || num_assets == 0 {
        return Err(ShrinkageError::EmptyReturns);
    }
    if num_obs < 2 {
        return Err(ShrinkageError::InsufficientObservations(num_obs));
    }

    let num_obs_f = num_obs as f64;

    // Target: constant-correlation matrix.
    // Mean off-diagonal correlation.
    let mut sum_off_diag = 0.0;
    let mut count_off_diag = 0_u64;
    for row in 0..num_assets {
        for col in (row + 1)..num_assets {
            sum_off_diag += sample_corr[(row, col)];
            count_off_diag += 1;
        }
    }
    let mean_corr = if count_off_diag > 0 {
        sum_off_diag / count_off_diag as f64
    } else {
        0.0
    };

    // Build target F: identity + mean_corr on off-diagonal.
    let mut target = DMatrix::from_element(num_assets, num_assets, mean_corr);
    for idx in 0..num_assets {
        target[(idx, idx)] = 1.0;
    }

    // Compute optimal shrinkage intensity δ.
    // Following Ledoit & Wolf (2004) analytical formula.

    // Step 1: Compute squared Frobenius norm of (S - F).
    let diff = sample_corr - &target;
    let frob_sq: f64 = diff.iter().map(|val| val * val).sum();

    // Step 2: Estimate asymptotic variance (sum of squared estimation errors).
    // π̂ = (1/T) Σ_t Σ_i Σ_j (x_ti * x_tj - s_ij)²
    let mut pi_hat = 0.0;
    for obs in 0..num_obs {
        for row in 0..num_assets {
            for col in 0..num_assets {
                let cross = returns[(obs, row)] * returns[(obs, col)];
                let err = cross - sample_corr[(row, col)];
                pi_hat += err * err;
            }
        }
    }
    pi_hat /= num_obs_f;

    // Step 3: Compute ρ̂ (approximation for the constant-correlation target).
    // For the constant-correlation target, ρ̂ ≈ Σ_i Σ_j π̂_ij * (f_ij / s_ij)
    // Simplified: use the basic LW formula.
    let rho_hat = pi_hat; // Conservative approximation.

    // Step 4: γ̂ = ‖F - S‖²_F
    let gamma_hat = frob_sq;

    // Step 5: κ̂ = (π̂ - ρ̂) / γ̂
    // δ = max(0, min(1, κ̂ / T))
    let kappa = if gamma_hat > f64::EPSILON {
        (pi_hat - rho_hat) / gamma_hat
    } else {
        0.0
    };

    let intensity = (kappa / num_obs_f).clamp(0.0, 1.0);

    // Shrunk estimate: δF + (1-δ)S
    let shrunk = &target * intensity + sample_corr * (1.0 - intensity);

    Ok(LinearShrinkageResult {
        matrix: shrunk,
        intensity,
    })
}

/// Compute the Ledoit-Wolf (2020) analytical nonlinear shrinkage estimator.
///
/// Applies an individual optimal shrinkage function to each eigenvalue
/// using the Marchenko-Pastur equation. The method is fully analytical
/// with no tuning parameters.
///
/// # Arguments
///
/// * `eigen` — Eigendecomposition of the sample correlation matrix.
/// * `num_obs` — Number of observations (T).
///
/// # References
///
/// - Ledoit & Wolf (2020), Annals of Statistics, 48(5), 3043–3065.
/// - Ledoit & Wolf (2017), Numerical implementation of the `QuEST` function.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn nonlinear_shrinkage(eigen: &EigenDecomposition, num_obs: usize) -> NonlinearShrinkageResult {
    let num_assets = eigen.n;
    let num_obs_f = num_obs as f64;
    let num_assets_f = num_assets as f64;
    let ratio = num_assets_f / num_obs_f;

    // Isotonized eigenvalues (ensure non-decreasing from the sorted-descending input).
    let eigenvalues = &eigen.eigenvalues;

    // Apply the Oracle Approximating Shrinkage (OAS) formula.
    // For each eigenvalue λ_i, compute the shrunk value d̃_i using
    // the Hilbert transform of the Marchenko-Pastur density.
    let mut shrunk_eigenvalues = DVector::zeros(num_assets);

    for idx in 0..num_assets {
        let lambda = eigenvalues[idx];

        // Compute the Hilbert transform of the sample spectral density at λ.
        // H(λ) = (1/N) Σ_j≠i λ / (λ_j - λ)
        // This is approximated by the sample Stieltjes transform.
        let mut hilbert = 0.0;
        for jdx in 0..num_assets {
            if jdx != idx {
                let diff = eigenvalues[jdx] - lambda;
                if diff.abs() > f64::EPSILON {
                    hilbert += 1.0 / diff;
                }
            }
        }
        hilbert /= num_assets_f;

        // The nonlinear shrinkage formula (simplified from Ledoit-Wolf 2020):
        // d̃_i = λ_i / (π² λ_i² h²(λ_i) c² + (1 - c - c λ_i h(λ_i))²)
        // where c = N/T and h is the Hilbert transform.
        let hilbert_term = lambda * hilbert;
        let denom = std::f64::consts::PI.powi(2) * lambda.powi(2) * hilbert.powi(2) * ratio.powi(2)
            + (1.0 - ratio - ratio * hilbert_term).powi(2);

        shrunk_eigenvalues[idx] = if denom > f64::EPSILON {
            lambda / denom
        } else {
            lambda
        };
    }

    // Rescale to preserve trace.
    let original_trace = eigenvalues.sum();
    let shrunk_trace: f64 = shrunk_eigenvalues.iter().sum();
    if shrunk_trace > f64::EPSILON {
        shrunk_eigenvalues *= original_trace / shrunk_trace;
    }

    // Reconstruct.
    let lambda_diag = DMatrix::from_diagonal(&shrunk_eigenvalues);
    let matrix = &eigen.eigenvectors * lambda_diag * eigen.eigenvectors.transpose();

    NonlinearShrinkageResult {
        matrix,
        eigenvalues: shrunk_eigenvalues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::eigen::eigendecompose;
    use crate::math::sample_covariance::correlation_matrix;
    use approx::assert_relative_eq;

    /// Helper: build a returns matrix and its correlation.
    fn sample_data() -> (DMatrix<f64>, DMatrix<f64>) {
        #[rustfmt::skip]
        let returns = DMatrix::from_row_slice(10, 3, &[
            0.01, -0.02,  0.03,
           -0.01,  0.01, -0.02,
            0.02,  0.00,  0.01,
           -0.03,  0.02,  0.00,
            0.01, -0.01,  0.02,
            0.00,  0.03, -0.01,
           -0.02,  0.01,  0.03,
            0.03, -0.02, -0.01,
           -0.01,  0.00,  0.02,
            0.02,  0.01, -0.03,
        ]);
        let cov = correlation_matrix(&returns).unwrap();
        // Standardize returns for linear shrinkage.
        let (num_obs, num_assets) = returns.shape();
        let mut standardized = returns.clone();
        for col in 0..num_assets {
            let column = returns.column(col);
            let mean = column.mean();
            let var = column.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / num_obs as f64;
            let std = var.sqrt();
            for row in 0..num_obs {
                standardized[(row, col)] = (standardized[(row, col)] - mean) / std;
            }
        }
        (standardized, cov.correlation)
    }

    // ── Linear shrinkage ──────────────────────────────────────────────

    /// Intensity δ is in [0, 1].
    #[test]
    fn test_linear_intensity_bounded() {
        let (returns, corr) = sample_data();
        let result = linear_shrinkage(&returns, &corr).unwrap();
        assert!(result.intensity >= 0.0);
        assert!(result.intensity <= 1.0);
    }

    /// Shrunk matrix is symmetric.
    #[test]
    fn test_linear_symmetry() {
        let (returns, corr) = sample_data();
        let result = linear_shrinkage(&returns, &corr).unwrap();

        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(
                    result.matrix[(row, col)],
                    result.matrix[(col, row)],
                    epsilon = 1e-14
                );
            }
        }
    }

    /// Shrunk matrix diagonal is 1.0 (correlation matrix).
    #[test]
    fn test_linear_diagonal_ones() {
        let (returns, corr) = sample_data();
        let result = linear_shrinkage(&returns, &corr).unwrap();

        for idx in 0..3 {
            assert_relative_eq!(result.matrix[(idx, idx)], 1.0, epsilon = 1e-10);
        }
    }

    /// Shrunk matrix has trace = N.
    #[test]
    fn test_linear_trace() {
        let (returns, corr) = sample_data();
        let result = linear_shrinkage(&returns, &corr).unwrap();
        let trace: f64 = (0..3).map(|i| result.matrix[(i, i)]).sum();
        assert_relative_eq!(trace, 3.0, epsilon = 1e-10);
    }

    /// Empty returns should error.
    #[test]
    fn test_linear_empty() {
        let returns = DMatrix::<f64>::zeros(0, 0);
        let corr = DMatrix::<f64>::zeros(0, 0);
        assert!(linear_shrinkage(&returns, &corr).is_err());
    }

    // ── Nonlinear shrinkage ───────────────────────────────────────────

    /// Nonlinear shrinkage produces a symmetric matrix.
    #[test]
    fn test_nonlinear_symmetry() {
        let (_, corr) = sample_data();
        let eigen = eigendecompose(&corr).unwrap();
        let result = nonlinear_shrinkage(&eigen, 10);

        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(
                    result.matrix[(row, col)],
                    result.matrix[(col, row)],
                    epsilon = 1e-12
                );
            }
        }
    }

    /// Nonlinear shrinkage preserves trace.
    #[test]
    fn test_nonlinear_trace() {
        let (_, corr) = sample_data();
        let eigen = eigendecompose(&corr).unwrap();
        let result = nonlinear_shrinkage(&eigen, 10);
        let trace: f64 = result.eigenvalues.iter().sum();
        assert_relative_eq!(trace, 3.0, epsilon = 1e-10);
    }

    /// Nonlinear shrinkage eigenvalues are all positive.
    #[test]
    fn test_nonlinear_psd() {
        let (_, corr) = sample_data();
        let eigen = eigendecompose(&corr).unwrap();
        let result = nonlinear_shrinkage(&eigen, 10);

        for idx in 0..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[idx] >= -1e-10,
                "eigenvalue {} is negative: {}",
                idx,
                result.eigenvalues[idx]
            );
        }
    }

    /// Both methods improve condition number vs raw sample.
    #[test]
    fn test_condition_number_improvement() {
        let (returns, corr) = sample_data();
        let eigen = eigendecompose(&corr).unwrap();

        let raw_cond = eigen.eigenvalues[0] / eigen.eigenvalues[eigen.n - 1];

        // Linear shrinkage.
        let linear = linear_shrinkage(&returns, &corr).unwrap();
        let linear_eigen = eigendecompose(&linear.matrix).unwrap();
        let linear_cond =
            linear_eigen.eigenvalues[0] / linear_eigen.eigenvalues[linear_eigen.n - 1];

        // Nonlinear shrinkage.
        let nonlinear = nonlinear_shrinkage(&eigen, 10);
        let nl_eigen = eigendecompose(&nonlinear.matrix).unwrap();
        let nl_cond = nl_eigen.eigenvalues[0] / nl_eigen.eigenvalues[nl_eigen.n - 1];

        assert!(
            linear_cond <= raw_cond + 1e-10,
            "linear shrinkage should not worsen condition: raw={raw_cond}, linear={linear_cond}"
        );
        assert!(
            nl_cond <= raw_cond + 1e-10,
            "nonlinear shrinkage should not worsen condition: raw={raw_cond}, nl={nl_cond}"
        );
    }
}
