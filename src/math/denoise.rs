// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Eigenvalue replacement and matrix reconstruction (denoising).
//!
//! After Marchenko-Pastur fitting identifies noise eigenvalues, this module
//! replaces them and reconstructs a cleaned covariance/correlation matrix
//! C̃ = V Λ̃ Vᵀ.
//!
//! Two replacement strategies are supported:
//!
//! - **Constant**: Replace all noise eigenvalues with their mean (preserves trace).
//! - **Target**: Replace noise eigenvalues with 1.0 (identity target), then
//!   rescale the entire spectrum to preserve trace.
//!
//! # References
//!
//! - López de Prado, M. (2018). *Advances in Financial Machine Learning*,
//!   Chapter 2, Code Snippet 2.6.
//! - Bun, J., Bouchaud, J. P., & Potters, M. (2017). Cleaning large
//!   correlation matrices. *Physics Reports*, 666, 1–109, Section 4.

use nalgebra::{DMatrix, DVector};

use super::eigen::EigenDecomposition;
use super::marchenko_pastur::MpFit;

/// Strategy for replacing noise eigenvalues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMethod {
    /// Replace all noise eigenvalues with their mean. Preserves trace exactly.
    Constant,
    /// Replace noise eigenvalues with 1.0 (identity target), then rescale
    /// the full spectrum to preserve trace.
    Target,
}

/// Result of denoising, containing the cleaned matrix and diagnostics.
#[derive(Debug, Clone)]
pub struct DenoiseResult {
    /// Cleaned correlation/covariance matrix (N × N).
    pub matrix: DMatrix<f64>,
    /// Cleaned eigenvalues (descending order).
    pub eigenvalues: DVector<f64>,
    /// Trace of the cleaned matrix.
    pub trace: f64,
    /// Method used.
    pub method: DenoiseMethod,
}

/// Denoise a correlation matrix by replacing noise eigenvalues.
///
/// Takes the eigendecomposition from [`super::eigen::eigendecompose`] and the
/// Marchenko-Pastur fit from [`super::marchenko_pastur::fit_sigma_sq`], replaces
/// noise eigenvalues according to the chosen method, and reconstructs the matrix.
///
/// # Trace preservation
///
/// Both methods preserve the trace of the original matrix:
/// - `Constant`: noise eigenvalues are set to their mean (sum unchanged).
/// - `Target`: the full spectrum is rescaled so the total sum matches.
///
/// # Arguments
///
/// * `eigen` — Eigendecomposition with eigenvalues sorted descending.
/// * `mp_fit` — Marchenko-Pastur fit result.
/// * `method` — Replacement strategy.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn denoise(eigen: &EigenDecomposition, mp_fit: &MpFit, method: DenoiseMethod) -> DenoiseResult {
    let original_trace = eigen.trace();
    let mut cleaned = eigen.eigenvalues.clone();

    match method {
        DenoiseMethod::Constant => {
            // Collect noise eigenvalue indices and their sum.
            let mut noise_sum = 0.0;
            let mut noise_count = 0_usize;

            for idx in 0..cleaned.len() {
                if cleaned[idx] <= mp_fit.lambda_plus {
                    noise_sum += cleaned[idx];
                    noise_count += 1;
                }
            }

            if noise_count > 0 {
                let noise_mean = noise_sum / noise_count as f64;
                for idx in 0..cleaned.len() {
                    if cleaned[idx] <= mp_fit.lambda_plus {
                        cleaned[idx] = noise_mean;
                    }
                }
            }
        }
        DenoiseMethod::Target => {
            // Replace noise eigenvalues with 1.0.
            for idx in 0..cleaned.len() {
                if cleaned[idx] <= mp_fit.lambda_plus {
                    cleaned[idx] = 1.0;
                }
            }

            // Rescale entire spectrum to preserve trace.
            let current_trace: f64 = cleaned.iter().sum();
            if current_trace > f64::EPSILON {
                let scale = original_trace / current_trace;
                cleaned *= scale;
            }
        }
    }

    // Reconstruct: C̃ = V Λ̃ Vᵀ
    let lambda_diag = DMatrix::from_diagonal(&cleaned);
    let matrix = &eigen.eigenvectors * lambda_diag * eigen.eigenvectors.transpose();

    let trace = cleaned.iter().sum();

    DenoiseResult {
        matrix,
        eigenvalues: cleaned,
        trace,
        method,
    }
}

/// Re-normalise a denoised matrix back to a proper correlation matrix
/// (unit diagonal). Standard post-denoising step per de Prado (2018), Ch. 2.
///
/// Given a symmetric PSD matrix **C**, returns **D⁻¹ C D⁻¹** where
/// **D** = diag(√C₁₁, √C₂₂, …).
#[must_use]
pub fn renormalize_to_correlation(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let n = matrix.nrows();
    let inv_std: DVector<f64> = DVector::from_fn(n, |i, _| {
        let d = matrix[(i, i)];
        if d > f64::EPSILON {
            1.0 / d.sqrt()
        } else {
            1.0
        }
    });
    let mut corr = matrix.clone();
    for i in 0..n {
        for j in 0..n {
            corr[(i, j)] *= inv_std[i] * inv_std[j];
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::eigen::eigendecompose;
    use crate::math::marchenko_pastur::fit_sigma_sq;
    use crate::math::sample_covariance::correlation_matrix;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    /// Helper: build a correlation matrix from returns and denoise it.
    fn denoise_from_returns(returns: &DMatrix<f64>, method: DenoiseMethod) -> DenoiseResult {
        let cov = correlation_matrix(returns).unwrap();
        let eigen = eigendecompose(&cov.correlation).unwrap();
        let mp_fit = fit_sigma_sq(&eigen.eigenvalues, cov.q).unwrap();
        denoise(&eigen, &mp_fit, method)
    }

    // ── Trace preservation ────────────────────────────────────────────

    /// Trace is preserved after Constant denoising.
    #[test]
    fn test_trace_preservation_constant() {
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
        let result = denoise_from_returns(&returns, DenoiseMethod::Constant);
        assert_relative_eq!(result.trace, 3.0, epsilon = 1e-10);
    }

    /// Trace is preserved after Target denoising.
    #[test]
    fn test_trace_preservation_target() {
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
        let result = denoise_from_returns(&returns, DenoiseMethod::Target);
        assert_relative_eq!(result.trace, 3.0, epsilon = 1e-10);
    }

    // ── Symmetry ──────────────────────────────────────────────────────

    /// Denoised matrix is symmetric.
    #[test]
    fn test_denoised_symmetry() {
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
        let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

        let num_assets = result.matrix.nrows();
        for row in 0..num_assets {
            for col in 0..num_assets {
                assert_relative_eq!(
                    result.matrix[(row, col)],
                    result.matrix[(col, row)],
                    epsilon = 1e-12
                );
            }
        }
    }

    // ── PSD ───────────────────────────────────────────────────────────

    /// Denoised matrix is positive semi-definite (all eigenvalues ≥ 0).
    #[test]
    fn test_denoised_psd() {
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
        let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

        for idx in 0..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[idx] >= -1e-10,
                "eigenvalue {} is negative: {}",
                idx,
                result.eigenvalues[idx]
            );
        }
    }

    // ── Pure noise → ~identity ────────────────────────────────────────

    /// Denoising a near-identity matrix (pure noise) should yield
    /// approximately the identity (scaled so trace = N).
    #[test]
    fn test_pure_noise_approaches_identity() {
        // Build a correlation matrix from random-ish returns.
        // With T >> N and independent columns, the correlation matrix
        // should be close to identity, and denoising should push it closer.
        let num_obs = 500;
        let num_assets = 5;
        let mut data = vec![0.0_f64; num_obs * num_assets];

        // Deterministic pseudo-random via simple LCG.
        let mut seed: u64 = 42;
        for val in &mut data {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *val = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        }

        let returns = DMatrix::from_row_slice(num_obs, num_assets, &data);
        let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

        // Diagonal should be close to 1.0.
        for idx in 0..num_assets {
            assert_relative_eq!(result.matrix[(idx, idx)], 1.0, epsilon = 0.15);
        }

        // Off-diagonal should be close to 0.0.
        for row in 0..num_assets {
            for col in 0..num_assets {
                if row != col {
                    assert!(
                        result.matrix[(row, col)].abs() < 0.3,
                        "off-diagonal ({},{}) too large: {}",
                        row,
                        col,
                        result.matrix[(row, col)]
                    );
                }
            }
        }
    }

    // ── Eigenvalues sorted ────────────────────────────────────────────

    /// Cleaned eigenvalues remain sorted descending.
    #[test]
    fn test_cleaned_eigenvalues_sorted() {
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
        let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

        for idx in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[idx - 1] >= result.eigenvalues[idx] - 1e-10,
                "eigenvalues not sorted at index {}",
                idx
            );
        }
    }
}
