// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Market mode removal (detoning).
//!
//! Removes the top k eigenmodes from a correlation matrix to isolate
//! idiosyncratic structure from systematic market factors. The removed
//! modes can optionally be re-added after denoising.
//!
//! Typical usage: remove k=1 (the market factor) before denoising for
//! minimum-variance portfolio construction; re-add for risk attribution.
//!
//! # References
//!
//! - López de Prado, M. (2018). *Advances in Financial Machine Learning*,
//!   Chapter 2, Section 2.6.

use nalgebra::{DMatrix, DVector};

use super::eigen::EigenDecomposition;

/// Result of detoning, containing the detoned matrix and the removed
/// components for optional re-addition.
#[derive(Debug, Clone)]
pub struct DetoneResult {
    /// Correlation matrix with top k eigenmodes removed.
    pub matrix: DMatrix<f64>,
    /// Eigenvalues after detoning (top k zeroed out).
    pub eigenvalues: DVector<f64>,
    /// The removed eigenvalues (for re-addition).
    pub removed_eigenvalues: DVector<f64>,
    /// Number of modes removed.
    pub k: usize,
}

/// Remove the top k eigenmodes from a correlation matrix.
///
/// Sets the largest k eigenvalues to zero and reconstructs the matrix.
/// The removed components are stored in the result for optional re-addition
/// via [`readd_tones`].
///
/// # Arguments
///
/// * `eigen` — Eigendecomposition with eigenvalues sorted descending.
/// * `k` — Number of eigenmodes to remove (default: 1).
///
/// # Panics
///
/// Panics if `k > eigen.n` (cannot remove more modes than dimensions).
#[must_use]
pub fn detone(eigen: &EigenDecomposition, k: usize) -> DetoneResult {
    assert!(
        k <= eigen.n,
        "cannot remove {k} modes from a {n}-dimensional matrix",
        n = eigen.n
    );

    let mut detoned_eigenvalues = eigen.eigenvalues.clone();
    let mut removed_eigenvalues = DVector::zeros(eigen.n);

    // Zero out the top k eigenvalues.
    for idx in 0..k {
        removed_eigenvalues[idx] = detoned_eigenvalues[idx];
        detoned_eigenvalues[idx] = 0.0;
    }

    // Reconstruct: C_detoned = V Λ_detoned Vᵀ
    let lambda_diag = DMatrix::from_diagonal(&detoned_eigenvalues);
    let matrix = &eigen.eigenvectors * lambda_diag * eigen.eigenvectors.transpose();

    DetoneResult {
        matrix,
        eigenvalues: detoned_eigenvalues,
        removed_eigenvalues,
        k,
    }
}

/// Re-add previously removed eigenmodes to a (possibly denoised) matrix.
///
/// Combines the denoised eigenvalues with the removed tones and reconstructs.
/// This is used when the full covariance structure (including market mode) is
/// needed after cleaning the idiosyncratic part.
///
/// # Arguments
///
/// * `denoised_eigenvalues` — Eigenvalues after denoising the detoned matrix.
/// * `detone_result` — The original detoning result (contains removed eigenvalues).
/// * `eigenvectors` — The original eigenvector matrix V.
#[must_use]
pub fn readd_tones(
    denoised_eigenvalues: &DVector<f64>,
    detone_result: &DetoneResult,
    eigenvectors: &DMatrix<f64>,
) -> DMatrix<f64> {
    let combined = denoised_eigenvalues + &detone_result.removed_eigenvalues;
    let lambda_diag = DMatrix::from_diagonal(&combined);
    eigenvectors * lambda_diag * eigenvectors.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::eigen::eigendecompose;
    use approx::assert_relative_eq;

    /// Helper: build a known correlation matrix and eigendecompose.
    fn sample_eigen() -> EigenDecomposition {
        #[rustfmt::skip]
        let corr = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.8, 0.5,
            0.8, 1.0, 0.6,
            0.5, 0.6, 1.0,
        ]);
        eigendecompose(&corr).unwrap()
    }

    // ── Removing top eigenmode ────────────────────────────────────────

    /// After removing k=1, the largest eigenvalue in the detoned result is zero.
    #[test]
    fn test_detone_removes_largest() {
        let eigen = sample_eigen();
        let result = detone(&eigen, 1);

        assert_relative_eq!(result.eigenvalues[0], 0.0, epsilon = 1e-14);
        assert!(result.eigenvalues[1] > 0.0);
    }

    /// After removing k=2, the two largest eigenvalues are zero.
    #[test]
    fn test_detone_removes_k2() {
        let eigen = sample_eigen();
        let result = detone(&eigen, 2);

        assert_relative_eq!(result.eigenvalues[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(result.eigenvalues[1], 0.0, epsilon = 1e-14);
        assert!(result.eigenvalues[2] > 0.0);
    }

    /// Removing k=0 returns the original matrix unchanged.
    #[test]
    fn test_detone_k0_unchanged() {
        let eigen = sample_eigen();
        let result = detone(&eigen, 0);
        let original = eigen.reconstruct();

        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(
                    result.matrix[(row, col)],
                    original[(row, col)],
                    epsilon = 1e-12
                );
            }
        }
    }

    // ── Re-addition ──────────────────────────────────────────────────

    /// Detone + re-add reconstructs the original matrix.
    #[test]
    fn test_readd_reconstructs_original() {
        let eigen = sample_eigen();
        let detone_result = detone(&eigen, 1);
        let readded = readd_tones(
            &detone_result.eigenvalues,
            &detone_result,
            &eigen.eigenvectors,
        );
        let original = eigen.reconstruct();

        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(readded[(row, col)], original[(row, col)], epsilon = 1e-12);
            }
        }
    }

    // ── Symmetry ──────────────────────────────────────────────────────

    /// Detoned matrix is symmetric.
    #[test]
    fn test_detoned_symmetry() {
        let eigen = sample_eigen();
        let result = detone(&eigen, 1);

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

    // ── Stored removed eigenvalues ───────────────────────────────────

    /// The removed eigenvalues are stored correctly.
    #[test]
    fn test_removed_eigenvalues_stored() {
        let eigen = sample_eigen();
        let original_top = eigen.eigenvalues[0];
        let result = detone(&eigen, 1);

        assert_relative_eq!(result.removed_eigenvalues[0], original_top, epsilon = 1e-14);
        assert_relative_eq!(result.removed_eigenvalues[1], 0.0, epsilon = 1e-14);
        assert_relative_eq!(result.removed_eigenvalues[2], 0.0, epsilon = 1e-14);
    }

    // ── Panic on k > n ───────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "cannot remove")]
    fn test_detone_k_too_large() {
        let eigen = sample_eigen();
        let _ = detone(&eigen, 4);
    }
}
