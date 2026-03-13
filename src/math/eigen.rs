// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Eigendecomposition of symmetric matrices.
//!
//! Wraps `nalgebra::SymmetricEigen` to produce eigenvalues sorted in
//! descending order with corresponding eigenvectors.
//!
//! Given a symmetric matrix C, computes C = V Λ Vᵀ where:
//! - Λ is a diagonal matrix of eigenvalues (sorted descending).
//! - V is the matrix of corresponding eigenvectors (columns).
//!
//! # References
//!
//! - Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*,
//!   4th edition, Chapter 8.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use thiserror::Error;

/// Errors that can occur during eigendecomposition.
#[derive(Debug, Error)]
pub enum EigenError {
    /// Input matrix is not square.
    #[error("matrix must be square, got {rows}×{cols}")]
    NotSquare { rows: usize, cols: usize },

    /// Input matrix is empty.
    #[error("empty matrix")]
    EmptyMatrix,
}

/// Result of an eigendecomposition, with eigenvalues sorted descending.
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    /// Eigenvalues sorted in descending order.
    pub eigenvalues: DVector<f64>,
    /// Eigenvectors as columns, ordered to match `eigenvalues`.
    pub eigenvectors: DMatrix<f64>,
    /// Dimension N.
    pub n: usize,
}

impl EigenDecomposition {
    /// Reconstruct the matrix from its eigendecomposition: V Λ Vᵀ.
    #[must_use]
    pub fn reconstruct(&self) -> DMatrix<f64> {
        let lambda = DMatrix::from_diagonal(&self.eigenvalues);
        &self.eigenvectors * lambda * self.eigenvectors.transpose()
    }

    /// Sum of eigenvalues (trace of the original matrix).
    #[must_use]
    pub fn trace(&self) -> f64 {
        self.eigenvalues.sum()
    }

    /// Number of eigenvalues above a threshold.
    #[must_use]
    pub fn count_above(&self, threshold: f64) -> usize {
        self.eigenvalues.iter().filter(|&&v| v > threshold).count()
    }
}

/// Compute the eigendecomposition of a symmetric matrix.
///
/// Returns eigenvalues sorted in descending order with corresponding
/// eigenvectors. The input matrix must be square and symmetric;
/// only the lower triangle is read by nalgebra.
///
/// # Errors
///
/// Returns [`EigenError`] if the matrix is not square or is empty.
pub fn eigendecompose(matrix: &DMatrix<f64>) -> Result<EigenDecomposition, EigenError> {
    let (rows, cols) = matrix.shape();

    if rows == 0 || cols == 0 {
        return Err(EigenError::EmptyMatrix);
    }

    if rows != cols {
        return Err(EigenError::NotSquare { rows, cols });
    }

    let eigen = SymmetricEigen::new(matrix.clone());

    // nalgebra returns eigenvalues in arbitrary order. Sort descending.
    let n = rows;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eigen.eigenvalues[b]
            .partial_cmp(&eigen.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eigenvalues = DVector::from_fn(n, |i, _| eigen.eigenvalues[indices[i]]);
    let eigenvectors = DMatrix::from_fn(n, n, |r, c| eigen.eigenvectors[(r, indices[c])]);

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Identity matrix has all eigenvalues = 1.
    #[test]
    fn test_identity_eigenvalues() {
        let eye = DMatrix::identity(3, 3);
        let result = eigendecompose(&eye).unwrap();

        assert_eq!(result.n, 3);
        for i in 0..3 {
            assert_relative_eq!(result.eigenvalues[i], 1.0, epsilon = 1e-10);
        }
    }

    /// Eigenvalues should be sorted descending.
    #[test]
    fn test_sorted_descending() {
        #[rustfmt::skip]
        let m = DMatrix::from_row_slice(3, 3, &[
            3.0, 1.0, 0.0,
            1.0, 2.0, 1.0,
            0.0, 1.0, 1.0,
        ]);
        let result = eigendecompose(&m).unwrap();

        for i in 1..result.n {
            assert!(
                result.eigenvalues[i - 1] >= result.eigenvalues[i],
                "eigenvalues not sorted descending at index {i}"
            );
        }
    }

    /// Reconstruction: V Λ Vᵀ ≈ original matrix.
    #[test]
    fn test_reconstruction() {
        #[rustfmt::skip]
        let m = DMatrix::from_row_slice(3, 3, &[
            2.0, 1.0, 0.5,
            1.0, 3.0, 1.0,
            0.5, 1.0, 2.0,
        ]);
        let result = eigendecompose(&m).unwrap();
        let reconstructed = result.reconstruct();

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(m[(i, j)], reconstructed[(i, j)], epsilon = 1e-10);
            }
        }
    }

    /// Trace preservation: sum of eigenvalues = trace of matrix.
    #[test]
    fn test_trace_preservation() {
        #[rustfmt::skip]
        let m = DMatrix::from_row_slice(3, 3, &[
            2.0, 1.0, 0.5,
            1.0, 3.0, 1.0,
            0.5, 1.0, 2.0,
        ]);
        let result = eigendecompose(&m).unwrap();
        let trace: f64 = (0..3).map(|i| m[(i, i)]).sum();
        assert_relative_eq!(result.trace(), trace, epsilon = 1e-10);
    }

    /// For a correlation matrix (diagonal = 1), trace = N.
    #[test]
    fn test_correlation_trace_equals_n() {
        #[rustfmt::skip]
        let corr = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.5, 0.3,
            0.5, 1.0, 0.4,
            0.3, 0.4, 1.0,
        ]);
        let result = eigendecompose(&corr).unwrap();
        assert_relative_eq!(result.trace(), 3.0, epsilon = 1e-10);
    }

    /// Eigenvectors should be orthonormal: VᵀV ≈ I.
    #[test]
    fn test_orthonormal_eigenvectors() {
        #[rustfmt::skip]
        let m = DMatrix::from_row_slice(3, 3, &[
            2.0, 1.0, 0.5,
            1.0, 3.0, 1.0,
            0.5, 1.0, 2.0,
        ]);
        let result = eigendecompose(&m).unwrap();
        let vtv = result.eigenvectors.transpose() * &result.eigenvectors;

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(vtv[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    /// Non-square matrix should error.
    #[test]
    fn test_non_square() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(eigendecompose(&m).is_err());
    }

    /// Empty matrix should error.
    #[test]
    fn test_empty() {
        let m = DMatrix::<f64>::zeros(0, 0);
        assert!(eigendecompose(&m).is_err());
    }

    /// `count_above` works correctly.
    #[test]
    fn test_count_above() {
        let eye = DMatrix::identity(5, 5);
        let result = eigendecompose(&eye).unwrap();
        assert_eq!(result.count_above(0.5), 5);
        assert_eq!(result.count_above(1.5), 0);
    }
}
