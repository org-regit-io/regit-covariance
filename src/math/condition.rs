// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Condition number monitoring and health classification.
//!
//! The condition number κ = `λ_max` / `λ_min` measures the sensitivity of a
//! matrix to perturbations. For covariance matrices used in portfolio
//! optimization, a high κ indicates that the inverse is numerically
//! unreliable and small estimation errors in the covariance will produce
//! large errors in portfolio weights.
//!
//! # Health classification
//!
//! | κ range      | Label      | Implication                           |
//! |--------------|------------|---------------------------------------|
//! | κ < 100      | Healthy    | Inverse is reliable                   |
//! | 100 ≤ κ < 1000 | Acceptable | Proceed with caution              |
//! | κ ≥ 1000     | Unstable   | Optimization results may be unreliable|
//!
//! # References
//!
//! - Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.
//! - López de Prado, M. (2018). *Advances in Financial Machine Learning*,
//!   Chapter 2.

use super::eigen::EigenDecomposition;

/// Health classification of a matrix based on its condition number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixHealth {
    /// κ < 100 — inverse is numerically reliable.
    Healthy,
    /// 100 ≤ κ < 1000 — proceed with caution.
    Acceptable,
    /// κ ≥ 1000 — optimization results may be unreliable.
    Unstable,
}

/// Result of condition number analysis.
#[derive(Debug, Clone)]
pub struct ConditionReport {
    /// Condition number κ = `λ_max` / `λ_min`.
    pub condition_number: f64,
    /// Health classification based on κ thresholds.
    pub health: MatrixHealth,
    /// Largest eigenvalue.
    pub lambda_max: f64,
    /// Smallest eigenvalue.
    pub lambda_min: f64,
}

/// Result of before/after condition number comparison.
#[derive(Debug, Clone)]
pub struct ConditionImprovement {
    /// Condition report before denoising.
    pub before: ConditionReport,
    /// Condition report after denoising.
    pub after: ConditionReport,
    /// Improvement factor: `κ_before` / `κ_after`.
    pub improvement_factor: f64,
}

/// Classify matrix health from a condition number.
#[must_use]
pub fn classify(condition_number: f64) -> MatrixHealth {
    if condition_number < 100.0 {
        MatrixHealth::Healthy
    } else if condition_number < 1000.0 {
        MatrixHealth::Acceptable
    } else {
        MatrixHealth::Unstable
    }
}

/// Compute the condition number report from an eigendecomposition.
///
/// The condition number is κ = `λ_max` / `λ_min` where eigenvalues are
/// sorted descending. If the smallest eigenvalue is zero or negative,
/// κ is reported as infinity.
#[must_use]
pub fn condition_report(eigen: &EigenDecomposition) -> ConditionReport {
    let lambda_max = eigen.eigenvalues[0];
    let lambda_min = eigen.eigenvalues[eigen.n - 1];

    let condition_number = if lambda_min > f64::EPSILON {
        lambda_max / lambda_min
    } else {
        f64::INFINITY
    };

    ConditionReport {
        condition_number,
        health: classify(condition_number),
        lambda_max,
        lambda_min,
    }
}

/// Compare condition numbers before and after denoising.
///
/// Returns the improvement factor `κ_before` / `κ_after`. A value > 1
/// indicates that denoising improved the conditioning.
#[must_use]
pub fn compare(before: &EigenDecomposition, after: &EigenDecomposition) -> ConditionImprovement {
    let before_report = condition_report(before);
    let after_report = condition_report(after);

    let improvement_factor = if after_report.condition_number > f64::EPSILON {
        before_report.condition_number / after_report.condition_number
    } else {
        f64::INFINITY
    };

    ConditionImprovement {
        before: before_report,
        after: after_report,
        improvement_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::eigen::eigendecompose;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    // ── Classification ──────────────────────────────────────────────

    #[test]
    fn test_classify_healthy() {
        assert_eq!(classify(1.0), MatrixHealth::Healthy);
        assert_eq!(classify(50.0), MatrixHealth::Healthy);
        assert_eq!(classify(99.9), MatrixHealth::Healthy);
    }

    #[test]
    fn test_classify_acceptable() {
        assert_eq!(classify(100.0), MatrixHealth::Acceptable);
        assert_eq!(classify(500.0), MatrixHealth::Acceptable);
        assert_eq!(classify(999.9), MatrixHealth::Acceptable);
    }

    #[test]
    fn test_classify_unstable() {
        assert_eq!(classify(1000.0), MatrixHealth::Unstable);
        assert_eq!(classify(10_000.0), MatrixHealth::Unstable);
        assert_eq!(classify(f64::INFINITY), MatrixHealth::Unstable);
    }

    // ── Condition report ────────────────────────────────────────────

    /// Identity matrix has κ = 1 (perfectly conditioned).
    #[test]
    fn test_identity_condition() {
        let identity = DMatrix::identity(4, 4);
        let eigen = eigendecompose(&identity).unwrap();
        let report = condition_report(&eigen);

        assert_relative_eq!(report.condition_number, 1.0, epsilon = 1e-10);
        assert_eq!(report.health, MatrixHealth::Healthy);
    }

    /// Known correlation matrix with computable condition number.
    #[test]
    fn test_known_condition_number() {
        #[rustfmt::skip]
        let corr = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.9, 0.9,
            0.9, 1.0, 0.9,
            0.9, 0.9, 1.0,
        ]);
        let eigen = eigendecompose(&corr).unwrap();
        let report = condition_report(&eigen);

        // For constant-correlation ρ=0.9, N=3:
        // λ_max = 1 + (N-1)ρ = 1 + 2*0.9 = 2.8
        // λ_min = 1 - ρ = 0.1
        // κ = 28
        assert_relative_eq!(report.condition_number, 28.0, epsilon = 1e-10);
        assert_eq!(report.health, MatrixHealth::Healthy);
    }

    // ── Comparison ──────────────────────────────────────────────────

    /// Denoising should improve condition number (improvement factor > 1).
    #[test]
    fn test_improvement_factor() {
        // "Before": ill-conditioned.
        #[rustfmt::skip]
        let raw = DMatrix::from_row_slice(3, 3, &[
            1.0,  0.99, 0.99,
            0.99, 1.0,  0.99,
            0.99, 0.99, 1.0,
        ]);
        // "After": better-conditioned.
        #[rustfmt::skip]
        let cleaned = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.5, 0.5,
            0.5, 1.0, 0.5,
            0.5, 0.5, 1.0,
        ]);

        let eigen_raw = eigendecompose(&raw).unwrap();
        let eigen_cleaned = eigendecompose(&cleaned).unwrap();
        let result = compare(&eigen_raw, &eigen_cleaned);

        assert!(
            result.improvement_factor > 1.0,
            "denoising should improve condition: factor={}",
            result.improvement_factor
        );
    }

    /// Comparing identical matrices gives improvement factor = 1.
    #[test]
    fn test_no_improvement() {
        let corr = DMatrix::identity(3, 3);
        let eigen = eigendecompose(&corr).unwrap();
        let result = compare(&eigen, &eigen);

        assert_relative_eq!(result.improvement_factor, 1.0, epsilon = 1e-10);
    }
}
