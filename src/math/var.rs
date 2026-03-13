// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Portfolio Value at Risk (`VaR`) computation.
//!
//! Two methods are implemented:
//!
//! ## Parametric (Gaussian) `VaR`
//!
//! Assumes returns are normally distributed:
//!
//! ```text
//! VaR_α = −(μ_p + z_α · σ_p)
//! ```
//!
//! where `μ_p` = wᵀμ is the portfolio expected return, `σ_p` = √(wᵀΣw) is the
//! portfolio volatility, and `z_α` is the standard normal quantile.
//!
//! ## Cornish-Fisher `VaR`
//!
//! Adjusts the Gaussian quantile for skewness (S) and excess kurtosis (K):
//!
//! ```text
//! z_CF = z_α + (z_α² − 1)S/6 + (z_α³ − 3z_α)K/24 − (2z_α³ − 5z_α)S²/36
//! ```
//!
//! This captures non-Gaussian tail behavior without requiring full
//! distributional assumptions.
//!
//! # References
//!
//! - PRIIPs Delegated Regulation (EU) 2017/653, Annex II, Part 1.
//! - Cornish, E. A., & Fisher, R. A. (1938). Moments and cumulants in the
//!   specification of distributions. *Revue de l'Institut International de
//!   Statistique*, 5(4), 307–320.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors from `VaR` computation.
#[derive(Debug, Error)]
pub enum VarError {
    /// Portfolio weights dimension does not match covariance matrix.
    #[error("weights length {weights} does not match covariance dimension {cov_dim}")]
    DimensionMismatch {
        /// Length of the weights vector.
        weights: usize,
        /// Dimension of the covariance matrix.
        cov_dim: usize,
    },

    /// Invalid confidence level (must be in (0, 1)).
    #[error("confidence level must be in (0, 1), got {0}")]
    InvalidConfidence(f64),

    /// Portfolio volatility is zero or negative.
    #[error("portfolio volatility is zero — all assets may be identical")]
    ZeroVolatility,
}

/// `VaR` computation result.
#[derive(Debug, Clone)]
pub struct VarResult {
    /// Value at Risk (positive number representing a loss).
    pub var: f64,
    /// Portfolio volatility `σ_p`.
    pub portfolio_volatility: f64,
    /// Portfolio expected return `μ_p`.
    pub portfolio_return: f64,
    /// Confidence level α used.
    pub confidence: f64,
}

/// Compute the standard normal quantile (inverse CDF) using the
/// rational approximation by Abramowitz and Stegun (1964), formula 26.2.23.
///
/// Accurate to ~4.5 × 10⁻⁴ for p ∈ (0, 1).
#[must_use]
fn normal_quantile(p: f64) -> f64 {
    // Use symmetry: for p > 0.5, compute for 1-p and negate.
    if p < 0.5 {
        return -normal_quantile(1.0 - p);
    }

    // Constants for the rational approximation.
    let t = (-2.0 * (1.0 - p).ln()).sqrt();

    // Coefficients from Abramowitz & Stegun.
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;

    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

/// Compute parametric (Gaussian) portfolio `VaR`.
///
/// # Arguments
///
/// * `weights` — Portfolio weight vector (should sum to 1).
/// * `expected_returns` — Expected return vector μ (same length as weights).
/// * `covariance` — Covariance (or correlation) matrix Σ.
/// * `confidence` — Confidence level α ∈ (0, 1), e.g., 0.975 for PRIIPs.
///
/// # Errors
///
/// Returns [`VarError`] if dimensions mismatch or confidence is invalid.
///
/// # References
///
/// - PRIIPs RTS, Annex II, Part 1, Section 12.
#[allow(clippy::cast_precision_loss)]
pub fn parametric_var(
    weights: &DVector<f64>,
    expected_returns: &DVector<f64>,
    covariance: &DMatrix<f64>,
    confidence: f64,
) -> Result<VarResult, VarError> {
    validate_inputs(weights, covariance, confidence)?;

    let portfolio_return = weights.dot(expected_returns);
    let portfolio_variance = (weights.transpose() * covariance * weights)[(0, 0)];
    let portfolio_volatility = portfolio_variance.sqrt();

    if portfolio_volatility < f64::EPSILON {
        return Err(VarError::ZeroVolatility);
    }

    let z_alpha = normal_quantile(confidence);
    let var = -(portfolio_return - z_alpha * portfolio_volatility);

    Ok(VarResult {
        var,
        portfolio_volatility,
        portfolio_return,
        confidence,
    })
}

/// Compute Cornish-Fisher adjusted portfolio `VaR`.
///
/// Adjusts the Gaussian quantile for portfolio skewness and excess kurtosis
/// to capture non-Gaussian tail behavior.
///
/// # Arguments
///
/// * `weights` — Portfolio weight vector.
/// * `expected_returns` — Expected return vector μ.
/// * `covariance` — Covariance (or correlation) matrix Σ.
/// * `confidence` — Confidence level α ∈ (0, 1).
/// * `skewness` — Portfolio return skewness S.
/// * `excess_kurtosis` — Portfolio return excess kurtosis K.
///
/// # Errors
///
/// Returns [`VarError`] if dimensions mismatch or confidence is invalid.
///
/// # References
///
/// - Cornish & Fisher (1938).
/// - PRIIPs RTS, Annex II, Part 1, Section 13.
#[allow(clippy::cast_precision_loss)]
pub fn cornish_fisher_var(
    weights: &DVector<f64>,
    expected_returns: &DVector<f64>,
    covariance: &DMatrix<f64>,
    confidence: f64,
    skewness: f64,
    excess_kurtosis: f64,
) -> Result<VarResult, VarError> {
    validate_inputs(weights, covariance, confidence)?;

    let portfolio_return = weights.dot(expected_returns);
    let portfolio_variance = (weights.transpose() * covariance * weights)[(0, 0)];
    let portfolio_volatility = portfolio_variance.sqrt();

    if portfolio_volatility < f64::EPSILON {
        return Err(VarError::ZeroVolatility);
    }

    // Cornish-Fisher expansion applied to the lower quantile (1 - α).
    // z is the quantile at (1 - α), which is negative for α > 0.5.
    let z = normal_quantile(1.0 - confidence);

    let z_cf = z + (z * z - 1.0) * skewness / 6.0 + (z.powi(3) - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z.powi(3) - 5.0 * z) * skewness * skewness / 36.0;

    // VaR = -(μ + z_CF * σ), where z_CF is negative → VaR is positive.
    let var = -(portfolio_return + z_cf * portfolio_volatility);

    Ok(VarResult {
        var,
        portfolio_volatility,
        portfolio_return,
        confidence,
    })
}

/// Validate common inputs for `VaR` functions.
fn validate_inputs(
    weights: &DVector<f64>,
    covariance: &DMatrix<f64>,
    confidence: f64,
) -> Result<(), VarError> {
    let (rows, cols) = covariance.shape();
    if weights.len() != rows || rows != cols {
        return Err(VarError::DimensionMismatch {
            weights: weights.len(),
            cov_dim: rows,
        });
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(VarError::InvalidConfidence(confidence));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper: equal-weight portfolio on a known covariance.
    fn test_setup() -> (DVector<f64>, DVector<f64>, DMatrix<f64>) {
        let weights = DVector::from_vec(vec![0.5, 0.5]);
        let expected_returns = DVector::from_vec(vec![0.0, 0.0]);
        #[rustfmt::skip]
        let cov = DMatrix::from_row_slice(2, 2, &[
            0.04, 0.01,
            0.01, 0.04,
        ]);
        (weights, expected_returns, cov)
    }

    // ── Normal quantile ─────────────────────────────────────────────

    /// z(0.975) ≈ 1.96.
    #[test]
    fn test_normal_quantile_975() {
        let z = normal_quantile(0.975);
        assert_relative_eq!(z, 1.96, epsilon = 0.01);
    }

    /// z(0.5) = 0 (symmetric).
    #[test]
    fn test_normal_quantile_50() {
        let z = normal_quantile(0.5);
        assert_relative_eq!(z, 0.0, epsilon = 0.01);
    }

    /// z(0.025) ≈ -1.96 (symmetry).
    #[test]
    fn test_normal_quantile_025() {
        let z = normal_quantile(0.025);
        assert_relative_eq!(z, -1.96, epsilon = 0.01);
    }

    // ── Parametric VaR ──────────────────────────────────────────────

    /// Parametric VaR is positive for zero-mean portfolios.
    #[test]
    fn test_parametric_var_positive() {
        let (weights, expected_returns, cov) = test_setup();
        let result = parametric_var(&weights, &expected_returns, &cov, 0.975).unwrap();
        assert!(result.var > 0.0, "VaR should be positive: {}", result.var);
    }

    /// VaR scales with confidence level: higher confidence → higher VaR.
    #[test]
    fn test_var_scales_with_confidence() {
        let (weights, expected_returns, cov) = test_setup();
        let var_95 = parametric_var(&weights, &expected_returns, &cov, 0.95)
            .unwrap()
            .var;
        let var_975 = parametric_var(&weights, &expected_returns, &cov, 0.975)
            .unwrap()
            .var;
        let var_99 = parametric_var(&weights, &expected_returns, &cov, 0.99)
            .unwrap()
            .var;

        assert!(var_99 > var_975);
        assert!(var_975 > var_95);
    }

    /// Known analytical VaR for equal-weight, zero-mean, 2-asset portfolio.
    /// σ_p = √(0.5² × 0.04 + 0.5² × 0.04 + 2 × 0.5 × 0.5 × 0.01)
    ///      = √(0.01 + 0.01 + 0.005) = √0.025 ≈ 0.15811
    /// VaR_0.975 = z_0.975 × σ_p ≈ 1.96 × 0.15811 ≈ 0.3099
    #[test]
    fn test_parametric_var_known_value() {
        let (weights, expected_returns, cov) = test_setup();
        let result = parametric_var(&weights, &expected_returns, &cov, 0.975).unwrap();

        assert_relative_eq!(
            result.portfolio_volatility,
            0.025_f64.sqrt(),
            epsilon = 1e-10
        );
        // Allow for approximation error in normal quantile.
        assert_relative_eq!(result.var, 1.96 * 0.025_f64.sqrt(), epsilon = 0.01);
    }

    /// Dimension mismatch should error.
    #[test]
    fn test_parametric_var_dimension_mismatch() {
        let weights = DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let expected_returns = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let cov = DMatrix::identity(2, 2);
        assert!(parametric_var(&weights, &expected_returns, &cov, 0.975).is_err());
    }

    /// Invalid confidence should error.
    #[test]
    fn test_parametric_var_invalid_confidence() {
        let (weights, expected_returns, cov) = test_setup();
        assert!(parametric_var(&weights, &expected_returns, &cov, 0.0).is_err());
        assert!(parametric_var(&weights, &expected_returns, &cov, 1.0).is_err());
    }

    // ── Cornish-Fisher VaR ──────────────────────────────────────────

    /// With zero skewness and kurtosis, CF VaR equals parametric VaR.
    #[test]
    fn test_cf_equals_parametric_when_gaussian() {
        let (weights, expected_returns, cov) = test_setup();
        let param = parametric_var(&weights, &expected_returns, &cov, 0.975).unwrap();
        let cf = cornish_fisher_var(&weights, &expected_returns, &cov, 0.975, 0.0, 0.0).unwrap();

        assert_relative_eq!(param.var, cf.var, epsilon = 1e-10);
    }

    /// Negative skewness (fat left tail) should increase VaR.
    #[test]
    fn test_cf_negative_skewness_increases_var() {
        let (weights, expected_returns, cov) = test_setup();
        let gaussian =
            cornish_fisher_var(&weights, &expected_returns, &cov, 0.975, 0.0, 0.0).unwrap();
        let negskew =
            cornish_fisher_var(&weights, &expected_returns, &cov, 0.975, -1.0, 0.0).unwrap();

        assert!(
            negskew.var > gaussian.var,
            "negative skew should increase VaR: gaussian={}, negskew={}",
            gaussian.var,
            negskew.var
        );
    }

    /// Positive excess kurtosis (fat tails) should increase VaR.
    #[test]
    fn test_cf_excess_kurtosis_increases_var() {
        let (weights, expected_returns, cov) = test_setup();
        let gaussian =
            cornish_fisher_var(&weights, &expected_returns, &cov, 0.975, 0.0, 0.0).unwrap();
        let leptokurtic =
            cornish_fisher_var(&weights, &expected_returns, &cov, 0.975, 0.0, 3.0).unwrap();

        assert!(
            leptokurtic.var > gaussian.var,
            "excess kurtosis should increase VaR: gaussian={}, lepto={}",
            gaussian.var,
            leptokurtic.var
        );
    }
}
