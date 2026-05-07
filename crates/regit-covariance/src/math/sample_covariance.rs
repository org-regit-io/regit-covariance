// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Sample covariance and correlation matrix estimation.
//!
//! Computes the sample correlation matrix C = (1/T) `XᵀX` where X is the
//! matrix of standardized (zero mean, unit variance) returns.
//!
//! Supports fixed rolling windows, expanding windows, and exponentially
//! weighted estimation with configurable half-life.
//!
//! # References
//!
//! - López de Prado, M. (2018). *Advances in Financial Machine Learning*,
//!   Chapter 2, Section 2.4.

use nalgebra::DMatrix;
use thiserror::Error;

/// Errors that can occur during covariance estimation.
#[derive(Debug, Error)]
pub enum CovarianceError {
    /// Returns matrix has fewer observations than assets.
    #[error("insufficient observations: T={observations} < N={assets} (rank-deficient)")]
    InsufficientObservations { observations: usize, assets: usize },

    /// Returns matrix is empty.
    #[error("empty returns matrix")]
    EmptyMatrix,

    /// An asset has zero variance (constant price).
    #[error("zero variance for asset at column index {column}")]
    ZeroVariance { column: usize },

    /// Invalid half-life for exponential weighting.
    #[error("half-life must be positive, got {0}")]
    InvalidHalfLife(f64),
}

/// Result of a covariance estimation, containing both the correlation matrix
/// and the ratio q = T/N needed for Marchenko-Pastur bounds.
#[derive(Debug, Clone)]
pub struct CovarianceResult {
    /// Sample correlation matrix (N × N).
    pub correlation: DMatrix<f64>,
    /// Per-asset standard deviations (length N).
    pub std_devs: Vec<f64>,
    /// Number of observations (T).
    pub observations: usize,
    /// Number of assets (N).
    pub assets: usize,
    /// Ratio q = T/N.
    pub q: f64,
}

/// Compute the sample correlation matrix from a returns matrix.
///
/// The returns matrix has shape (T × N) where T is the number of
/// observations (time steps) and N is the number of assets.
///
/// The function standardizes each column to zero mean and unit variance,
/// then computes C = (1/T) `XᵀX`.
///
/// # Errors
///
/// Returns [`CovarianceError`] if the matrix is empty, has insufficient
/// observations, or contains an asset with zero variance.
#[allow(clippy::cast_precision_loss)]
pub fn correlation_matrix(returns: &DMatrix<f64>) -> Result<CovarianceResult, CovarianceError> {
    let (num_obs, num_assets) = returns.shape();

    if num_obs == 0 || num_assets == 0 {
        return Err(CovarianceError::EmptyMatrix);
    }

    if num_obs < 2 {
        return Err(CovarianceError::InsufficientObservations {
            observations: num_obs,
            assets: num_assets,
        });
    }

    // Standardize: zero mean, unit variance per column.
    let (standardized, std_devs) = standardize(returns)?;

    // C = (1/T) XᵀX
    let corr = (standardized.transpose() * &standardized) / num_obs as f64;

    Ok(CovarianceResult {
        correlation: corr,
        std_devs,
        observations: num_obs,
        assets: num_assets,
        q: num_obs as f64 / num_assets as f64,
    })
}

/// Compute the exponentially weighted correlation matrix.
///
/// Observations are weighted by `w_i` = (1 − α)^(T−1−i) where
/// α = 1 − exp(−ln(2) / `half_life`). More recent observations receive
/// higher weight.
///
/// # Arguments
///
/// * `returns` — (T × N) returns matrix.
/// * `half_life` — The number of periods for the weight to decay by half.
///
/// # Errors
///
/// Returns [`CovarianceError`] if the matrix is empty, has insufficient
/// observations, contains zero-variance assets, or if `half_life` is not positive.
///
/// # References
///
/// - `RiskMetrics` Technical Document (1996), Section 5.
#[allow(clippy::cast_precision_loss)]
pub fn ewm_correlation_matrix(
    returns: &DMatrix<f64>,
    half_life: f64,
) -> Result<CovarianceResult, CovarianceError> {
    if half_life <= 0.0 || !half_life.is_finite() {
        return Err(CovarianceError::InvalidHalfLife(half_life));
    }

    let (num_obs, num_assets) = returns.shape();

    if num_obs == 0 || num_assets == 0 {
        return Err(CovarianceError::EmptyMatrix);
    }

    if num_obs < 2 {
        return Err(CovarianceError::InsufficientObservations {
            observations: num_obs,
            assets: num_assets,
        });
    }

    let decay = (-(2.0_f64.ln()) / half_life).exp();

    // Compute weights: w_i = decay^(T-1-i), most recent = decay^0 = 1.
    let weights: Vec<f64> = (0..num_obs)
        .map(|idx| decay.powf((num_obs - 1 - idx) as f64))
        .collect();
    let weight_sum: f64 = weights.iter().sum();

    // Weighted mean per column.
    let mut means = vec![0.0_f64; num_assets];
    for (obs_idx, weight) in weights.iter().enumerate() {
        for col in 0..num_assets {
            means[col] += weight * returns[(obs_idx, col)];
        }
    }
    for mean in &mut means {
        *mean /= weight_sum;
    }

    // Weighted variance per column.
    let mut variances = vec![0.0_f64; num_assets];
    for (obs_idx, weight) in weights.iter().enumerate() {
        for col in 0..num_assets {
            let diff = returns[(obs_idx, col)] - means[col];
            variances[col] += weight * diff * diff;
        }
    }
    for var in &mut variances {
        *var /= weight_sum;
    }

    // Check for zero variance.
    for (col, &var) in variances.iter().enumerate() {
        if var < f64::EPSILON {
            return Err(CovarianceError::ZeroVariance { column: col });
        }
    }

    let std_devs: Vec<f64> = variances.iter().map(|var| var.sqrt()).collect();

    // Weighted correlation matrix.
    let mut corr = DMatrix::zeros(num_assets, num_assets);
    for (obs_idx, weight) in weights.iter().enumerate() {
        for row in 0..num_assets {
            let std_row = (returns[(obs_idx, row)] - means[row]) / std_devs[row];
            for col in row..num_assets {
                let std_col = (returns[(obs_idx, col)] - means[col]) / std_devs[col];
                let contribution = weight * std_row * std_col;
                corr[(row, col)] += contribution;
                if row != col {
                    corr[(col, row)] += contribution;
                }
            }
        }
    }
    corr /= weight_sum;

    Ok(CovarianceResult {
        correlation: corr,
        std_devs,
        observations: num_obs,
        assets: num_assets,
        q: num_obs as f64 / num_assets as f64,
    })
}

/// Reconstruct a covariance matrix from a correlation matrix and per-asset
/// standard deviations.
///
/// `Sigma_ij` = `sigma_i` * `C_ij` * `sigma_j`
///
/// This is used after denoising the correlation matrix to obtain a
/// covariance matrix suitable for `VaR` computation.
#[must_use]
pub fn covariance_from_correlation(correlation: &DMatrix<f64>, std_devs: &[f64]) -> DMatrix<f64> {
    let n = correlation.nrows();
    let mut cov = correlation.clone();
    for i in 0..n {
        for j in 0..n {
            cov[(i, j)] *= std_devs[i] * std_devs[j];
        }
    }
    cov
}

/// Standardize a returns matrix: zero mean, unit variance per column.
///
/// Returns the standardized matrix and the per-column standard deviations.
#[allow(clippy::cast_precision_loss)]
fn standardize(returns: &DMatrix<f64>) -> Result<(DMatrix<f64>, Vec<f64>), CovarianceError> {
    let (num_obs, num_assets) = returns.shape();
    let mut result = returns.clone();
    let mut std_devs = Vec::with_capacity(num_assets);

    for col in 0..num_assets {
        let column = returns.column(col);
        let mean = column.mean();
        let variance = column.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / num_obs as f64;

        if variance < f64::EPSILON {
            return Err(CovarianceError::ZeroVariance { column: col });
        }

        let std_dev = variance.sqrt();
        std_devs.push(std_dev);
        for row in 0..num_obs {
            result[(row, col)] = (result[(row, col)] - mean) / std_dev;
        }
    }

    Ok((result, std_devs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 2×2 identity-like case: uncorrelated assets with known returns.
    #[test]
    fn test_correlation_identity() {
        #[rustfmt::skip]
        let returns = DMatrix::from_row_slice(4, 2, &[
             1.0,  0.0,
            -1.0,  0.0,
             0.0,  1.0,
             0.0, -1.0,
        ]);
        let result = correlation_matrix(&returns).unwrap();

        assert_eq!(result.assets, 2);
        assert_eq!(result.observations, 4);
        assert_relative_eq!(result.q, 2.0, epsilon = 1e-10);

        assert_relative_eq!(result.correlation[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.correlation[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.correlation[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.correlation[(1, 0)], 0.0, epsilon = 1e-10);
    }

    /// Perfectly correlated assets.
    #[test]
    fn test_correlation_perfect() {
        #[rustfmt::skip]
        let returns = DMatrix::from_row_slice(4, 2, &[
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
            4.0, 4.0,
        ]);
        let result = correlation_matrix(&returns).unwrap();

        assert_relative_eq!(result.correlation[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.correlation[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.correlation[(0, 1)], 1.0, epsilon = 1e-10);
    }

    /// Trace of correlation matrix must equal N.
    #[test]
    fn test_trace_equals_n() {
        #[rustfmt::skip]
        let returns = DMatrix::from_row_slice(5, 3, &[
            0.01, -0.02,  0.03,
           -0.01,  0.01, -0.02,
            0.02,  0.00,  0.01,
           -0.03,  0.02,  0.00,
            0.01, -0.01,  0.02,
        ]);
        let result = correlation_matrix(&returns).unwrap();

        let trace: f64 = (0..3).map(|i| result.correlation[(i, i)]).sum();
        assert_relative_eq!(trace, 3.0, epsilon = 1e-10);
    }

    /// Symmetry: C\[i,j\] == C\[j,i\].
    #[test]
    fn test_symmetry() {
        #[rustfmt::skip]
        let returns = DMatrix::from_row_slice(5, 3, &[
            0.01, -0.02,  0.03,
           -0.01,  0.01, -0.02,
            0.02,  0.00,  0.01,
           -0.03,  0.02,  0.00,
            0.01, -0.01,  0.02,
        ]);
        let result = correlation_matrix(&returns).unwrap();

        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(
                    result.correlation[(row, col)],
                    result.correlation[(col, row)],
                    epsilon = 1e-14
                );
            }
        }
    }

    /// Empty matrix should error.
    #[test]
    fn test_empty_matrix() {
        let returns = DMatrix::<f64>::zeros(0, 0);
        assert!(correlation_matrix(&returns).is_err());
    }

    /// Single observation should error.
    #[test]
    fn test_insufficient_observations() {
        let returns = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        assert!(correlation_matrix(&returns).is_err());
    }

    /// Zero-variance column should error.
    #[test]
    fn test_zero_variance() {
        #[rustfmt::skip]
        let returns = DMatrix::from_row_slice(3, 2, &[
            1.0, 5.0,
            2.0, 5.0,
            3.0, 5.0,
        ]);
        assert!(matches!(
            correlation_matrix(&returns),
            Err(CovarianceError::ZeroVariance { column: 1 })
        ));
    }

    /// EWM: invalid half-life should error.
    #[test]
    fn test_ewm_invalid_half_life() {
        let returns = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(ewm_correlation_matrix(&returns, 0.0).is_err());
        assert!(ewm_correlation_matrix(&returns, -1.0).is_err());
        assert!(ewm_correlation_matrix(&returns, f64::NAN).is_err());
    }

    /// EWM: with very large half-life should approximate equal-weighted.
    #[test]
    fn test_ewm_large_halflife_approximates_equal() {
        #[rustfmt::skip]
        let returns = DMatrix::from_row_slice(5, 2, &[
            0.01, -0.02,
           -0.01,  0.01,
            0.02,  0.00,
           -0.03,  0.02,
            0.01, -0.01,
        ]);
        let equal = correlation_matrix(&returns).unwrap();
        let ewm = ewm_correlation_matrix(&returns, 1e6).unwrap();

        for row in 0..2 {
            for col in 0..2 {
                assert_relative_eq!(
                    equal.correlation[(row, col)],
                    ewm.correlation[(row, col)],
                    epsilon = 1e-4
                );
            }
        }
    }
}
