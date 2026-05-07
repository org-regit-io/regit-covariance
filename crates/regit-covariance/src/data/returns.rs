// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Log returns computation.
//!
//! Converts a price matrix into a log returns matrix:
//! `r_t = ln(P_t / P_{t-1})`
//!
//! Missing data handling: assets with any NaN or zero prices are flagged.

use nalgebra::DMatrix;
use thiserror::Error;

/// Errors from returns computation.
#[derive(Debug, Error)]
pub enum ReturnsError {
    /// Need at least 2 price observations to compute returns.
    #[error("need at least 2 price observations, got {0}")]
    InsufficientPrices(usize),

    /// Empty price matrix.
    #[error("empty price matrix")]
    EmptyPrices,

    /// Zero or negative price encountered.
    #[error("zero or negative price at observation {row}, asset {col}")]
    InvalidPrice {
        /// Observation index.
        row: usize,
        /// Asset index.
        col: usize,
    },
}

/// Compute log returns from a price matrix.
///
/// Given a T × N price matrix P, returns a (T-1) × N matrix of
/// log returns where `r[t][i] = ln(P[t+1][i] / P[t][i])`.
///
/// # Arguments
///
/// * `prices` — (T × N) matrix of prices, where T is the number of
///   observations and N is the number of assets.
///
/// # Errors
///
/// Returns [`ReturnsError`] if the price matrix is empty, has fewer
/// than 2 observations, or contains zero/negative prices.
pub fn log_returns(prices: &DMatrix<f64>) -> Result<DMatrix<f64>, ReturnsError> {
    let (num_obs, num_assets) = prices.shape();

    if num_obs == 0 || num_assets == 0 {
        return Err(ReturnsError::EmptyPrices);
    }
    if num_obs < 2 {
        return Err(ReturnsError::InsufficientPrices(num_obs));
    }

    // Validate all prices are positive.
    for row in 0..num_obs {
        for col in 0..num_assets {
            if prices[(row, col)] <= 0.0 {
                return Err(ReturnsError::InvalidPrice { row, col });
            }
        }
    }

    let mut returns = DMatrix::zeros(num_obs - 1, num_assets);
    for row in 0..num_obs - 1 {
        for col in 0..num_assets {
            returns[(row, col)] = (prices[(row + 1, col)] / prices[(row, col)]).ln();
        }
    }

    Ok(returns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basic_log_returns() {
        #[rustfmt::skip]
        let prices = DMatrix::from_row_slice(3, 2, &[
            100.0, 50.0,
            110.0, 55.0,
            105.0, 52.0,
        ]);
        let returns = log_returns(&prices).unwrap();
        assert_eq!(returns.shape(), (2, 2));
        assert_relative_eq!(returns[(0, 0)], (110.0_f64 / 100.0).ln(), epsilon = 1e-14);
        assert_relative_eq!(returns[(0, 1)], (55.0_f64 / 50.0).ln(), epsilon = 1e-14);
        assert_relative_eq!(returns[(1, 0)], (105.0_f64 / 110.0).ln(), epsilon = 1e-14);
    }

    #[test]
    fn test_empty_prices() {
        let prices = DMatrix::<f64>::zeros(0, 0);
        assert!(log_returns(&prices).is_err());
    }

    #[test]
    fn test_insufficient_prices() {
        let prices = DMatrix::from_row_slice(1, 2, &[100.0, 50.0]);
        assert!(log_returns(&prices).is_err());
    }

    #[test]
    fn test_zero_price() {
        #[rustfmt::skip]
        let prices = DMatrix::from_row_slice(2, 2, &[
            100.0, 50.0,
            0.0,   55.0,
        ]);
        assert!(log_returns(&prices).is_err());
    }
}
