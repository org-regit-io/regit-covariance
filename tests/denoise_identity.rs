// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! For pure noise (uncorrelated returns), denoising should produce
//! something close to the identity matrix.

use approx::assert_relative_eq;
use nalgebra::DMatrix;
use regit_covariance::math::denoise::{DenoiseMethod, denoise};
use regit_covariance::math::eigen::eigendecompose;
use regit_covariance::math::marchenko_pastur::fit_sigma_sq;
use regit_covariance::math::sample_covariance::correlation_matrix;

/// Deterministic LCG-based synthetic returns generator.
/// Each column is independently generated (uncorrelated by construction).
fn uncorrelated_returns(num_obs: usize, num_assets: usize, seed: u64) -> DMatrix<f64> {
    let mut data = vec![0.0_f64; num_obs * num_assets];
    let mut state = seed;
    for val in &mut data {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *val = ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
    }
    DMatrix::from_row_slice(num_obs, num_assets, &data)
}

/// Helper: denoise pipeline.
fn denoise_from_returns(
    returns: &DMatrix<f64>,
    method: DenoiseMethod,
) -> regit_covariance::math::denoise::DenoiseResult {
    let cov = correlation_matrix(returns).unwrap();
    let eigen = eigendecompose(&cov.correlation).unwrap();
    let mp_fit = fit_sigma_sq(&eigen.eigenvalues, cov.q).unwrap();
    denoise(&eigen, &mp_fit, method)
}

// ── Constant method ───────────────────────────────────────────────────

#[test]
fn identity_diagonal_constant_5_assets() {
    let returns = uncorrelated_returns(500, 5, 42);
    let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

    for i in 0..5 {
        assert_relative_eq!(result.matrix[(i, i)], 1.0, epsilon = 0.15);
    }
}

#[test]
fn identity_offdiag_constant_5_assets() {
    let returns = uncorrelated_returns(500, 5, 42);
    let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

    for row in 0..5 {
        for col in 0..5 {
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

#[test]
fn identity_diagonal_constant_10_assets() {
    let returns = uncorrelated_returns(500, 10, 7);
    let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

    for i in 0..10 {
        assert_relative_eq!(result.matrix[(i, i)], 1.0, epsilon = 0.15);
    }
}

#[test]
fn identity_offdiag_constant_10_assets() {
    let returns = uncorrelated_returns(500, 10, 7);
    let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

    for row in 0..10 {
        for col in 0..10 {
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

// ── Target method ─────────────────────────────────────────────────────

#[test]
fn identity_diagonal_target_5_assets() {
    let returns = uncorrelated_returns(500, 5, 42);
    let result = denoise_from_returns(&returns, DenoiseMethod::Target);

    for i in 0..5 {
        assert_relative_eq!(result.matrix[(i, i)], 1.0, epsilon = 0.15);
    }
}

#[test]
fn identity_offdiag_target_5_assets() {
    let returns = uncorrelated_returns(500, 5, 42);
    let result = denoise_from_returns(&returns, DenoiseMethod::Target);

    for row in 0..5 {
        for col in 0..5 {
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

#[test]
fn identity_diagonal_target_10_assets() {
    let returns = uncorrelated_returns(500, 10, 7);
    let result = denoise_from_returns(&returns, DenoiseMethod::Target);

    for i in 0..10 {
        assert_relative_eq!(result.matrix[(i, i)], 1.0, epsilon = 0.15);
    }
}

#[test]
fn identity_offdiag_target_10_assets() {
    let returns = uncorrelated_returns(500, 10, 7);
    let result = denoise_from_returns(&returns, DenoiseMethod::Target);

    for row in 0..10 {
        for col in 0..10 {
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

// ── High T/N ratio should produce near-perfect identity ───────────────

#[test]
fn identity_high_ratio_constant() {
    // With T/N = 200, sample correlation is already very close to identity,
    // and denoising should keep it that way.
    let returns = uncorrelated_returns(1000, 5, 42);
    let result = denoise_from_returns(&returns, DenoiseMethod::Constant);

    for i in 0..5 {
        assert_relative_eq!(result.matrix[(i, i)], 1.0, epsilon = 0.1);
    }
    for row in 0..5 {
        for col in 0..5 {
            if row != col {
                assert!(
                    result.matrix[(row, col)].abs() < 0.15,
                    "high-ratio off-diagonal ({},{}) too large: {}",
                    row,
                    col,
                    result.matrix[(row, col)]
                );
            }
        }
    }
}
