// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Property-based tests for trace preservation and symmetry after denoising.

use nalgebra::DMatrix;
use proptest::prelude::*;
use regit_covariance::math::denoise::{DenoiseMethod, denoise};
use regit_covariance::math::eigen::eigendecompose;
use regit_covariance::math::marchenko_pastur::fit_sigma_sq;
use regit_covariance::math::sample_covariance::correlation_matrix;

/// Deterministic LCG-based synthetic returns generator.
///
/// Generates a (T x N) returns matrix using a linear congruential generator
/// seeded by the given value. Returns are centered around zero with small
/// magnitude, mimicking daily log returns.
fn synthetic_returns(num_obs: usize, num_assets: usize, seed: u64) -> DMatrix<f64> {
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

/// Helper: run the full denoise pipeline from returns and return the result.
fn denoise_from_returns(
    returns: &DMatrix<f64>,
    method: DenoiseMethod,
) -> regit_covariance::math::denoise::DenoiseResult {
    let cov = correlation_matrix(returns).unwrap();
    let eigen = eigendecompose(&cov.correlation).unwrap();
    let mp_fit = fit_sigma_sq(&eigen.eigenvalues, cov.q).unwrap();
    denoise(&eigen, &mp_fit, method)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// After Constant denoising, trace of the correlation matrix must equal N.
    #[test]
    fn trace_equals_n_constant(
        num_assets in 3_usize..=20,
        num_obs in 50_usize..=500,
        seed in 1_u64..10_000,
    ) {
        let returns = synthetic_returns(num_obs, num_assets, seed);
        let result = denoise_from_returns(&returns, DenoiseMethod::Constant);
        let expected_trace = num_assets as f64;
        prop_assert!(
            (result.trace - expected_trace).abs() < 1e-8,
            "Constant: trace={}, expected={}, diff={}",
            result.trace, expected_trace, (result.trace - expected_trace).abs()
        );
    }

    /// After Target denoising, trace of the correlation matrix must equal N.
    #[test]
    fn trace_equals_n_target(
        num_assets in 3_usize..=20,
        num_obs in 50_usize..=500,
        seed in 1_u64..10_000,
    ) {
        let returns = synthetic_returns(num_obs, num_assets, seed);
        let result = denoise_from_returns(&returns, DenoiseMethod::Target);
        let expected_trace = num_assets as f64;
        prop_assert!(
            (result.trace - expected_trace).abs() < 1e-8,
            "Target: trace={}, expected={}, diff={}",
            result.trace, expected_trace, (result.trace - expected_trace).abs()
        );
    }

    /// After denoising, the matrix must be symmetric: M[i,j] == M[j,i].
    #[test]
    fn matrix_is_symmetric_after_denoise(
        num_assets in 3_usize..=15,
        num_obs in 50_usize..=300,
        seed in 1_u64..10_000,
    ) {
        let returns = synthetic_returns(num_obs, num_assets, seed);
        let result = denoise_from_returns(&returns, DenoiseMethod::Constant);
        let n = result.matrix.nrows();
        for row in 0..n {
            for col in (row + 1)..n {
                let diff = (result.matrix[(row, col)] - result.matrix[(col, row)]).abs();
                prop_assert!(
                    diff < 1e-12,
                    "Asymmetry at ({},{}): {} vs {}, diff={}",
                    row, col,
                    result.matrix[(row, col)],
                    result.matrix[(col, row)],
                    diff
                );
            }
        }
    }

    /// After Target denoising, the matrix must also be symmetric.
    #[test]
    fn matrix_is_symmetric_after_target_denoise(
        num_assets in 3_usize..=15,
        num_obs in 50_usize..=300,
        seed in 1_u64..10_000,
    ) {
        let returns = synthetic_returns(num_obs, num_assets, seed);
        let result = denoise_from_returns(&returns, DenoiseMethod::Target);
        let n = result.matrix.nrows();
        for row in 0..n {
            for col in (row + 1)..n {
                let diff = (result.matrix[(row, col)] - result.matrix[(col, row)]).abs();
                prop_assert!(
                    diff < 1e-12,
                    "Asymmetry at ({},{}): {} vs {}, diff={}",
                    row, col,
                    result.matrix[(row, col)],
                    result.matrix[(col, row)],
                    diff
                );
            }
        }
    }
}
