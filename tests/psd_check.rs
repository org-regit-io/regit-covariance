// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! After denoising, the correlation matrix must be positive semi-definite.
//! All eigenvalues must be >= 0 (or >= -epsilon for numerical tolerance).

use nalgebra::DMatrix;
use regit_covariance::math::denoise::{DenoiseMethod, denoise};
use regit_covariance::math::eigen::eigendecompose;
use regit_covariance::math::marchenko_pastur::fit_sigma_sq;
use regit_covariance::math::sample_covariance::correlation_matrix;

/// Deterministic LCG-based synthetic returns generator.
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

/// Helper: run the full denoise pipeline.
fn denoise_pipeline(
    num_obs: usize,
    num_assets: usize,
    seed: u64,
    method: DenoiseMethod,
) -> regit_covariance::math::denoise::DenoiseResult {
    let returns = synthetic_returns(num_obs, num_assets, seed);
    let cov = correlation_matrix(&returns).unwrap();
    let eigen = eigendecompose(&cov.correlation).unwrap();
    let mp_fit = fit_sigma_sq(&eigen.eigenvalues, cov.q).unwrap();
    denoise(&eigen, &mp_fit, method)
}

/// Assert all eigenvalues are >= -epsilon (PSD within numerical tolerance).
fn assert_psd(result: &regit_covariance::math::denoise::DenoiseResult, label: &str) {
    let epsilon = 1e-10;
    for idx in 0..result.eigenvalues.len() {
        assert!(
            result.eigenvalues[idx] >= -epsilon,
            "{}: eigenvalue[{}] = {} is negative (below -epsilon={})",
            label,
            idx,
            result.eigenvalues[idx],
            epsilon
        );
    }
}

// ── Constant method, various q ratios ─────────────────────────────────

#[test]
fn psd_constant_q_low() {
    // q ~ 3.3 (low ratio: 10 assets, 33 observations)
    let result = denoise_pipeline(33, 10, 42, DenoiseMethod::Constant);
    assert_psd(&result, "Constant q~3.3");
}

#[test]
fn psd_constant_q_medium() {
    // q ~ 10 (50 observations, 5 assets)
    let result = denoise_pipeline(50, 5, 42, DenoiseMethod::Constant);
    assert_psd(&result, "Constant q~10");
}

#[test]
fn psd_constant_q_high() {
    // q ~ 50 (500 observations, 10 assets)
    let result = denoise_pipeline(500, 10, 42, DenoiseMethod::Constant);
    assert_psd(&result, "Constant q~50");
}

#[test]
fn psd_constant_many_assets() {
    // 20 assets, 200 observations (q=10)
    let result = denoise_pipeline(200, 20, 123, DenoiseMethod::Constant);
    assert_psd(&result, "Constant 20 assets");
}

#[test]
fn psd_constant_few_assets() {
    // 3 assets, 100 observations (q~33)
    let result = denoise_pipeline(100, 3, 999, DenoiseMethod::Constant);
    assert_psd(&result, "Constant 3 assets");
}

// ── Target method, various q ratios ───────────────────────────────────

#[test]
fn psd_target_q_low() {
    let result = denoise_pipeline(33, 10, 42, DenoiseMethod::Target);
    assert_psd(&result, "Target q~3.3");
}

#[test]
fn psd_target_q_medium() {
    let result = denoise_pipeline(50, 5, 42, DenoiseMethod::Target);
    assert_psd(&result, "Target q~10");
}

#[test]
fn psd_target_q_high() {
    let result = denoise_pipeline(500, 10, 42, DenoiseMethod::Target);
    assert_psd(&result, "Target q~50");
}

#[test]
fn psd_target_many_assets() {
    let result = denoise_pipeline(200, 20, 123, DenoiseMethod::Target);
    assert_psd(&result, "Target 20 assets");
}

#[test]
fn psd_target_few_assets() {
    let result = denoise_pipeline(100, 3, 999, DenoiseMethod::Target);
    assert_psd(&result, "Target 3 assets");
}

// ── Multiple seeds to catch edge cases ────────────────────────────────

#[test]
fn psd_constant_multiple_seeds() {
    for seed in [1, 7, 42, 137, 256, 1024, 9999] {
        let result = denoise_pipeline(100, 8, seed, DenoiseMethod::Constant);
        assert_psd(&result, &format!("Constant seed={seed}"));
    }
}

#[test]
fn psd_target_multiple_seeds() {
    for seed in [1, 7, 42, 137, 256, 1024, 9999] {
        let result = denoise_pipeline(100, 8, seed, DenoiseMethod::Target);
        assert_psd(&result, &format!("Target seed={seed}"));
    }
}
