// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration test: full pipeline from synthetic returns to risk metrics.

use nalgebra::{DMatrix, DVector};

use regit_covariance::math::condition;
use regit_covariance::math::denoise::{DenoiseMethod, denoise};
use regit_covariance::math::eigen::eigendecompose;
use regit_covariance::math::marchenko_pastur::fit_sigma_sq;
use regit_covariance::math::sample_covariance::{correlation_matrix, covariance_from_correlation};
use regit_covariance::math::sri;
use regit_covariance::math::var;

/// Deterministic synthetic returns.
fn synthetic_returns(num_obs: usize, num_assets: usize, seed: u64) -> DMatrix<f64> {
    let mut data = vec![0.0_f64; num_obs * num_assets];
    let mut state = seed;
    for val in &mut data {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
    }
    DMatrix::from_row_slice(num_obs, num_assets, &data)
}

/// Full pipeline: correlation → eigen → MP → denoise → condition → VaR → SRI.
#[test]
fn full_pipeline_synthetic() {
    let (t, n) = (252, 10);
    let returns = synthetic_returns(t, n, 42);

    // Correlation.
    let cov = correlation_matrix(&returns).unwrap();
    assert_eq!(cov.correlation.nrows(), n);
    assert_eq!(cov.correlation.ncols(), n);

    // Eigen.
    let eigen = eigendecompose(&cov.correlation).unwrap();
    assert_eq!(eigen.eigenvalues.len(), n);
    for i in 1..n {
        assert!(eigen.eigenvalues[i - 1] >= eigen.eigenvalues[i]);
    }

    // MP fit.
    let mp = fit_sigma_sq(&eigen.eigenvalues, cov.q).unwrap();
    assert!(mp.sigma_sq > 0.0);
    assert!(mp.lambda_plus > mp.lambda_minus);
    assert_eq!(mp.signal_count + mp.noise_count, n);

    // Denoise.
    let denoised = denoise(&eigen, &mp, DenoiseMethod::Target);
    assert!((denoised.trace - n as f64).abs() < 1e-8);

    // Condition improvement.
    let eigen_d = eigendecompose(&denoised.matrix).unwrap();
    let cond = condition::compare(&eigen, &eigen_d);
    assert!(
        cond.improvement_factor >= 1.0,
        "denoising should not worsen conditioning"
    );

    // VaR.
    let weights = DVector::from_element(n, 1.0 / n as f64);
    let mu = DVector::zeros(n);
    let mut cov_mat = covariance_from_correlation(&denoised.matrix, &cov.std_devs);
    cov_mat *= 252.0;

    let var_result = var::parametric_var(&weights, &mu, &cov_mat, 0.975).unwrap();
    assert!(var_result.var > 0.0);
    assert!(var_result.portfolio_volatility > 0.0);

    // SRI classification.
    let z = 1.96;
    let vev = sri::var_equivalent_volatility(var_result.var, z);
    let mrm = sri::classify_mrm(vev).unwrap();
    assert!(
        (1..=7).contains(&mrm.mrm),
        "MRM must be 1-7, got {}",
        mrm.mrm
    );
}

/// Condition number should improve (or stay same) after denoising.
#[test]
fn condition_improves_various_sizes() {
    for (t, n) in [(100, 5), (252, 10), (500, 20)] {
        let returns = synthetic_returns(t, n, 42);
        let cov = correlation_matrix(&returns).unwrap();
        let eigen = eigendecompose(&cov.correlation).unwrap();
        let mp = fit_sigma_sq(&eigen.eigenvalues, cov.q).unwrap();
        let denoised = denoise(&eigen, &mp, DenoiseMethod::Target);
        let eigen_d = eigendecompose(&denoised.matrix).unwrap();
        let cond = condition::compare(&eigen, &eigen_d);
        assert!(
            cond.improvement_factor >= 0.99,
            "T={t}, N={n}: improvement={} < 1.0",
            cond.improvement_factor
        );
    }
}

/// VaR scales with portfolio size (more assets → diversification).
#[test]
fn diversification_reduces_var() {
    let returns = synthetic_returns(500, 20, 42);
    let cov = correlation_matrix(&returns).unwrap();
    let eigen = eigendecompose(&cov.correlation).unwrap();
    let mp = fit_sigma_sq(&eigen.eigenvalues, cov.q).unwrap();
    let denoised = denoise(&eigen, &mp, DenoiseMethod::Target);
    let mu = DVector::zeros(20);
    let mut cov_mat = covariance_from_correlation(&denoised.matrix, &cov.std_devs);
    cov_mat *= 252.0;

    // Equal-weight across 20 assets.
    let w20 = DVector::from_element(20, 1.0 / 20.0);
    let var20 = var::parametric_var(&w20, &mu, &cov_mat, 0.975).unwrap();

    // Concentrated in 1 asset.
    let mut w1 = DVector::zeros(20);
    w1[0] = 1.0;
    let var1 = var::parametric_var(&w1, &mu, &cov_mat, 0.975).unwrap();

    assert!(
        var20.var < var1.var,
        "diversified VaR ({}) should be less than concentrated ({})",
        var20.var,
        var1.var
    );
}
