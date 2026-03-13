// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Quickstart: end-to-end denoising pipeline on synthetic data.
//!
//! ```sh
//! cargo run --example quickstart
//! ```

use nalgebra::{DMatrix, DVector};

use regit_covariance::math::condition;
use regit_covariance::math::denoise::{DenoiseMethod, denoise};
use regit_covariance::math::eigen::eigendecompose;
use regit_covariance::math::marchenko_pastur::fit_sigma_sq;
use regit_covariance::math::sample_covariance::correlation_matrix;
use regit_covariance::math::var;

fn main() {
    // 1. Generate a simple synthetic returns matrix (T=500, N=20).
    let (num_obs, num_assets) = (500, 20);
    let returns = synthetic_returns(num_obs, num_assets);

    // 2. Compute sample correlation matrix.
    let cov = correlation_matrix(&returns).expect("correlation matrix");
    println!(
        "Sample correlation: {}×{}, q = {:.2}",
        num_assets, num_assets, cov.q
    );

    // 3. Eigendecompose.
    let eigen = eigendecompose(&cov.correlation).expect("eigendecomposition");
    println!(
        "Eigenvalue range: {:.4} .. {:.4}",
        eigen.eigenvalues[eigen.eigenvalues.len() - 1],
        eigen.eigenvalues[0],
    );

    // 4. Fit Marchenko-Pastur noise floor.
    let mp = fit_sigma_sq(&eigen.eigenvalues, cov.q).expect("MP fit");
    println!(
        "MP fit: σ²={:.4}, λ₊={:.4}, signal={}, noise={}",
        mp.sigma_sq, mp.lambda_plus, mp.signal_count, mp.noise_count,
    );

    // 5. Denoise via target shrinkage.
    let denoised = denoise(&eigen, &mp, DenoiseMethod::Target);
    println!(
        "Denoised trace: {:.4} (should ≈ {num_assets})",
        denoised.trace
    );

    // 6. Compare condition numbers.
    let eigen_d = eigendecompose(&denoised.matrix).expect("denoised eigen");
    let cond = condition::compare(&eigen, &eigen_d);
    println!(
        "Condition number: {:.1} → {:.1} ({:.1}× improvement)",
        cond.before.condition_number, cond.after.condition_number, cond.improvement_factor,
    );

    // 7. Parametric VaR (equal-weight portfolio, annualized).
    let weights = DVector::from_element(num_assets, 1.0 / num_assets as f64);
    let mu = DVector::zeros(num_assets);
    let mut cov_annual = regit_covariance::math::sample_covariance::covariance_from_correlation(
        &denoised.matrix,
        &cov.std_devs,
    );
    cov_annual *= 252.0;

    let var_result = var::parametric_var(&weights, &mu, &cov_annual, 0.975).expect("VaR");
    println!(
        "Annual VaR(97.5%): {:.2}%, portfolio vol: {:.2}%",
        var_result.var * 100.0,
        var_result.portfolio_volatility * 100.0,
    );
}

/// Deterministic synthetic returns for demonstration.
fn synthetic_returns(t: usize, n: usize) -> DMatrix<f64> {
    let mut state: u64 = 42;
    let mut next = || -> f64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        (u - 0.5) * 0.04 // daily-scale returns
    };

    DMatrix::from_fn(t, n, |_, _| next())
}
