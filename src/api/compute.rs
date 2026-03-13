// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Computation pipeline — supports both live (Yahoo Finance) and synthetic data.
//!
//! In live mode, fetches prices for given tickers, computes log returns,
//! and runs the full denoising pipeline. In synthetic mode, generates
//! random correlated returns for demonstration.

use nalgebra::{DMatrix, DVector};

use crate::data::provider::yahoo;
use crate::data::returns::log_returns;
use crate::math::condition;
use crate::math::denoise::{DenoiseMethod, denoise, renormalize_to_correlation};
use crate::math::eigen::eigendecompose;
use crate::math::marchenko_pastur::fit_sigma_sq;
use crate::math::sample_covariance::{correlation_matrix, covariance_from_correlation};
use crate::math::sri::{self, DivergenceFlag};
use crate::math::var;

use super::state::{
    ComputationResult, ComputeRequest, ConditionResponse, MpFitResponse, RiskMetricsResponse,
    SseEvent,
};

/// Run the full computation pipeline.
///
/// If `request.tickers` is non-empty, fetches real prices from Yahoo Finance.
/// Otherwise, generates synthetic data.
///
/// # Errors
///
/// Returns an error string if price fetching or pipeline computation fails.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub async fn run_pipeline(
    request: &ComputeRequest,
    computation_id: &str,
) -> Result<(ComputationResult, Vec<SseEvent>), String> {
    let mut events = Vec::new();
    let live_mode = !request.tickers.is_empty();

    // Input validation: cap number of tickers and reject malformed input.
    let max_tickers = 100;
    if request.tickers.len() > max_tickers {
        return Err(format!(
            "Too many tickers: {} (max {max_tickers})",
            request.tickers.len()
        ));
    }
    for t in &request.tickers {
        if !t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
        {
            return Err(format!("Invalid ticker or ISIN: {t}"));
        }
    }

    let (returns, num_assets, num_observations, tickers, portfolio_weights, currencies) =
        if live_mode {
            // ── Live mode: fetch real prices ────────────────────────────

            // Auto-resolve any ISINs to Yahoo tickers before fetching prices.
            let mut resolved_tickers = Vec::with_capacity(request.tickers.len());
            for t in &request.tickers {
                if is_isin(t) {
                    events.push(SseEvent {
                        event: "status".into(),
                        data: serde_json::json!({
                            "message": format!("Resolving ISIN {t} to Yahoo ticker"),
                            "stage": 1, "total_stages": 7
                        }),
                    });
                    let ticker = yahoo::resolve_isin(t)
                        .await
                        .map_err(|e| format!("ISIN resolution failed for {t}: {e}"))?;
                    resolved_tickers.push(ticker);
                } else {
                    resolved_tickers.push(t.clone());
                }
            }

            events.push(SseEvent {
            event: "status".into(),
            data: serde_json::json!({
                "message": format!("Fetching prices for {} tickers from Yahoo Finance", resolved_tickers.len()),
                "stage": 1, "total_stages": 7
            }),
        });

            let ticker_data = yahoo::fetch_prices(&resolved_tickers, request.period_days)
                .await
                .map_err(|e| format!("Price fetch failed: {e}"))?;

            let currencies: Vec<String> =
                ticker_data.iter().map(|td| td.currency.clone()).collect();

            events.push(SseEvent {
                event: "status".into(),
                data: serde_json::json!({
                    "message": "Aligning prices to common dates",
                    "stage": 1, "total_stages": 7
                }),
            });

            let (_dates, price_matrix) = yahoo::align_prices(&ticker_data);

            if price_matrix.nrows() < 3 {
                return Err(format!(
                    "Only {} common trading days found — need at least 3",
                    price_matrix.nrows()
                ));
            }

            let returns = log_returns(&price_matrix)
                .map_err(|e| format!("Log returns computation failed: {e}"))?;

            let n = returns.ncols();
            let t = returns.nrows();
            let tickers = resolved_tickers;

            // Portfolio weights: use provided or default to equal-weight.
            let weights = if request.weights.len() == n {
                DVector::from_vec(request.weights.clone())
            } else {
                DVector::from_element(n, 1.0 / n as f64)
            };

            (returns, n, t, tickers, weights, currencies)
        } else {
            // ── Synthetic mode ──────────────────────────────────────────
            events.push(SseEvent {
                event: "status".into(),
                data: serde_json::json!({
                    "message": "Generating synthetic returns",
                    "stage": 1, "total_stages": 7
                }),
            });

            let returns = generate_synthetic_returns(
                request.num_observations,
                request.num_assets,
                request.seed,
            );
            let n = request.num_assets;
            let t = request.num_observations;
            let tickers: Vec<String> = (0..n).map(|i| format!("Asset{i}")).collect();
            let weights = DVector::from_element(n, 1.0 / n as f64);
            let currencies = vec!["SYN".to_string(); n];

            (returns, n, t, tickers, weights, currencies)
        };

    // Stage 2: Compute sample correlation.
    events.push(SseEvent {
        event: "status".into(),
        data: serde_json::json!({
            "message": format!("Computing {num_assets}x{num_assets} correlation matrix from {num_observations} observations"),
            "stage": 2, "total_stages": 7
        }),
    });

    let cov_result =
        correlation_matrix(&returns).map_err(|e| format!("Correlation computation failed: {e}"))?;

    // Stage 3: Eigendecomposition.
    let eigen = eigendecompose(&cov_result.correlation)
        .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

    let eigenvalues_raw: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    events.push(SseEvent {
        event: "eigenvalues_raw".into(),
        data: serde_json::json!({
            "eigenvalues": eigenvalues_raw,
            "num_assets": num_assets,
            "num_observations": num_observations,
            "q": cov_result.q,
            "tickers": tickers,
            "currencies": currencies,
        }),
    });

    // Stage 4: Marchenko-Pastur fit.
    let mp_fit = fit_sigma_sq(&eigen.eigenvalues, cov_result.q)
        .map_err(|e| format!("MP fit failed: {e}"))?;

    let mp_response = MpFitResponse {
        sigma_sq: mp_fit.sigma_sq,
        lambda_plus: mp_fit.lambda_plus,
        lambda_minus: mp_fit.lambda_minus,
        signal_count: mp_fit.signal_count,
        noise_count: mp_fit.noise_count,
    };
    events.push(SseEvent {
        event: "mp_fit".into(),
        data: serde_json::to_value(&mp_response).unwrap_or_default(),
    });

    // Stage 5: Denoise + re-normalise to correlation (unit diagonal).
    let denoised = denoise(&eigen, &mp_fit, DenoiseMethod::Target);
    let denoised_corr = renormalize_to_correlation(&denoised.matrix);
    let eigenvalues_denoised: Vec<f64> = denoised.eigenvalues.iter().copied().collect();

    // Flatten correlation matrices for frontend heatmaps (row-major).
    let corr_raw_flat: Vec<f64> = cov_result.correlation.iter().copied().collect();
    let corr_denoised_flat: Vec<f64> = denoised_corr.iter().copied().collect();

    events.push(SseEvent {
        event: "eigenvalues_denoised".into(),
        data: serde_json::json!({
            "eigenvalues": eigenvalues_denoised,
            "method": "target",
            "trace": denoised.trace,
            "corr_raw": corr_raw_flat,
            "corr_denoised": corr_denoised_flat,
            "tickers": tickers,
            "n": num_assets,
        }),
    });

    // Stage 6: Condition number.
    let eigen_denoised = eigendecompose(&denoised.matrix)
        .map_err(|e| format!("Denoised eigendecomposition failed: {e}"))?;

    let cond_improvement = condition::compare(&eigen, &eigen_denoised);
    let cond_response = ConditionResponse {
        before: cond_improvement.before.condition_number,
        after: cond_improvement.after.condition_number,
        improvement_factor: cond_improvement.improvement_factor,
        health: format!("{:?}", cond_improvement.after.health),
    };
    events.push(SseEvent {
        event: "condition_number".into(),
        data: serde_json::to_value(&cond_response).unwrap_or_default(),
    });

    // Stage 7: Risk metrics.
    // Convert correlation matrices back to covariance for VaR computation.
    // Annualize: Σ_annual = Σ_daily × 252 (PRIIPs SRI thresholds are annual).
    let annualization_factor = 252.0;
    let mut cov_raw = covariance_from_correlation(&cov_result.correlation, &cov_result.std_devs);
    cov_raw *= annualization_factor;
    let mut cov_denoised = covariance_from_correlation(&denoised_corr, &cov_result.std_devs);
    cov_denoised *= annualization_factor;
    let expected_returns = DVector::zeros(num_assets);

    // VaR from raw covariance (= "prescribed" proxy).
    let var_raw = var::parametric_var(&portfolio_weights, &expected_returns, &cov_raw, 0.975)
        .map_err(|e| format!("VaR (raw) failed: {e}"))?;

    // VaR from denoised covariance (= "kernel" estimate).
    let var_denoised =
        var::parametric_var(&portfolio_weights, &expected_returns, &cov_denoised, 0.975)
            .map_err(|e| format!("VaR (denoised) failed: {e}"))?;

    // Compute portfolio return series for skewness/kurtosis.
    let portfolio_returns: Vec<f64> = (0..returns.nrows())
        .map(|t| {
            (0..num_assets)
                .map(|i| portfolio_weights[i] * returns[(t, i)])
                .sum::<f64>()
        })
        .collect();
    let n_obs = portfolio_returns.len() as f64;
    let port_mean: f64 = portfolio_returns.iter().sum::<f64>() / n_obs;
    let port_std = (portfolio_returns
        .iter()
        .map(|r| (r - port_mean).powi(2))
        .sum::<f64>()
        / n_obs)
        .sqrt();
    let (skewness, excess_kurtosis) = if port_std > f64::EPSILON {
        let s = portfolio_returns
            .iter()
            .map(|r| ((r - port_mean) / port_std).powi(3))
            .sum::<f64>()
            / n_obs;
        let k = portfolio_returns
            .iter()
            .map(|r| ((r - port_mean) / port_std).powi(4))
            .sum::<f64>()
            / n_obs
            - 3.0;
        (s, k)
    } else {
        (0.0, 0.0)
    };

    // Cornish-Fisher VaR (adjusted for skewness and kurtosis).
    let var_cf = var::cornish_fisher_var(
        &portfolio_weights,
        &expected_returns,
        &cov_denoised,
        0.975,
        skewness,
        excess_kurtosis,
    )
    .map_err(|e| format!("VaR (CF) failed: {e}"))?;

    // SRI classification.
    let z_975 = 1.96;
    let prescribed_vev = sri::var_equivalent_volatility(var_raw.var, z_975);
    let kernel_vev = sri::var_equivalent_volatility(var_denoised.var, z_975);

    // If user provided a prescribed SRI from the KID, use it directly.
    let (prescribed_sri, kernel_sri, divergence_flag) =
        if let Some(kid_sri) = request.prescribed_sri {
            let kernel = sri::classify_mrm(kernel_vev)
                .map_err(|e| format!("SRI classification failed: {e}"))?;
            let diff = kid_sri.abs_diff(kernel.mrm);
            let flag = match diff {
                0 => "green",
                1 => "yellow",
                _ => "red",
            };
            (kid_sri, kernel.mrm, flag.to_string())
        } else {
            let divergence =
                sri::divergence_report(prescribed_vev, kernel_vev).map_err(|e| format!("{e}"))?;
            let flag = match divergence.flag {
                DivergenceFlag::Green => "green",
                DivergenceFlag::Yellow => "yellow",
                DivergenceFlag::Red => "red",
            };
            (
                divergence.prescribed_sri,
                divergence.kernel_sri,
                flag.to_string(),
            )
        };

    let risk_response = RiskMetricsResponse {
        var_parametric: var_denoised.var,
        var_cornish_fisher: var_cf.var,
        portfolio_volatility: var_denoised.portfolio_volatility,
        prescribed_sri,
        kernel_sri,
        divergence_flag,
    };

    // Per-asset SRI breakdown (individual annualized volatility → VEV → MRM).
    let per_asset_sri: Vec<serde_json::Value> = (0..num_assets)
        .map(|i| {
            let asset_vol = cov_denoised[(i, i)].sqrt();
            let asset_vev = sri::var_equivalent_volatility(z_975 * asset_vol, z_975);
            let asset_mrm = sri::classify_mrm(asset_vev).map(|r| r.mrm).unwrap_or(0);
            serde_json::json!({
                "ticker": tickers[i],
                "volatility": asset_vol,
                "sri": asset_mrm,
            })
        })
        .collect();

    let mut risk_data = serde_json::to_value(&risk_response).unwrap_or_default();
    risk_data["per_asset"] = serde_json::Value::Array(per_asset_sri);
    risk_data["skewness"] = serde_json::json!(skewness);
    risk_data["excess_kurtosis"] = serde_json::json!(excess_kurtosis);
    events.push(SseEvent {
        event: "risk_metrics".into(),
        data: risk_data,
    });

    // Complete.
    let result = ComputationResult {
        id: computation_id.to_string(),
        status: "complete".into(),
        num_assets,
        num_observations,
        eigenvalues_raw: Some(eigenvalues_raw),
        eigenvalues_denoised: Some(eigenvalues_denoised),
        mp_fit: Some(mp_response),
        condition: Some(cond_response),
        risk_metrics: Some(risk_response),
    };

    events.push(SseEvent {
        event: "complete".into(),
        data: serde_json::to_value(&result).unwrap_or_default(),
    });

    Ok((result, events))
}

/// Generate a T × N synthetic returns matrix using a deterministic LCG.
///
/// Produces realistic-looking correlated returns by embedding a few
/// common factors plus idiosyncratic noise.
#[allow(clippy::cast_precision_loss)]
fn generate_synthetic_returns(num_obs: usize, num_assets: usize, seed: u64) -> DMatrix<f64> {
    let mut state = seed;
    let mut next = || -> f64 {
        // Linear congruential generator (Numerical Recipes).
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        (u - 0.5) * 3.46
    };

    let num_factors = 3.min(num_assets);
    let mut factor_loadings = DMatrix::zeros(num_assets, num_factors);
    for col in 0..num_factors {
        for row in 0..num_assets {
            factor_loadings[(row, col)] = next() * 0.3;
        }
    }

    let mut returns = DMatrix::zeros(num_obs, num_assets);
    for obs in 0..num_obs {
        let mut factors = vec![0.0; num_factors];
        for f in &mut factors {
            *f = next() * 0.01;
        }
        for asset in 0..num_assets {
            let mut r = 0.0;
            for (fidx, &fret) in factors.iter().enumerate() {
                r += factor_loadings[(asset, fidx)] * fret;
            }
            r += next() * 0.02;
            returns[(obs, asset)] = r;
        }
    }

    returns
}

/// Check if a string looks like an ISIN (2 uppercase letters + 10 alphanumeric).
fn is_isin(s: &str) -> bool {
    s.len() == 12
        && s[..2].chars().all(|c| c.is_ascii_uppercase())
        && s[2..].chars().all(|c| c.is_ascii_alphanumeric())
}
