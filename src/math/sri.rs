// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! PRIIPs Summary Risk Indicator (SRI) computation.
//!
//! The SRI is a 1–7 scale risk classification mandated by the PRIIPs
//! Regulation (EU) 1286/2014 and its Delegated Regulation (EU) 2017/653.
//! It combines market risk (MRM) derived from `VaR` with credit risk (CRM).
//!
//! This module implements market risk classification only. Credit risk
//! assessment depends on external credit quality data and is out of scope
//! for the covariance kernel.
//!
//! # Market Risk Measure (MRM) class
//!
//! | MRM | VaR-equivalent volatility (VEV) range |
//! |-----|---------------------------------------|
//! | 1   | VEV < 0.5%                            |
//! | 2   | 0.5% ≤ VEV < 5%                       |
//! | 3   | 5% ≤ VEV < 12%                        |
//! | 4   | 12% ≤ VEV < 20%                       |
//! | 5   | 20% ≤ VEV < 30%                       |
//! | 6   | 30% ≤ VEV < 80%                       |
//! | 7   | VEV ≥ 80%                             |
//!
//! # References
//!
//! - PRIIPs Delegated Regulation (EU) 2017/653, Annex II, Part 1.
//! - PRIIPs RTS, Annex II, Part 1, Sections 3–5.

use thiserror::Error;

/// Errors from SRI computation.
#[derive(Debug, Error)]
pub enum SriError {
    /// Invalid `VaR`-equivalent volatility (negative or NaN).
    #[error("VEV must be non-negative, got {0}")]
    InvalidVev(f64),
}

/// Result of SRI computation.
#[derive(Debug, Clone)]
pub struct SriResult {
    /// Market Risk Measure class (1–7).
    pub mrm: u8,
    /// `VaR`-equivalent volatility used for classification.
    pub vev: f64,
}

/// Result of divergence analysis between prescribed and kernel SRI.
#[derive(Debug, Clone)]
pub struct DivergenceReport {
    /// SRI from prescribed (regulatory) methodology.
    pub prescribed_sri: u8,
    /// SRI from kernel (denoised covariance) methodology.
    pub kernel_sri: u8,
    /// Absolute difference in SRI classes.
    pub sri_difference: u8,
    /// Prescribed `VaR`-equivalent volatility.
    pub prescribed_vev: f64,
    /// Kernel `VaR`-equivalent volatility.
    pub kernel_vev: f64,
    /// Confidence flag based on divergence magnitude.
    pub flag: DivergenceFlag,
}

/// Confidence flag for divergence reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceFlag {
    /// SRI classes agree — no concern.
    Green,
    /// SRI classes differ by 1 — worth monitoring.
    Yellow,
    /// SRI classes differ by ≥ 2 — investigate.
    Red,
}

/// Compute the `VaR`-equivalent volatility (VEV) from annualized portfolio `VaR`.
///
/// `VEV = VaR / z_α` (i.e., the implied annualized volatility).
/// For the standard PRIIPs confidence level (97.5%), z ≈ 1.96.
///
/// # Arguments
///
/// * `var` — Portfolio `VaR` at 97.5% confidence (annualized, positive).
/// * `z_alpha` — Normal quantile at the confidence level (≈ 1.96 for 97.5%).
#[must_use]
pub fn var_equivalent_volatility(var: f64, z_alpha: f64) -> f64 {
    if z_alpha.abs() < f64::EPSILON {
        return 0.0;
    }
    var / z_alpha
}

/// Classify `VaR`-equivalent volatility into a Market Risk Measure (MRM) class.
///
/// Thresholds per PRIIPs Delegated Regulation (EU) 2017/653, Annex II.
///
/// # Errors
///
/// Returns [`SriError`] if VEV is negative or NaN.
pub fn classify_mrm(vev: f64) -> Result<SriResult, SriError> {
    if vev.is_nan() || vev < 0.0 {
        return Err(SriError::InvalidVev(vev));
    }

    // VEV thresholds in percentage form.
    // The input VEV is a decimal (e.g., 0.15 for 15%), convert to percentage.
    let vev_pct = vev * 100.0;

    let mrm = if vev_pct < 0.5 {
        1
    } else if vev_pct < 5.0 {
        2
    } else if vev_pct < 12.0 {
        3
    } else if vev_pct < 20.0 {
        4
    } else if vev_pct < 30.0 {
        5
    } else if vev_pct < 80.0 {
        6
    } else {
        7
    };

    Ok(SriResult { mrm, vev })
}

/// Compare prescribed and kernel SRI to produce a divergence report.
///
/// # Arguments
///
/// * `prescribed_vev` — VEV from prescribed (regulatory) methodology.
/// * `kernel_vev` — VEV from kernel (denoised covariance) methodology.
///
/// # Errors
///
/// Returns [`SriError`] if either VEV is invalid.
pub fn divergence_report(
    prescribed_vev: f64,
    kernel_vev: f64,
) -> Result<DivergenceReport, SriError> {
    let prescribed = classify_mrm(prescribed_vev)?;
    let kernel = classify_mrm(kernel_vev)?;

    let sri_difference = prescribed.mrm.abs_diff(kernel.mrm);

    let flag = match sri_difference {
        0 => DivergenceFlag::Green,
        1 => DivergenceFlag::Yellow,
        _ => DivergenceFlag::Red,
    };

    Ok(DivergenceReport {
        prescribed_sri: prescribed.mrm,
        kernel_sri: kernel.mrm,
        sri_difference,
        prescribed_vev,
        kernel_vev,
        flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MRM classification ──────────────────────────────────────────

    /// MRM 1: VEV < 0.5%.
    #[test]
    fn test_mrm_1() {
        let result = classify_mrm(0.003).unwrap();
        assert_eq!(result.mrm, 1);
    }

    /// MRM 2: 0.5% ≤ VEV < 5%.
    #[test]
    fn test_mrm_2() {
        assert_eq!(classify_mrm(0.005).unwrap().mrm, 2);
        assert_eq!(classify_mrm(0.03).unwrap().mrm, 2);
        assert_eq!(classify_mrm(0.049).unwrap().mrm, 2);
    }

    /// MRM 3: 5% ≤ VEV < 12%.
    #[test]
    fn test_mrm_3() {
        assert_eq!(classify_mrm(0.05).unwrap().mrm, 3);
        assert_eq!(classify_mrm(0.10).unwrap().mrm, 3);
    }

    /// MRM 4: 12% ≤ VEV < 20%.
    #[test]
    fn test_mrm_4() {
        assert_eq!(classify_mrm(0.12).unwrap().mrm, 4);
        assert_eq!(classify_mrm(0.15).unwrap().mrm, 4);
    }

    /// MRM 5: 20% ≤ VEV < 30%.
    #[test]
    fn test_mrm_5() {
        assert_eq!(classify_mrm(0.20).unwrap().mrm, 5);
        assert_eq!(classify_mrm(0.25).unwrap().mrm, 5);
    }

    /// MRM 6: 30% ≤ VEV < 80%.
    #[test]
    fn test_mrm_6() {
        assert_eq!(classify_mrm(0.30).unwrap().mrm, 6);
        assert_eq!(classify_mrm(0.50).unwrap().mrm, 6);
    }

    /// MRM 7: VEV ≥ 80%.
    #[test]
    fn test_mrm_7() {
        assert_eq!(classify_mrm(0.80).unwrap().mrm, 7);
        assert_eq!(classify_mrm(1.50).unwrap().mrm, 7);
    }

    /// Boundary: exactly 0 → MRM 1.
    #[test]
    fn test_mrm_zero() {
        assert_eq!(classify_mrm(0.0).unwrap().mrm, 1);
    }

    /// Negative VEV should error.
    #[test]
    fn test_mrm_negative() {
        assert!(classify_mrm(-0.01).is_err());
    }

    /// NaN VEV should error.
    #[test]
    fn test_mrm_nan() {
        assert!(classify_mrm(f64::NAN).is_err());
    }

    // ── VaR-equivalent volatility ───────────────────────────────────

    /// VEV = VaR / z_alpha.
    #[test]
    fn test_vev_computation() {
        let vev = var_equivalent_volatility(0.3099, 1.96);
        // 0.3099 / 1.96 ≈ 0.1581
        assert!((vev - 0.1581).abs() < 0.001);
    }

    // ── Divergence reporting ────────────────────────────────────────

    /// Same MRM → Green flag.
    #[test]
    fn test_divergence_green() {
        let report = divergence_report(0.15, 0.16).unwrap();
        assert_eq!(report.prescribed_sri, 4);
        assert_eq!(report.kernel_sri, 4);
        assert_eq!(report.flag, DivergenceFlag::Green);
    }

    /// MRM differs by 1 → Yellow flag.
    #[test]
    fn test_divergence_yellow() {
        // Prescribed VEV 15% → MRM 4, Kernel VEV 22% → MRM 5.
        let report = divergence_report(0.15, 0.22).unwrap();
        assert_eq!(report.prescribed_sri, 4);
        assert_eq!(report.kernel_sri, 5);
        assert_eq!(report.sri_difference, 1);
        assert_eq!(report.flag, DivergenceFlag::Yellow);
    }

    /// MRM differs by ≥ 2 → Red flag.
    #[test]
    fn test_divergence_red() {
        // Prescribed VEV 3% → MRM 2, Kernel VEV 15% → MRM 4.
        let report = divergence_report(0.03, 0.15).unwrap();
        assert_eq!(report.prescribed_sri, 2);
        assert_eq!(report.kernel_sri, 4);
        assert_eq!(report.sri_difference, 2);
        assert_eq!(report.flag, DivergenceFlag::Red);
    }
}
