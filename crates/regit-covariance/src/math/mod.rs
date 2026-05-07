// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Mathematical core — covariance estimation, denoising, and risk metrics.
//!
//! Each submodule implements exactly one mathematical operation, following
//! the principle of one concern per file. All functions are pure:
//! data in, result out, no side effects.
//!
//! # Pipeline
//!
//! ```text
//! Returns → Sample Covariance → Eigendecomposition
//!     → Marchenko-Pastur filtering (denoise)
//!     → Detoning (optional)
//!     → Ledoit-Wolf shrinkage (alternative)
//!     → Risk metrics (VaR, SRI)
//! ```
//!
//! # References
//!
//! - Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues
//!   for some sets of random matrices.
//! - Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
//!   large-dimensional covariance matrices.
//! - Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of
//!   large-dimensional covariance matrices.
//! - López de Prado, M. (2018). Advances in Financial Machine Learning,
//!   Chapters 2–3.

#![forbid(unsafe_code)]

pub mod condition;
pub mod denoise;
pub mod detone;
pub mod eigen;
pub mod ledoit_wolf;
pub mod marchenko_pastur;
pub mod sample_covariance;
pub mod sri;
pub mod var;
