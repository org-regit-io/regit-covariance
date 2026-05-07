<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MATH.md — regit-covariance

> Full formula derivations for every algorithm implemented in this crate.
> Each section maps to a source module and cites the primary paper reference.
> All formulas are shown in plain-text notation using code blocks.

---

## Table of contents

1. [Sample correlation matrix](#sample-correlation-matrix--srcmathsample_covariancers)
2. [Exponentially weighted correlation](#exponentially-weighted-correlation--srcmathsample_covariancers)
3. [Eigendecomposition](#eigendecomposition--srcmatheigenrs)
4. [Marchenko-Pastur density](#marchenko-pastur-density--srcmathmarchenko_pasturrs)
5. [Marchenko-Pastur noise variance fit](#marchenko-pastur-noise-variance-fit--srcmathmarchenko_pasturrs)
6. [Eigenvalue denoising — Constant](#eigenvalue-denoising--constant--srcmathdenoisers)
7. [Eigenvalue denoising — Target](#eigenvalue-denoising--target--srcmathdenoisers)
8. [Detoning — market mode removal](#detoning--market-mode-removal--srcmathdetoners)
9. [Ledoit-Wolf linear shrinkage](#ledoit-wolf-linear-shrinkage-2004--srcmathledoit_wolfrs)
10. [Ledoit-Wolf nonlinear analytical shrinkage](#ledoit-wolf-nonlinear-analytical-shrinkage-2020--srcmathledoit_wolfrs)
11. [Condition number](#condition-number--srcmathconditionrs)
12. [Parametric (Gaussian) VaR](#parametric-gaussian-var--srcmathvarrs)
13. [Cornish-Fisher VaR](#cornish-fisher-var--srcmathvarrs)
14. [Standard normal quantile](#standard-normal-quantile--srcmathvarrs)
15. [VaR-Equivalent Volatility (VEV)](#var-equivalent-volatility-vev--srcmathsrirs)
16. [PRIIPs SRI classification](#priips-sri-classification--srcmathsrirs)
17. [Divergence detection — prescribed vs kernel SRI](#divergence-detection--prescribed-vs-kernel-sri--srcmathsrirs)
18. [Log returns](#log-returns--srcdatareturnsrs)
19. [Algorithm references](#algorithm-references)

---

## Sample correlation matrix — `src/math/sample_covariance.rs`

**Source:** Marchenko & Pastur (1967) setup; Lopez de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2, Section 2.4.

### Setup

Given a returns matrix `R` of shape `T x N` (rows are time steps, columns are assets), let `r_{t,i}` be the return of asset `i` at time `t`. The implementation expects log returns:

```
r_{t,i} = ln(P_{t,i} / P_{t-1,i})
```

(See [Log returns](#log-returns--srcdatareturnsrs) for the conversion from price levels.)

### Standardisation

Each column is recentred and rescaled to zero mean and unit variance. Let `mu_i` and `s_i^2` be the sample mean and sample variance of column `i`:

```
mu_i  = (1/T) * sum_{t=1}^{T} r_{t,i}
s_i^2 = (1/T) * sum_{t=1}^{T} (r_{t,i} - mu_i)^2
```

Note: the implementation uses the **population variance** (`1/T`), not the unbiased sample variance (`1/(T-1)`). This is intentional and consistent with the random-matrix-theory literature where the eigenvalue spectrum of `(1/T) X^T X` is the object of interest.

The standardised column is:

```
x_{t,i} = (r_{t,i} - mu_i) / s_i
```

### Correlation matrix

The sample correlation matrix is then:

```
C = (1/T) * X^T * X
```

where `X` is the `T x N` matrix of standardised returns. Equivalently, in element form:

```
C_{i,j} = (1/T) * sum_{t=1}^{T} x_{t,i} * x_{t,j}
```

By construction the diagonal `C_{i,i} = 1` for all `i`, the matrix is symmetric, and its trace equals `N`.

### Observation ratio

A critical quantity throughout this crate is the observation ratio:

```
q = T / N
```

When `q` is much larger than 1, the sample correlation matrix is well-estimated. When `q` is close to 1 (or below), the matrix is heavily contaminated by estimation noise — this is the regime that random matrix theory addresses. For a fund with `N = 500` assets and `T = 252` daily observations, `q = 0.504`, and the sample matrix is rank-deficient with at least `N - T = 248` zero eigenvalues.

### Edge cases

**Empty matrix.** Returns `CovarianceError::EmptyMatrix`.

**Single observation (`T < 2`).** Returns `CovarianceError::InsufficientObservations` — variance estimation requires at least two points.

**Zero-variance column (`s_i^2 < f64::EPSILON`).** Returns `CovarianceError::ZeroVariance { column: i }`. A constant price series produces a zero-row in `X` and a degenerate correlation entry. The caller must drop or impute the offending asset before retrying.

---

## Exponentially weighted correlation — `src/math/sample_covariance.rs`

**Source:** RiskMetrics Technical Document (J.P. Morgan / Reuters, 1996), Section 5.

### Decay parameter

The implementation parametrises decay by **half-life** `h` (the number of periods over which weight halves):

```
decay = exp(-ln(2) / h)
```

so that `decay^h = 1/2`. Larger `h` means slower decay (weights spread further into the past); `h -> infinity` recovers equal weighting.

### Weights

For `T` observations indexed `0, 1, ..., T-1` (oldest to newest), the weight of observation `i` is:

```
w_i = decay^(T - 1 - i)
```

The most recent observation (`i = T-1`) has weight 1; older observations decay geometrically. Let:

```
W = sum_{i=0}^{T-1} w_i
```

### Weighted moments

Per-column weighted mean and weighted variance:

```
mu_i      = (1/W) * sum_t w_t * r_{t,i}
s_i^2     = (1/W) * sum_t w_t * (r_{t,i} - mu_i)^2
```

Off-diagonal entries are computed directly (no second pass through standardisation):

```
C_{i,j} = (1/W) * sum_t w_t * (r_{t,i} - mu_i) * (r_{t,j} - mu_j) / (s_i * s_j)
```

### Convergence to equal-weighted estimator

As `h -> infinity`, `decay -> 1`, all weights tend to 1, and the EWM correlation converges to the equal-weighted form. This is verified in `test_ewm_large_halflife_approximates_equal` with `h = 1e6`.

### Edge cases

**Non-positive or non-finite half-life.** Returns `CovarianceError::InvalidHalfLife(h)`.

**Zero weighted variance.** Same `ZeroVariance` error as the equal-weighted case.

---

## Eigendecomposition — `src/math/eigen.rs`

**Source:** Golub & Van Loan, *Matrix Computations*, 4th edition (2013), Chapter 8.

### Spectral decomposition

For a symmetric matrix `C` of dimension `N x N`, the spectral theorem guarantees a factorisation:

```
C = V * Lambda * V^T
```

where:
- `Lambda = diag(lambda_1, ..., lambda_N)` is the diagonal matrix of (real) eigenvalues
- `V` is the orthogonal matrix of eigenvectors as columns: `V^T * V = I`

The implementation wraps `nalgebra::SymmetricEigen` and post-sorts the output so that `lambda_1 >= lambda_2 >= ... >= lambda_N`. Eigenvectors are reordered to track their eigenvalues.

### Properties exposed via `EigenDecomposition`

- `reconstruct()` — returns `V * Lambda * V^T`. Used for verifying the factorisation and as a building block for denoising and detoning.
- `trace()` — returns `sum_i lambda_i`. For a correlation matrix this equals `N`.
- `count_above(threshold)` — used by Marchenko-Pastur to count signal eigenvalues.

### Why descending order

Two downstream operations rely on the descending convention:

1. **Detoning** removes the top `k` eigenvalues — these correspond to the largest factors (the market mode). Sorting puts them at index `0, 1, ..., k-1` for direct slicing.
2. **Condition number** is computed as `lambda[0] / lambda[n-1]` (largest divided by smallest).

### Edge cases

**Non-square matrix.** Returns `EigenError::NotSquare`.

**Empty matrix.** Returns `EigenError::EmptyMatrix`.

**Rank deficiency (`q < 1`).** When `T < N`, the sample correlation matrix has at least `N - T` exact zero eigenvalues. The decomposition still succeeds; the resulting `lambda_min = 0` makes the condition number infinite. Denoising or shrinkage is essential before any inversion.

**Symmetry.** `nalgebra::SymmetricEigen` reads only the lower triangle. Off-symmetric input is silently treated as if its lower triangle were the symmetric part — callers must supply a symmetric matrix.

---

## Marchenko-Pastur density — `src/math/marchenko_pastur.rs`

**Source:** Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of random matrices." *Matematicheskii Sbornik*, 114(4), 507-536.

### Setup

Let `X` be a `T x N` matrix of i.i.d. entries with zero mean and variance `sigma^2`. As `T, N -> infinity` with `q = T / N` held fixed and `q >= 1`, the eigenvalues of the sample correlation matrix `(1/T) * X^T * X` converge to the Marchenko-Pastur (MP) density.

### Density

The MP probability density function is:

```
f_MP(lambda) = (q / (2 * pi * sigma^2 * lambda)) * sqrt((lambda_+ - lambda) * (lambda - lambda_-))
```

defined for `lambda` in the support `[lambda_-, lambda_+]` and zero outside. Note the `sqrt` factor vanishes at the support endpoints, giving the characteristic semi-elliptical shape.

### Bounds

```
sqrt_inv_q = sqrt(1 / q)
lambda_+ = sigma^2 * (1 + sqrt_inv_q)^2
lambda_- = sigma^2 * (1 - sqrt_inv_q)^2
```

For pure noise on a true correlation matrix (`sigma^2 = 1`):

| q | lambda_- | lambda_+ |
|---|---|---|
| 1 | 0.000 | 4.000 |
| 2 | 0.086 | 2.914 |
| 5 | 0.302 | 2.098 |
| 10 | 0.475 | 1.725 |

The bulk shrinks toward `1` as `q` grows: with abundant data, all sample eigenvalues concentrate near unity.

### Interpretation

Eigenvalues of an empirical correlation matrix that fall **inside** `[lambda_-, lambda_+]` are statistically consistent with pure noise. Eigenvalues **above** `lambda_+` carry genuine signal (a real correlation structure that survives noise filtering). The market mode of a typical equity universe shows up as a single eigenvalue far above `lambda_+`; sector modes show up as a small cluster just above the upper edge.

### Numerical notes

- The density is computed only inside the support; outside it is returned as `0.0`.
- The denominator `2 * pi * sigma^2 * lambda` is checked against `f64::EPSILON` — for `lambda` exactly at the lower edge when `sigma^2` is tiny, a guard returns `0.0` rather than producing `inf`.
- The bounds are computed with explicit handling of `q <= 0` and non-finite inputs (return `(0, 0)` so callers can detect the invalid state).

---

## Marchenko-Pastur noise variance fit — `src/math/marchenko_pastur.rs`

**Source:** Lopez de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2, Code Snippet 2.4.

### Problem

Given an empirical eigenvalue spectrum `{lambda_1, ..., lambda_N}` from a sample correlation matrix and the observation ratio `q`, find the `sigma^2` that best matches the noise floor of the MP law. Once `sigma^2` is known, the upper edge `lambda_+(sigma^2, q)` partitions the spectrum into noise (below) and signal (above).

### Fixed-point iteration

The implementation uses the algorithm of Lopez de Prado (2018):

```
1. Initialise sigma^2 = mean(lambda_1, ..., lambda_N)
2. Repeat until convergence:
     compute lambda_+ from (sigma^2, q)
     let N_noise = { i : lambda_i <= lambda_+ }
     sigma^2_new = mean { lambda_i : i in N_noise }
     if |sigma^2_new - sigma^2| < tol: stop
     sigma^2 = sigma^2_new
3. Return sigma^2 and the implied (lambda_-, lambda_+)
```

with `tol = 1e-10` and `max_iter = 1000`.

**Why a fixed point.** If we knew which eigenvalues were noise, the noise variance would simply be their mean. Since we do not know, we guess (the mean of all eigenvalues, which over-estimates `sigma^2` if signal modes are present), recompute the upper edge, and re-mean. The iteration shrinks the noise set monotonically until it stabilises.

**Convergence in practice.** For correlation matrices with a strong market mode, the iteration converges in 3-8 steps. The starting estimate (`mean(lambda)`) is robust because adding the (large) signal eigenvalues is a small relative perturbation when `N` is large.

### Output

`MpFit { sigma_sq, lambda_plus, lambda_minus, signal_count, noise_count, q }` where:
- `signal_count` = number of `lambda_i > lambda_+`
- `noise_count` = `N - signal_count`

### Edge cases

**Invalid `q` (`q <= 0` or non-finite).** Returns `MpError::InvalidQ`.

**Empty eigenvalues.** Returns `MpError::EmptyEigenvalues`.

**All eigenvalues are signal.** If a transient iteration sets the noise set to empty, `sigma^2` is clamped to `f64::EPSILON` and the loop exits. This is rare for real returns; it can occur on contrived inputs (e.g., a single-asset spectrum).

---

## Eigenvalue denoising — Constant — `src/math/denoise.rs`

**Source:** Lopez de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2, Code Snippet 2.6. Bun, Bouchaud & Potters (2017), *Cleaning large correlation matrices*, *Physics Reports* 666, Section 4.

### Procedure

Once the noise edge `lambda_+` is identified, the spectrum is partitioned:

```
Noise:  { i : lambda_i <= lambda_+ }
Signal: { i : lambda_i  > lambda_+ }
```

The **constant** method replaces every noise eigenvalue with the noise-set mean and leaves signal eigenvalues untouched:

```
noise_mean = (1 / |Noise|) * sum_{i in Noise} lambda_i

lambda_tilde_i = noise_mean    if i in Noise
              = lambda_i       if i in Signal
```

### Reconstruction

The cleaned correlation matrix is reassembled from the original eigenvectors:

```
C_tilde = V * diag(lambda_tilde_1, ..., lambda_tilde_N) * V^T
```

### Trace preservation

```
sum_{i in Noise} lambda_tilde_i = |Noise| * noise_mean
                                 = |Noise| * (1 / |Noise|) * sum_{i in Noise} lambda_i
                                 = sum_{i in Noise} lambda_i
```

The noise-set sum is unchanged, signal eigenvalues are unchanged, so the total trace is unchanged. Since the trace of a correlation matrix equals `N`, the diagonal of `C_tilde` averages to 1 (and is exactly 1 for orthonormal `V`).

### Why this works

The flat replacement removes the noise-induced eigenvalue dispersion that random matrix theory predicts. It does not introduce any opinion about the true noise structure: under the null hypothesis that all noise eigenvalues come from i.i.d. randomness, their mean is the maximum-likelihood estimator of the true noise level.

---

## Eigenvalue denoising — Target — `src/math/denoise.rs`

**Source:** Lopez de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2.

### Procedure

The **target** method shrinks noise eigenvalues toward the identity (`lambda_i -> 1`) and then rescales the entire spectrum to preserve the original trace:

```
1. For each i in Noise:  lambda_tilde_i := 1.0
2. For each i in Signal: lambda_tilde_i := lambda_i  (unchanged)
3. Compute current_trace = sum_i lambda_tilde_i
4. If current_trace > EPSILON:
     scale = original_trace / current_trace
     lambda_tilde_i := scale * lambda_tilde_i  for all i
```

### Why rescale

Step 1 alone breaks trace preservation: replacing `|Noise|` eigenvalues with `1.0` typically does not match their original sum. The post-multiplication by `original_trace / current_trace` restores the trace exactly. Because the rescale is a uniform multiplicative factor across all eigenvalues, the relative ordering is preserved and the matrix remains PSD.

### Renormalisation to unit diagonal

After reconstruction `C_tilde = V * Lambda_tilde * V^T`, the diagonal entries may drift slightly from 1.0 due to the trace rescaling. The helper `renormalize_to_correlation(C_tilde)` performs:

```
D = diag(sqrt(C_tilde_{1,1}), ..., sqrt(C_tilde_{N,N}))
C_renorm = D^{-1} * C_tilde * D^{-1}
```

so the result has unit diagonal and the off-diagonal entries are rescaled accordingly. This is the standard post-denoising step per de Prado (2018), Chapter 2.

### Choosing Constant vs Target

- **Constant** is the default and the more conservative choice: it preserves the bulk's noise-mean structure and produces matrices closer to the original sample.
- **Target** pushes the noise floor to a hard `1.0` (the identity target), which is appropriate when the user wants to bias toward shrinking the matrix toward the identity (no correlation among noise components). It tends to produce slightly better-conditioned matrices.

Both methods produce symmetric, PSD, trace-preserving outputs.

---

## Detoning — market mode removal — `src/math/detone.rs`

**Source:** Lopez de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2, Section 2.6.

### Motivation

The largest eigenvalue of a real-world equity correlation matrix typically captures the **market mode**: a single collective factor that drives all assets in the same direction. While genuine, this mode dominates so heavily that it can obscure finer structure (sectors, factors, idiosyncratic clusters). Detoning removes the top `k` eigenmodes (usually `k = 1`) to expose the residual correlations.

### Procedure

```
1. Take the eigendecomposition C = V * Lambda * V^T (eigenvalues sorted descending)
2. Save removed_eigenvalues_i = lambda_i for i = 0 .. k-1
3. Set lambda_i := 0 for i = 0 .. k-1
4. Reconstruct: C_detoned = V * diag(0, ..., 0, lambda_k, ..., lambda_{N-1}) * V^T
5. Return (C_detoned, removed_eigenvalues)
```

### Re-addition

If the full covariance is needed later (e.g., for risk attribution), the helper `readd_tones` reconstructs:

```
C_full = V * diag(lambda_tilde_1 + removed_1, ..., lambda_tilde_N + removed_N) * V^T
```

This combines a denoised idiosyncratic spectrum (`lambda_tilde`) with the original market modes — useful when the denoising step was applied to the detoned matrix to clean noise without contaminating the market signal.

### When to use detoned vs full

- **Detoned** matrix: clustering, sector identification, residual-factor analysis — anywhere the market mode is a confound.
- **Full** denoised matrix: portfolio risk calculations (VaR, SRI), portfolio optimisation. The market mode contains real risk that must not be removed before risk reporting.

### Diagonal

Detoning does **not** rescale the diagonal back to `1`. The detoned matrix is a covariance object whose diagonal reflects the residual variance after market-mode removal. Callers who need a correlation interpretation should pass the result through `renormalize_to_correlation`.

### Edge cases

**`k = 0`.** Reconstructs the input unchanged (identity operation).

**`k > N`.** Panics with a clear message — cannot remove more modes than dimensions.

---

## Ledoit-Wolf linear shrinkage (2004) — `src/math/ledoit_wolf.rs`

**Source:** Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365-411.

### Estimator

Linear shrinkage convex-combines the sample correlation `S` with a structured target `F`:

```
C_LW = delta * F + (1 - delta) * S
```

where `delta` in `[0, 1]` is the shrinkage intensity. The implementation uses the **constant-correlation target**:

```
mean_corr = mean { S_{i,j} : i < j }    (mean of strictly upper-triangular entries)

F_{i,j} = 1            if i == j
        = mean_corr    if i != j
```

This target preserves the trace (`F_{i,i} = 1 = S_{i,i}`) and represents the simplest non-trivial structured correlation: every pair has the same correlation.

### Optimal intensity

The Ledoit-Wolf optimal `delta` minimises the expected Frobenius distance to the true correlation. The analytical formula (with the constant-correlation target):

```
gamma_hat = ||F - S||_F^2                                      (squared Frobenius distance)

pi_hat    = (1/T) * sum_t sum_{i,j} (x_{t,i} * x_{t,j} - S_{i,j})^2
                                                                (sum of asymptotic variances)

rho_hat   ~ pi_hat                                             (conservative approx)

kappa     = (pi_hat - rho_hat) / gamma_hat
delta     = clamp(kappa / T, 0, 1)
```

The implementation uses the conservative approximation `rho_hat = pi_hat`, which yields a slightly more aggressive (larger) `delta` than the full Ledoit-Wolf formula. For most practical inputs this is within numerical noise of the optimal and avoids the bookkeeping of the full off-diagonal cross-moment terms.

### Properties

- **Always PSD.** A convex combination of two PSD matrices (`S`, `F`) is PSD.
- **Always well-conditioned.** As `delta -> 1`, the estimator approaches the constant-correlation matrix whose condition number is bounded — see [Condition number](#condition-number--srcmathconditionrs) for the closed-form `(1 + (N-1) * rho) / (1 - rho)`.
- **Trace preservation.** Since `S_{i,i} = F_{i,i} = 1`, the diagonal of `C_LW` is `1` exactly.

### When to use linear vs nonlinear shrinkage

Linear shrinkage is the right tool when:
- `N` and `T` are both moderate (say `N < 100`)
- a structural prior of "constant pairwise correlation" is reasonable
- speed matters (linear shrinkage is `O(N^2 * T)`, no eigendecomposition required)

For larger `N`, see [Ledoit-Wolf nonlinear analytical shrinkage](#ledoit-wolf-nonlinear-analytical-shrinkage-2020--srcmathledoit_wolfrs).

---

## Ledoit-Wolf nonlinear analytical shrinkage (2020) — `src/math/ledoit_wolf.rs`

**Source:** Ledoit, O. & Wolf, M. (2020). "Analytical nonlinear shrinkage of large-dimensional covariance matrices." *Annals of Statistics*, 48(5), 3043-3065. Implementation reference: Ledoit & Wolf (2017), numerical implementation of the QuEST function.

### Idea

Rather than mixing the sample matrix with one structured target, the 2020 estimator applies an **individual** optimal shrinkage to each eigenvalue. The shrinkage function is derived from the Marchenko-Pastur equation and the Stieltjes transform of the limiting spectral distribution. There are no tuning parameters.

### Procedure

For each sample eigenvalue `lambda_i`, compute a sample Hilbert-transform proxy:

```
H(lambda_i) = (1/N) * sum_{j != i} 1 / (lambda_j - lambda_i)
```

This is the discrete Stieltjes-transform-style sum that approximates the Hilbert transform of the empirical spectral density at `lambda_i`. Let `c = N / T` (the inverse of the observation ratio `q`). The shrunk eigenvalue is:

```
denominator = pi^2 * lambda_i^2 * H(lambda_i)^2 * c^2
            + (1 - c - c * lambda_i * H(lambda_i))^2

lambda_tilde_i = lambda_i / denominator
```

### Trace preservation

After applying the per-eigenvalue formula, the spectrum is rescaled to match the original trace:

```
scale = sum(lambda_i) / sum(lambda_tilde_i)
lambda_tilde_i := scale * lambda_tilde_i
```

This step ensures the diagonal of `V * diag(lambda_tilde) * V^T` averages to `1` for a correlation input.

### Reconstruction

```
C_tilde = V * diag(lambda_tilde_1, ..., lambda_tilde_N) * V^T
```

### Properties

- **PSD by construction.** All `lambda_tilde_i >= 0` because the denominator is positive and `lambda_i >= 0`.
- **No tuning.** The formula is fully data-driven; there is no equivalent of `delta`.
- **Better conditioning than the sample matrix.** The shrinkage compresses the eigenvalue range, lowering the condition number. Verified in `test_condition_number_improvement`.

### Implementation note

The Hilbert-transform sum has a singularity when `lambda_j = lambda_i` for `j != i` (degenerate eigenvalues). The implementation guards with `if diff.abs() > f64::EPSILON` and skips degenerate pairs. This is a sample-eigenvalue-degeneracy guard and does not affect typical inputs where eigenvalues are simple.

### Comparison with constant denoising

The MP-based [Constant denoising](#eigenvalue-denoising--constant--srcmathdenoisers) replaces noise eigenvalues with a single mean — a step function of `lambda`. Nonlinear shrinkage applies a smooth, monotone shrinkage curve. The two are complementary: MP-based denoising is interpretable and respects the noise/signal partition; nonlinear shrinkage is theoretically optimal in a Frobenius-loss sense but harder to inspect.

---

## Condition number — `src/math/condition.rs`

**Source:** Golub & Van Loan, *Matrix Computations* (2013); Lopez de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2.

### Definition

For a symmetric PSD matrix `Sigma` with eigenvalues `lambda_1 >= ... >= lambda_N >= 0`:

```
kappa(Sigma) = lambda_max / lambda_min = lambda_1 / lambda_N
```

(The 2-norm condition number; for symmetric matrices the largest and smallest eigenvalues coincide with the largest and smallest singular values.)

### Why it matters

Mean-variance portfolio optimisation requires inverting `Sigma`. The relative error in `Sigma^{-1}` from a relative perturbation `epsilon` in `Sigma` is bounded by `kappa(Sigma) * epsilon`. With `kappa = 10^4` and `epsilon = 10^{-3}` (typical sampling error), the inverse can be off by an order of magnitude — and so can the resulting portfolio weights.

### Health classification

The implementation classifies a condition number into three bands:

| `kappa` range | Health | Implication |
|---|---|---|
| `kappa < 100` | Healthy | Inverse is numerically reliable |
| `100 <= kappa < 1000` | Acceptable | Proceed with caution |
| `kappa >= 1000` | Unstable | Optimisation results may be unreliable |

These thresholds come from common practice in numerical linear algebra. They are not universal — for some applications (e.g., risk reporting where only quadratic forms are evaluated, no matrix inverse) far higher condition numbers are tolerable.

### Closed-form for known structures

For a constant-correlation matrix with off-diagonal `rho` and dimension `N`:

```
lambda_max = 1 + (N - 1) * rho
lambda_min = 1 - rho
kappa      = (1 + (N - 1) * rho) / (1 - rho)
```

The test `test_known_condition_number` uses this with `N = 3`, `rho = 0.9`: `kappa = 2.8 / 0.1 = 28`.

### Before/after comparison

`compare(before, after)` returns a `ConditionImprovement` with the improvement factor `kappa_before / kappa_after`. Values > 1 mean denoising helped; values < 1 mean it hurt (rare, indicates a misapplied method or signal eigenvalue erroneously folded into the noise set).

### Edge cases

**`lambda_min = 0` (rank-deficient).** The condition number is reported as `f64::INFINITY`. This is the default state of any sample correlation matrix with `T < N`.

**`lambda_min < f64::EPSILON`.** Treated as zero — same `INFINITY` outcome.

---

## Parametric (Gaussian) VaR — `src/math/var.rs`

**Source:** PRIIPs Delegated Regulation (EU) 2017/653, Annex II, Part 1, Section 12.

### Setup

For a portfolio with weight vector `w`, expected return vector `mu`, and covariance matrix `Sigma`:

```
mu_p     = w^T * mu                         (portfolio expected return)
sigma_p  = sqrt(w^T * Sigma * w)            (portfolio volatility)
```

### Formula

Under the Gaussian assumption that the portfolio return is `N(mu_p, sigma_p^2)`, the VaR at confidence level `alpha` is:

```
VaR_alpha = -(mu_p - z_alpha * sigma_p)
          = z_alpha * sigma_p - mu_p
```

where `z_alpha = Phi^{-1}(alpha)` is the upper-tail standard normal quantile. For PRIIPs the standard confidence is `alpha = 0.975`, giving `z ~ 1.96`.

The minus sign converts a return-distribution quantile into a positive loss number. For a zero-mean portfolio the formula collapses to `VaR = z_alpha * sigma_p`.

### Implementation notes

- The bilinear form `w^T * Sigma * w` is computed via `nalgebra` matrix-vector multiplication, then indexed `[(0, 0)]` from the resulting `1x1` matrix.
- A guard `sigma_p < f64::EPSILON` returns `VarError::ZeroVolatility` — typically indicating identical or perfectly anti-correlated assets.
- Confidence is validated to `alpha in (0, 1)`; boundary values produce `VarError::InvalidConfidence`.

---

## Cornish-Fisher VaR — `src/math/var.rs`

**Source:** Cornish, E. A. & Fisher, R. A. (1938). "Moments and cumulants in the specification of distributions." *Revue de l'Institut International de Statistique*, 5(4), 307-320. PRIIPs Delegated Regulation (EU) 2017/653, Annex II, Part 1, Section 13.

### Motivation

The Gaussian VaR ignores skewness and kurtosis. For products with non-symmetric or fat-tailed return distributions, this systematically under- or over-estimates tail risk. The Cornish-Fisher expansion adjusts the Gaussian quantile using sample skewness `S` and excess kurtosis `K`.

### Adjusted quantile

Let `z = z_{1-alpha} = Phi^{-1}(1 - alpha)` be the **lower-tail** quantile (negative for `alpha > 0.5`). The Cornish-Fisher modified quantile is:

```
z_CF = z
     + (z^2 - 1) * S / 6
     + (z^3 - 3*z) * K / 24
     - (2*z^3 - 5*z) * S^2 / 36
```

The four terms correspond to:
1. Gaussian baseline
2. First-order skewness correction
3. First-order kurtosis correction
4. Second-order skewness correction (S^2 cross term)

### VaR formula

```
VaR_CF = -(mu_p + z_CF * sigma_p)
```

Because `z` (and hence `z_CF`) is negative, `z_CF * sigma_p` is negative, and the leading minus turns the loss into a positive number. Negative skewness (fat left tail) makes `z_CF` more negative and increases VaR; positive excess kurtosis (heavy tails) does the same.

### Sign conventions

The implementation works with the **lower-tail** `z = Phi^{-1}(1 - alpha)` rather than the upper-tail `z_alpha = Phi^{-1}(alpha)`. The two conventions are related by `z_{1-alpha} = -z_alpha`. With the lower-tail convention, the Cornish-Fisher polynomial above matches the form usually printed in regulatory texts (e.g., PRIIPs RTS Annex II) and the VaR formula `-(mu + z_CF * sigma)` produces a positive number directly.

### Validation

- With `S = K = 0`, `z_CF = z = -z_alpha`, and the Cornish-Fisher VaR exactly matches the parametric Gaussian VaR. Verified in `test_cf_equals_parametric_when_gaussian`.
- Negative skewness increases VaR (fatter left tail). Verified in `test_cf_negative_skewness_increases_var`.
- Positive excess kurtosis increases VaR (heavier tails). Verified in `test_cf_excess_kurtosis_increases_var`.

### When to use

The Cornish-Fisher correction is reliable when departures from normality are moderate (`|S| < 1`, `K < 4`). For severely non-Gaussian distributions, the polynomial can produce non-monotone quantiles and the expansion breaks down. PRIIPs explicitly prescribes Cornish-Fisher for category 2 products (those with non-derivative complex payoffs).

---

## Standard normal quantile — `src/math/var.rs`

**Source:** Abramowitz & Stegun, *Handbook of Mathematical Functions* (1964), formula 26.2.23.

### Formula

For `p in (0.5, 1)`:

```
t = sqrt(-2 * ln(1 - p))

z = t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
```

with constants:

```
c0 = 2.515517      d1 = 1.432788
c1 = 0.802853      d2 = 0.189269
c2 = 0.010328      d3 = 0.001308
```

For `p < 0.5`, the implementation uses the symmetry `Phi^{-1}(p) = -Phi^{-1}(1 - p)`.

### Accuracy

Maximum absolute error `~ 4.5e-4` over `p in (0, 1)`. Sufficient for VaR computation at standard confidence levels (90%, 95%, 97.5%, 99%) where the third-decimal precision is well below the noise from the underlying covariance estimate. For higher precision (e.g., when computing tail expectations), a higher-order rational approximation or Newton refinement on `erfc` would be required.

### Validation

- `z(0.975) ~ 1.96` — verified in `test_normal_quantile_975`
- `z(0.5) = 0` — verified in `test_normal_quantile_50`
- `z(0.025) ~ -1.96` — verified by symmetry in `test_normal_quantile_025`

---

## VaR-Equivalent Volatility (VEV) — `src/math/sri.rs`

**Source:** PRIIPs Delegated Regulation (EU) 2017/653, Annex II.

### Definition

The VaR-Equivalent Volatility expresses a tail-risk number in units of (Gaussian-equivalent) volatility:

```
VEV = VaR / z_alpha
```

For PRIIPs at `alpha = 0.975`, `z_alpha = 1.96`:

```
VEV = VaR / 1.96
```

### Interpretation

If the portfolio return were Gaussian with mean zero, then `VaR_0.975 = z * sigma`, so `VEV = sigma`. For non-Gaussian returns, VEV is the volatility of the Gaussian distribution that would produce the same Cornish-Fisher VaR. It serves as the input to the SRI classification — see [PRIIPs SRI classification](#priips-sri-classification--srcmathsrirs).

### Implementation note

The implementation uses the simple `VEV = VaR / z_alpha` form. Some PRIIPs templates include an additional `sqrt(T_holding)` factor when scaling from a per-period VaR to an annualised VEV — this scaling is performed by the caller before passing into `var_equivalent_volatility`, not inside this function.

### Edge cases

**`z_alpha ~ 0`.** Returns `0.0` to avoid division by zero. This is unreachable in practice for any meaningful confidence level.

---

## PRIIPs SRI classification — `src/math/sri.rs`

**Source:** PRIIPs Delegated Regulation (EU) 2017/653, Annex II, Part 1, Sections 3-5. PRIIPs Regulation (EU) 1286/2014.

### Market Risk Measure (MRM) classes

The Summary Risk Indicator (SRI) is a 1-7 scale published in Key Information Documents (KIDs) for retail investors. The market-risk component (MRM) is determined from the VEV using the regulatory thresholds:

| MRM class | VEV range |
|---|---|
| 1 | `VEV < 0.5%` |
| 2 | `0.5% <= VEV < 5%` |
| 3 | `5% <= VEV < 12%` |
| 4 | `12% <= VEV < 20%` |
| 5 | `20% <= VEV < 30%` |
| 6 | `30% <= VEV < 80%` |
| 7 | `VEV >= 80%` |

The full SRI also incorporates a Credit Risk Measure (CRM) for products with credit exposure; that calculation depends on issuer rating and is out of scope for this crate (which deals only with covariance-driven market risk).

### Implementation

```
classify_mrm(vev: f64) -> SriResult { mrm: u8, vev: f64 }
```

The input VEV is in decimal form (e.g., `0.15` for 15%). The function multiplies by 100 internally to compare against the percentage thresholds, then returns the integer MRM class.

### Edge cases

**`VEV < 0` or `NaN`.** Returns `SriError::InvalidVev`.

**`VEV = 0`.** Classifies as MRM 1 (the lowest risk band). A truly zero-volatility product is a degenerate case (constant payoff), but the boundary is handled cleanly.

---

## Divergence detection — prescribed vs kernel SRI — `src/math/sri.rs`

**Source:** Implementation-specific. The divergence framework is the contribution of the regit-covariance kernel: a regulator-readable comparison between the SRI computed by the prescribed PRIIPs methodology and the SRI computed from a denoised covariance matrix.

### Procedure

```
prescribed_VEV  -> classify_mrm -> prescribed_SRI
kernel_VEV      -> classify_mrm -> kernel_SRI

sri_difference  = |prescribed_SRI - kernel_SRI|

flag = Green     if sri_difference == 0
     = Yellow    if sri_difference == 1
     = Red       if sri_difference >= 2
```

### Interpretation

| Flag | Meaning |
|---|---|
| Green | Noise has negligible impact on the regulatory risk classification — the prescribed methodology is producing the right answer |
| Yellow | Marginal divergence — the product sits near an MRM boundary; small estimation errors could swing the class either way |
| Red | Material divergence — the prescribed SRI is materially distorted by estimation noise, either overstating risk (penalising the product commercially) or understating it (exposing investors to inadequately disclosed risk) |

### Why this matters

PRIIPs prescribes a specific recipe (sample covariance, Cornish-Fisher, threshold lookup). When `q = T / N` is small — typical for funds with hundreds of underlying assets and only a few years of daily history — the sample covariance is dominated by noise, and the resulting VEV can drift by 5-30% across resampling realisations. Without a denoising baseline, the regulatory SRI can be unstable. The divergence flag surfaces this instability to risk-management and compliance teams.

### Edge cases

**Invalid VEV on either side.** Propagates `SriError::InvalidVev`.

---

## Log returns — `src/data/returns.rs`

**Source:** Standard definition.

### Conversion

For a price matrix `P` of shape `T x N` (rows are dates, columns are assets), the log-return matrix has shape `(T-1) x N`:

```
r_{t,i} = ln(P_{t,i} / P_{t-1,i})       for t = 1, ..., T-1
```

### Properties

- **Time-additive.** The log return over `k` periods equals the sum of the `k` per-period log returns: `ln(P_T / P_0) = sum_{t=1}^{T} ln(P_t / P_{t-1})`. Simple percentage returns do not have this property.
- **Symmetric around zero.** A `+50%` move and the reverse `-33.3%` move (both bringing a price back to its origin) have log returns of equal magnitude and opposite sign. Simple returns do not.
- **Approximately equal to simple returns for small moves.** For `r_simple < 5%`, `ln(1 + r_simple) ~ r_simple - r_simple^2/2`, an error below 0.13%.

### Edge cases

**Non-positive prices.** Returns `ReturnsError::NonPositivePrice` — `ln` is undefined for `P <= 0`. This catches data corruption (Yahoo Finance occasionally returns 0 or NaN for missing observations).

**`T < 2`.** Returns `ReturnsError::InsufficientObservations` — at least two prices are needed to form one return.

---

## Algorithm references

| Algorithm | Primary reference |
|---|---|
| Sample correlation / covariance | Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Ch. 2 |
| Exponentially weighted moments | RiskMetrics Technical Document (J.P. Morgan / Reuters, 1996), Section 5 |
| Symmetric eigendecomposition | Golub & Van Loan, *Matrix Computations*, 4th ed. (Johns Hopkins, 2013), Ch. 8 |
| Marchenko-Pastur law | Marchenko & Pastur, *Matematicheskii Sbornik* 114(4):507-536 (1967) |
| RMT noise filtering of correlation matrices | Laloux, Cizeau, Bouchaud & Potters, *Physical Review Letters* 83(7):1467-1470 (1999) |
| Cleaning correlation matrices (review) | Bun, Bouchaud & Potters, *Physics Reports* 666:1-109 (2017) |
| Eigenvalue replacement (Constant + Target) | Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Ch. 2, Code Snippet 2.6 |
| Detoning | Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Ch. 2, Section 2.6 |
| Linear shrinkage | Ledoit & Wolf, *Journal of Multivariate Analysis* 88(2):365-411 (2004) |
| Nonlinear analytical shrinkage | Ledoit & Wolf, *Annals of Statistics* 48(5):3043-3065 (2020) |
| QuEST function (numerical implementation) | Ledoit & Wolf, *Journal of Multivariate Analysis* 159:55-77 (2017) |
| Statistical physics view of risk | Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing*, 2nd ed. (Cambridge, 2003) |
| Machine learning for asset managers | Lopez de Prado, *Machine Learning for Asset Managers* (Cambridge, 2020) |
| Standard normal quantile | Abramowitz & Stegun, *Handbook of Mathematical Functions*, formula 26.2.23 (1964) |
| Cornish-Fisher expansion | Cornish & Fisher, *Revue de l'Institut International de Statistique* 5(4):307-320 (1938) |
| PRIIPs SRI / VEV / MRM | Commission Delegated Regulation (EU) 2017/653, Annex II |
| PRIIPs Regulation (parent act) | Regulation (EU) 1286/2014 of the European Parliament and of the Council |

---

*Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.*
