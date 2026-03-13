<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mathematical Foundations

This document describes the mathematical methods underpinning the **regit-covariance** pipeline. The system ingests asset return series, estimates and denoises their covariance structure using Random Matrix Theory, and produces risk metrics aligned with the European PRIIPs regulation. The end-to-end flow is:

> **Log returns** $\rightarrow$ **Sample covariance / correlation** $\rightarrow$ **Marchenko-Pastur fit** $\rightarrow$ **Eigenvalue denoising** $\rightarrow$ **Detoning** $\rightarrow$ **Shrinkage** $\rightarrow$ **VaR & VEV** $\rightarrow$ **SRI classification** $\rightarrow$ **Divergence detection**

Each section below formalises one stage of this pipeline.

---

## 1. Sample Covariance and Correlation

Given $N$ assets observed over $T$ periods, let $r_{t,i} = \ln(P_{t,i} / P_{t-1,i})$ denote the log return of asset $i$ at time $t$. The sample covariance matrix is

$$
\hat{\Sigma}_{ij} = \frac{1}{T-1} \sum_{t=1}^{T} (r_{t,i} - \bar{r}_i)(r_{t,j} - \bar{r}_j)
$$

where $\bar{r}_i = T^{-1}\sum_t r_{t,i}$. The sample correlation matrix is obtained by standardising:

$$
\hat{C}_{ij} = \frac{\hat{\Sigma}_{ij}}{\sqrt{\hat{\Sigma}_{ii}\,\hat{\Sigma}_{jj}}}
$$

A critical quantity is the **observation ratio**

$$
q = \frac{T}{N}
$$

When $q$ is not much larger than 1, the sample correlation matrix is heavily contaminated by estimation noise. Random Matrix Theory provides the tools to separate signal from noise in this regime.

---

## 2. Marchenko-Pastur Distribution

The Marchenko-Pastur (MP) law describes the limiting spectral density of large random covariance matrices. Consider a $T \times N$ matrix $\mathbf{X}$ whose entries are i.i.d. with zero mean and variance $\sigma^2$. As $T, N \to \infty$ with $q = T/N$ held fixed (and $q > 1$), the eigenvalues $\lambda$ of the sample correlation matrix $\frac{1}{T}\mathbf{X}^\top\mathbf{X}$ converge to the density

$$
f_{\text{MP}}(\lambda) = \frac{q}{2\pi\sigma^2} \;\frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\lambda}
$$

for $\lambda \in [\lambda_-,\;\lambda_+]$, and $f_{\text{MP}}(\lambda) = 0$ otherwise. The bounds are

$$
\lambda_{\pm} = \sigma^2 \left(1 + \frac{1}{q} \pm \frac{2}{\sqrt{q}}\right)
$$

When applied to a true correlation matrix ($\sigma^2 = 1$ for pure noise), eigenvalues falling within $[\lambda_-, \lambda_+]$ are consistent with random noise. Eigenvalues exceeding $\lambda_+$ carry genuine signal.

### Fitting $\sigma^2$ from the bulk

In practice, the noise variance $\sigma^2$ is not known a priori. It is estimated by fitting the MP density to the empirical eigenvalue histogram of $\hat{C}$, restricting attention to eigenvalues below an initial guess of $\lambda_+$. The fit minimises the sum of squared differences between the empirical and theoretical CDFs, yielding $\hat{\sigma}^2$ and the corresponding noise edge $\hat{\lambda}_+$.

---

## 3. Eigenvalue Denoising — Target Shrinkage

Once the noise edge $\hat{\lambda}_+$ is identified, eigenvalues are partitioned into **noise** ($\lambda_i \leq \hat{\lambda}_+$) and **signal** ($\lambda_i > \hat{\lambda}_+$) groups.

The **target shrinkage** method replaces every noise eigenvalue with the average of the noise eigenvalues:

$$
\tilde{\lambda}_i =
\begin{cases}
\displaystyle\frac{1}{|\mathcal{N}|}\sum_{j \in \mathcal{N}} \lambda_j & \text{if } i \in \mathcal{N} \\[6pt]
\lambda_i & \text{if } i \notin \mathcal{N}
\end{cases}
$$

where $\mathcal{N} = \{i : \lambda_i \leq \hat{\lambda}_+\}$. This preserves the trace of the correlation matrix ($\sum_i \tilde{\lambda}_i = \sum_i \lambda_i = N$), ensuring the diagonal remains unity after reconstruction.

The denoised correlation matrix is then reconstructed via

$$
\tilde{C} = \mathbf{V}\,\text{diag}(\tilde{\lambda}_1, \dots, \tilde{\lambda}_N)\,\mathbf{V}^\top
$$

where $\mathbf{V}$ is the matrix of eigenvectors of $\hat{C}$.

---

## 4. Detoning — Removing the Market Mode

The largest eigenvalue of a financial correlation matrix typically corresponds to the **market mode**: a collective factor driving all assets in the same direction. While genuine, this mode can obscure finer structure (e.g., sector clusters) in eigenvector analysis.

**Detoning** removes the contribution of the $k$ largest eigenvalues (usually $k = 1$):

$$
C_{\text{detoned}} = \tilde{C} - \sum_{j=1}^{k} \tilde{\lambda}_j \;\mathbf{v}_j \mathbf{v}_j^\top
$$

The diagonal is then rescaled to restore unit variances. The detoned matrix is useful for clustering and sector identification, but the full (non-detoned) denoised matrix $\tilde{C}$ is used for portfolio risk calculations.

---

## 5. Ledoit-Wolf Shrinkage

The Ledoit-Wolf estimator provides a complementary denoising approach by shrinking the sample covariance toward a structured target. The **Oracle Approximating Shrinkage** (OAS) estimator computes the optimal shrinkage intensity analytically, without cross-validation.

The shrunk estimator is

$$
\hat{\Sigma}_{\text{LW}} = (1 - \alpha)\,\hat{\Sigma} + \alpha\,\mu\,\mathbf{I}
$$

where $\mu = \frac{\text{tr}(\hat{\Sigma})}{N}$ is the grand-mean variance and $\alpha \in [0, 1]$ is the shrinkage intensity. The analytical formula for the optimal $\alpha$ minimises the expected Frobenius-norm loss $\mathbb{E}\|\hat{\Sigma}_{\text{LW}} - \Sigma\|_F^2$ and depends on moments of the sample eigenvalues:

$$
\alpha^* = \frac{(1 - 2/N)\,\hat{\phi} + \text{tr}(\hat{\Sigma}^2)}{(T + 1 - 2/N)\left[\text{tr}(\hat{\Sigma}^2) - \text{tr}(\hat{\Sigma})^2/N\right]}
$$

where $\hat{\phi} = \sum_{i,j} \text{Var}(\hat{\Sigma}_{ij})$ is estimated from the data. The intensity $\alpha^*$ is clipped to $[0, 1]$.

This estimator is well-conditioned by construction and can serve as a benchmark or be combined with the RMT-based approach.

---

## 6. Condition Number

The **condition number** of a covariance (or correlation) matrix is

$$
\kappa(\Sigma) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

A large condition number indicates near-singularity, meaning the matrix is ill-conditioned. This is critical for portfolio optimisation because:

- **Mean-variance optimisation** requires inverting $\Sigma$. When $\kappa(\Sigma)$ is large, small estimation errors in returns or covariances are amplified into extreme portfolio weights.
- **Numerical instability** increases: solvers may produce unreliable results or fail entirely.

| $\kappa(\Sigma)$ | Health |
|---|---|
| $< 100$ | Well-conditioned |
| $100 - 1{,}000$ | Moderate — proceed with caution |
| $> 1{,}000$ | Ill-conditioned — denoising essential |

Denoising via RMT or Ledoit-Wolf shrinkage dramatically reduces $\kappa$, producing more stable and investable portfolios.

---

## 7. Parametric VaR

The **parametric (Gaussian) Value-at-Risk** at confidence level $\alpha$ for a portfolio with weight vector $\mathbf{w}$, expected return vector $\boldsymbol{\mu}$, and covariance matrix $\Sigma$ is

$$
\text{VaR}_\alpha = -\left(\mathbf{w}^\top \boldsymbol{\mu} - z_\alpha \sqrt{\mathbf{w}^\top \Sigma\,\mathbf{w}}\right)
$$

where $z_\alpha = \Phi^{-1}(\alpha)$ is the quantile of the standard normal distribution. For the PRIIPs regulatory framework, the relevant confidence level is **97.5%**, giving $z_{0.975} \approx 1.96$.

Under the Gaussian assumption, VaR is fully determined by the first two moments of the portfolio return distribution. When returns exhibit skewness or heavy tails, the Cornish-Fisher expansion provides a correction.

---

## 8. Cornish-Fisher VaR

The **Cornish-Fisher expansion** adjusts the Gaussian quantile to account for non-normality. Let $S$ denote the skewness and $K$ the excess kurtosis of the portfolio return distribution. The modified quantile is

$$
z_{\text{CF}} = z + \frac{(z^2 - 1)}{6}\,S + \frac{(z^3 - 3z)}{24}\,K - \frac{(2z^3 - 5z)}{36}\,S^2
$$

where $z = z_\alpha$ is the Gaussian quantile. The Cornish-Fisher VaR is then

$$
\text{VaR}_{\text{CF}} = -\left(\mathbf{w}^\top \boldsymbol{\mu} - z_{\text{CF}} \sqrt{\mathbf{w}^\top \Sigma\,\mathbf{w}}\right)
$$

This expansion is accurate when departures from normality are moderate. It is the method prescribed by the PRIIPs regulation (EU 2017/653) for computing risk measures of structured products.

---

## 9. VaR-Equivalent Volatility (VEV)

The PRIIPs framework maps the Cornish-Fisher VaR to a **VaR-Equivalent Volatility**, defined as

$$
\text{VEV} = \frac{\sqrt{T} \cdot \text{VaR}_{\text{CF}}}{z_\alpha}
$$

where $T$ is the recommended holding period (in years) and $z_\alpha = 1.96$ for the 97.5th percentile. The VEV expresses the tail risk in units of annualised volatility, enabling comparison across products with different return distributions. It serves as the input to the SRI classification scale.

---

## 10. SRI Classification — PRIIPs MRM Scale

The **Summary Risk Indicator** (SRI) is determined by the **Market Risk Measure** (MRM) class, which is assigned based on VEV thresholds defined in EU Delegated Regulation 2017/653, Annex II:

| MRM Class | VEV Range |
|---|---|
| 1 | $\text{VEV} < 0.5\%$ |
| 2 | $0.5\% \leq \text{VEV} < 5\%$ |
| 3 | $5\% \leq \text{VEV} < 12\%$ |
| 4 | $12\% \leq \text{VEV} < 20\%$ |
| 5 | $20\% \leq \text{VEV} < 30\%$ |
| 6 | $30\% \leq \text{VEV} < 80\%$ |
| 7 | $\text{VEV} \geq 80\%$ |

The SRI ranges from 1 (lowest risk) to 7 (highest risk) and must be disclosed in the Key Information Document (KID) provided to retail investors.

---

## 11. Divergence Detection

The pipeline computes the SRI twice:

1. **Prescribed SRI** — from the raw (sample) covariance matrix, following the regulatory methodology as-is.
2. **Kernel SRI** — from the denoised covariance matrix, after RMT cleaning.

If estimation noise inflates or deflates the covariance structure, these two SRI values may diverge. The system applies a **traffic-light flagging** scheme:

| Flag | Condition | Interpretation |
|---|---|---|
| Green | Prescribed SRI $=$ Kernel SRI | Noise has negligible impact on the risk classification. |
| Amber | $|\text{Prescribed SRI} - \text{Kernel SRI}| = 1$ | Marginal divergence — the product sits near an MRM boundary. |
| Red | $|\text{Prescribed SRI} - \text{Kernel SRI}| \geq 2$ | Material divergence — estimation noise is materially distorting the risk classification. |

A red flag signals that the regulatory risk label may be misleading: either overstating risk (penalising the product commercially) or understating it (exposing investors to inadequately disclosed risk).

---

## References

1. **Marchenko, V. A. & Pastur, L. A.** (1967). "Distribution of eigenvalues for some sets of random matrices." *Matematicheskii Sbornik*, 114(4), 507--536.

2. **Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M.** (1999). "Noise Dressing of Financial Correlation Matrices." *Physical Review Letters*, 83(7), 1467--1470.

3. **Bouchaud, J.-P. & Potters, M.** (2003). *Theory of Financial Risk and Derivative Pricing: From Statistical Physics to Risk Management*. Cambridge University Press, 2nd edition.

4. **Lopez de Prado, M.** (2020). *Machine Learning for Asset Managers*. Cambridge University Press. Elements in Quantitative Finance.

5. **Ledoit, O. & Wolf, M.** (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365--411.

6. **European Commission** (2017). *Commission Delegated Regulation (EU) 2017/653* supplementing Regulation (EU) No 1286/2014 on key information documents for packaged retail and insurance-based investment products (PRIIPs).
