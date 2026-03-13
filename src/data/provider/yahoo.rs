// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Yahoo Finance price provider.
//!
//! Uses the Yahoo Finance v8 chart API to fetch historical adjusted
//! close prices. Free, rate-limited, suitable for prototyping and
//! demonstration. Not intended for production use in regulated environments.
//!
//! # Rate limiting
//!
//! Fetches tickers sequentially with a configurable delay between requests
//! to avoid hitting rate limits.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use thiserror::Error;

/// Errors from Yahoo Finance fetching.
#[derive(Debug, Error)]
pub enum YahooError {
    /// HTTP request failed.
    #[error("HTTP error fetching {ticker}: {source}")]
    Http {
        /// The ticker that failed.
        ticker: String,
        /// The underlying reqwest error.
        source: reqwest::Error,
    },

    /// Yahoo returned an error or unexpected response.
    #[error("Yahoo API error for {ticker}: {message}")]
    Api {
        /// The ticker that failed.
        ticker: String,
        /// Error message.
        message: String,
    },

    /// No price data returned for ticker.
    #[error("no price data for {0}")]
    NoData(String),
}

/// Fetched price data for a single ticker.
#[derive(Debug, Clone)]
pub struct TickerPrices {
    /// Ticker symbol.
    pub ticker: String,
    /// Currency of the listing (e.g., "USD", "`GBp`", "EUR").
    pub currency: String,
    /// Timestamps (Unix epoch seconds).
    pub timestamps: Vec<i64>,
    /// Adjusted close prices (aligned with timestamps).
    pub prices: Vec<f64>,
}

/// Fetch historical adjusted close prices from Yahoo Finance.
///
/// # Arguments
///
/// * `tickers` — List of ticker symbols (e.g., `["AAPL", "MSFT"]`).
/// * `period_days` — Number of calendar days of history to fetch.
///
/// # Errors
///
/// Returns [`YahooError`] if any ticker fails to fetch.
pub async fn fetch_prices(
    tickers: &[String],
    period_days: u32,
) -> Result<Vec<TickerPrices>, YahooError> {
    let client = reqwest::Client::builder()
        .user_agent("regit-covariance/0.1")
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| YahooError::Http {
            ticker: "client".into(),
            source: e,
        })?;

    let now = Utc::now().timestamp();
    let period_seconds = i64::from(period_days) * 86400;
    let start = now - period_seconds;

    let mut results = Vec::with_capacity(tickers.len());

    for ticker in tickers {
        let data = fetch_single(&client, ticker, start, now).await?;
        results.push(data);

        // Rate limiting: 200ms between requests.
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    Ok(results)
}

/// Fetch a single ticker from Yahoo Finance v8 chart API.
async fn fetch_single(
    client: &reqwest::Client,
    ticker: &str,
    period1: i64,
    period2: i64,
) -> Result<TickerPrices, YahooError> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history"
    );

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| YahooError::Http {
            ticker: ticker.into(),
            source: e,
        })?;

    if !resp.status().is_success() {
        return Err(YahooError::Api {
            ticker: ticker.into(),
            message: format!("HTTP {}", resp.status()),
        });
    }

    let body: YahooResponse = resp.json().await.map_err(|e| YahooError::Http {
        ticker: ticker.into(),
        source: e,
    })?;

    let result = body
        .chart
        .result
        .into_iter()
        .next()
        .ok_or_else(|| YahooError::NoData(ticker.into()))?;

    let meta = result.meta;
    let currency = meta
        .as_ref()
        .and_then(|m| m.currency.clone())
        .unwrap_or_else(|| "USD".to_string());

    let timestamps = result.timestamp.unwrap_or_default();
    let adj_close = result
        .indicators
        .adjclose
        .and_then(|v| v.into_iter().next())
        .map(|a| a.adjclose)
        .unwrap_or_default();

    if timestamps.is_empty() || adj_close.is_empty() {
        return Err(YahooError::NoData(ticker.into()));
    }

    // Filter out None values (non-trading days).
    let mut clean_timestamps = Vec::new();
    let mut clean_prices = Vec::new();
    for (ts, price) in timestamps.iter().zip(adj_close.iter()) {
        if let Some(p) = price
            && *p > 0.0
        {
            clean_timestamps.push(*ts);
            clean_prices.push(*p);
        }
    }

    if clean_prices.is_empty() {
        return Err(YahooError::NoData(ticker.into()));
    }

    // Detect Yahoo's scaled/corrupt price data for certain mutual funds.
    // Some funds return adjclose values with inconsistent scaling. Rather than
    // silently altering data, we detect and reject corrupt series.
    if let Some(ref m) = meta {
        detect_price_corruption(&clean_prices, m, ticker)?;
    }

    // Fix GBX/GBP unit flips common in London-listed securities.
    // Yahoo sometimes reports prices in GBp (pence) and sometimes in GBP (pounds),
    // causing ~100x jumps. Detect and normalize to the dominant unit.
    fix_currency_unit_flips(&mut clean_prices);

    Ok(TickerPrices {
        ticker: ticker.to_string(),
        currency,
        timestamps: clean_timestamps,
        prices: clean_prices,
    })
}

/// Align multiple tickers to a common set of dates.
///
/// Only keeps dates where ALL tickers have data. Timestamps are normalized
/// to calendar dates (midnight UTC) before alignment, since Yahoo reports
/// different intraday timestamps for different exchanges.
///
/// Returns (dates, T × N price matrix) where T = common dates, N = tickers.
#[must_use]
pub fn align_prices(ticker_data: &[TickerPrices]) -> (Vec<i64>, nalgebra::DMatrix<f64>) {
    if ticker_data.is_empty() {
        return (Vec::new(), nalgebra::DMatrix::zeros(0, 0));
    }

    // Build a map: calendar_date → price for each ticker.
    // Normalize timestamps to midnight UTC to handle cross-exchange time differences.
    let mut all_maps: Vec<HashMap<i64, f64>> = Vec::new();
    for td in ticker_data {
        let map: HashMap<i64, f64> = td
            .timestamps
            .iter()
            .zip(td.prices.iter())
            .map(|(&ts, &price)| (normalize_to_date(ts), price))
            .collect();
        all_maps.push(map);
    }

    // Find common dates (present in all tickers).
    let mut common: Vec<i64> = all_maps[0].keys().copied().collect();
    for map in &all_maps[1..] {
        common.retain(|ts| map.contains_key(ts));
    }
    common.sort_unstable();

    let num_obs = common.len();
    let num_assets = ticker_data.len();
    let mut matrix = nalgebra::DMatrix::zeros(num_obs, num_assets);

    for (row, ts) in common.iter().enumerate() {
        for (col, map) in all_maps.iter().enumerate() {
            matrix[(row, col)] = map[ts];
        }
    }

    (common, matrix)
}

/// Normalize a Unix timestamp to midnight UTC of the same calendar date.
fn normalize_to_date(ts: i64) -> i64 {
    // Floor to the start of the day (86400 seconds per day).
    (ts / 86400) * 86400
}

/// Convert Unix timestamps to date strings for display.
#[must_use]
pub fn timestamps_to_dates(timestamps: &[i64]) -> Vec<String> {
    timestamps
        .iter()
        .filter_map(|&ts| {
            DateTime::from_timestamp(ts, 0).map(|dt| dt.date_naive().format("%Y-%m-%d").to_string())
        })
        .collect()
}

/// Resolve an ISIN to a Yahoo Finance ticker symbol using the search API.
///
/// Prefers EUR or USD-listed tickers over `GBp` (pence) listings to avoid
/// inflating volatility with FX noise from GBP-denominated price series.
/// When the ISIN search returns only a GBp/LSE ticker, performs a secondary
/// search by fund name to find EUR-listed alternatives on XETRA, Milan, etc.
///
/// # Errors
///
/// Returns [`YahooError`] if the search fails or no matching ticker is found.
pub async fn resolve_isin(isin: &str) -> Result<String, YahooError> {
    let client = reqwest::Client::builder()
        .user_agent("regit-covariance/0.1")
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| YahooError::Http {
            ticker: isin.into(),
            source: e,
        })?;

    let quotes = yahoo_search(&client, isin).await?;

    if quotes.is_empty() {
        return Err(YahooError::NoData(format!(
            "no Yahoo ticker found for ISIN {isin}"
        )));
    }

    // If the best result is on LSE (GBp), try to find a EUR-listed alternative.
    let best = pick_preferred_listing(&quotes);

    if is_gbp_listing(&best.symbol) {
        // Secondary search: fetch the fund name from the chart meta, then search
        // by name to find EUR/USD alternatives on other exchanges.
        if let Ok(name) = fetch_fund_name(&client, &best.symbol).await {
            let alt_quotes = yahoo_search(&client, &name).await.unwrap_or_default();
            if !alt_quotes.is_empty() {
                let alt_best = pick_preferred_listing(&alt_quotes);
                if !is_gbp_listing(&alt_best.symbol) {
                    tracing::info!(
                        "ISIN {isin}: preferring {sym} over {orig} (avoiding GBp FX noise)",
                        sym = alt_best.symbol,
                        orig = best.symbol
                    );
                    return Ok(alt_best.symbol.clone());
                }
            }
        }
    }

    Ok(best.symbol.clone())
}

/// Search Yahoo Finance and return the list of matching quotes.
async fn yahoo_search(
    client: &reqwest::Client,
    query: &str,
) -> Result<Vec<YahooSearchQuote>, YahooError> {
    let url = format!(
        "https://query1.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
    );

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| YahooError::Http {
            ticker: query.into(),
            source: e,
        })?;

    if !resp.status().is_success() {
        return Err(YahooError::Api {
            ticker: query.into(),
            message: format!("search HTTP {}", resp.status()),
        });
    }

    let body: YahooSearchResponse = resp.json().await.map_err(|e| YahooError::Http {
        ticker: query.into(),
        source: e,
    })?;

    Ok(body.quotes)
}

/// Fetch the long/short name of a ticker from the chart meta endpoint.
async fn fetch_fund_name(client: &reqwest::Client, ticker: &str) -> Result<String, YahooError> {
    let url =
        format!("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1d");

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| YahooError::Http {
            ticker: ticker.into(),
            source: e,
        })?;

    let body: YahooResponse = resp.json().await.map_err(|e| YahooError::Http {
        ticker: ticker.into(),
        source: e,
    })?;

    let name = body
        .chart
        .result
        .first()
        .and_then(|r| r.meta.as_ref())
        .and_then(|m| m.long_name.clone().or(m.short_name.clone()))
        .ok_or_else(|| YahooError::NoData(format!("no name found for {ticker}")))?;

    Ok(name)
}

/// Pick the preferred listing from search results, ranking by exchange:
/// XETRA (.DE) > Milan (.MI) > Paris (.PA) > other EUR > USD > `GBp` (.L).
fn pick_preferred_listing(quotes: &[YahooSearchQuote]) -> &YahooSearchQuote {
    quotes
        .iter()
        .min_by_key(|q| exchange_rank(&q.symbol))
        .unwrap_or(&quotes[0])
}

/// Lower rank = more preferred. Avoids `GBp` listings that add FX noise.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn exchange_rank(symbol: &str) -> u8 {
    if symbol.ends_with(".DE") {
        0 // XETRA — EUR
    } else if symbol.ends_with(".MI") {
        1 // Milan — EUR
    } else if symbol.ends_with(".PA") || symbol.ends_with(".AS") {
        2 // Paris / Amsterdam — EUR
    } else if symbol.ends_with(".SW") {
        3 // Swiss Exchange — CHF, close to EUR
    } else if !symbol.contains('.') {
        4 // US listing — USD
    } else if symbol.ends_with(".L") {
        6 // LSE — GBp, avoid
    } else {
        5 // Other exchanges
    }
}

/// Check if a ticker is likely listed in `GBp` (London Stock Exchange).
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn is_gbp_listing(symbol: &str) -> bool {
    symbol.ends_with(".L")
}

/// Detect corrupt/inconsistently-scaled price data from Yahoo Finance.
///
/// Some mutual funds (especially on Frankfurt with `0P` prefix symbols) return
/// `adjclose` values with mixed scaling — some at the correct NAV level while
/// others are scaled by ~100,000× or more. Rather than silently altering data,
/// this function detects the corruption and returns an error so the user is
/// informed that the data source is unreliable for this instrument.
fn detect_price_corruption(
    prices: &[f64],
    meta: &YahooMeta,
    ticker: &str,
) -> Result<(), YahooError> {
    let Some(market_price) = meta.regular_market_price else {
        return Ok(());
    };

    if market_price <= 0.0 || prices.is_empty() {
        return Ok(());
    }

    let corrupt_count = prices.iter().filter(|&&p| p / market_price > 100.0).count();

    if corrupt_count > 0 {
        return Err(YahooError::Api {
            ticker: ticker.into(),
            message: format!(
                "Yahoo returned corrupt price data: {corrupt_count}/{} data points are \
                 scaled ~{:.0}× above the actual market price ({market_price:.2}). \
                 This is a known Yahoo Finance bug for some mutual funds. \
                 Try using the direct ticker symbol instead of the ISIN.",
                prices.len(),
                prices
                    .iter()
                    .map(|p| p / market_price)
                    .fold(0.0_f64, f64::max),
            ),
        });
    }

    Ok(())
}

/// Fix currency unit flips in price series.
///
/// London-listed securities on Yahoo Finance sometimes alternate between
/// GBX (pence) and GBP (pounds), causing ~100× jumps. This detects
/// consecutive price ratios near 100 (or 1/100) and normalizes all prices
/// to the dominant unit.
fn fix_currency_unit_flips(prices: &mut [f64]) {
    if prices.len() < 2 {
        return;
    }

    // Detect which prices are "low" (likely GBP) vs "high" (likely GBX).
    // Use median as the reference level.
    let mut sorted = prices.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    if median <= 0.0 {
        return;
    }

    // Any price that is ~100× smaller than the median is in a different unit.
    // Threshold: if price < median / 20, it's likely in pounds instead of pence
    // (or vice versa). We multiply/divide by 100 to normalize.
    let low_threshold = median / 20.0;
    let high_threshold = median * 20.0;

    for price in prices.iter_mut() {
        if *price < low_threshold && *price > 0.0 {
            // Price is ~100× too low — multiply by 100.
            *price *= 100.0;
        } else if *price > high_threshold {
            // Price is ~100× too high — divide by 100.
            *price /= 100.0;
        }
    }
}

// ── Yahoo Finance JSON response structures ──────────────────────────

#[derive(Debug, Deserialize)]
struct YahooSearchResponse {
    quotes: Vec<YahooSearchQuote>,
}

#[derive(Debug, Deserialize)]
struct YahooSearchQuote {
    symbol: String,
    #[serde(default)]
    #[allow(dead_code)]
    exchange: String,
}

#[derive(Debug, Deserialize)]
struct YahooResponse {
    chart: YahooChart,
}

#[derive(Debug, Deserialize)]
struct YahooChart {
    result: Vec<YahooResult>,
}

#[derive(Debug, Deserialize)]
struct YahooResult {
    meta: Option<YahooMeta>,
    timestamp: Option<Vec<i64>>,
    indicators: YahooIndicators,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YahooMeta {
    currency: Option<String>,
    #[serde(default)]
    long_name: Option<String>,
    #[serde(default)]
    short_name: Option<String>,
    #[serde(default)]
    regular_market_price: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct YahooIndicators {
    adjclose: Option<Vec<YahooAdjClose>>,
}

#[derive(Debug, Deserialize)]
struct YahooAdjClose {
    adjclose: Vec<Option<f64>>,
}
