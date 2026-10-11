# Protocol fixed before viewing results

Question: how does next-hour hourly bike-demand forecasting behave when recent demand reports are unavailable, and does a preset outage-aware training recipe help?

Run date: 11 October 2026. Public historical data, not current operational demand.

Use UCI Bike Sharing hour.csv, DOI https://doi.org/10.24432/C5W894. Reindex onto a complete hourly grid. Preserve missing counts as NaN; do not treat them as zeros. Exclude rows without an observed target, and the first 168 hours from training.

Three expanding chronological folds: train strictly before 1 July / 1 August / 1 September 2012; evaluate the following calendar month. Each test timestamp appears in one fold. Models refit once per fold. Parameters are preset, without test-based selection. The forecast is for y[t] at the end of t-1; actual observations earlier in a test month are allowed only at later forecast origins.

Use calendar hour/day-of-week/month/year and strictly past demand lags 1,2,3,24,168 plus the mean and available fraction of the previous 24 hours. Exclude current weather and current demand components casual/registered, which reveal the target. Calendar values are known at the forecast origin.

At every test origin simulate k=0,1,3,6 missing recent hourly reports. Mask ALL dependent recent lags and recompute the rolling mean and availability from unmasked history. Earlier reports recover; this is a rolling reporting delay, not a permanent outage for the entire test month. Natural gaps remain even at k=0.

Compare four methods on identical observed test rows: ordinary HistGradientBoostingRegressor, outage-aware HGB, calendar-only HGB, and a seasonal fallback (168h, then 24h, then training median). HGB: Poisson loss, 150 iterations, 15 leaves, learning rate 0.08, min leaf 30, L2 1, early stopping disabled. The ordinary and outage-aware models have identical rows, features and estimator budgets. Calendar-only has fewer features. Seasonal fallback does not train a tree.

Outage-aware training replaces features for each training row with one preset simulated mask, drawn from 0/1/3/6h with probabilities 1/2,1/6,1/6,1/6. No extra training rows. Seeds 42,43,44 correspond to folds. Model seed 42. Threads limited to two; no speed comparisons are claimed.

Primary endpoint: pooled MAE in rentals/hour over the three test months. Report RMSE, per-fold metrics, all four scenarios and all methods. No cherry-picked winner, uncertainty bars, SOTA claim or generalization beyond this dataset. Each fold has one masking draw; variation across folds is available but is not a confidence interval.

Record downloaded hashes, complete predictions, environment, configuration and checks. Publish negative results too. Checks cover causal dependencies, missingness propagation, hidden observation invariance, grid validation, deterministic masks and matched test rows.
