# Supply Chain Forecasting Pipeline

This pipeline uses statistical tests to find the best model for a wide range of products, and produces monthly demand forecasts for a duty free store at the US-Mexico border. This demand information is synthesized with supply chain information and stocking priorities to suggest ordering from three different depots. A small Streamlit page shows the forecast and actual sales for any month, and lists the ordering suggestions.

**Live demo:** https://supply-chain-forecast.streamlit.app

All sales, product and inventory data here are synthetic. The border data
is public.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_synthetic_data.py      # optional: data is already committed
pytest                                         # unit tests
streamlit run app.py                           # the page
python -m forecasting.pipeline --cutoff 2025-06 --out forecasts.csv
python -m forecasting.backtest --start 2022-03 --end 2025-06 --step 3
```

Each selection on the page reruns the forecasts for that product's depot
(about 1–3 seconds).

## Layout

```
app.py                       Streamlit page
config/settings.yaml         every threshold, service level and depot rule
data/
  border_crossings_monthly.csv   public monthly border crossings (raw, cleaned in code)
  df_store_1_daily_sales.csv     synthetic POS extract, 2019-01 to 2025-09
  products.csv                   synthetic product master
forecasting/
  data.py         loading, border cleaning, zero-filled monthly series
  diagnostics.py  statistical tests that feed model selection
  selection.py    the priority rules
  models.py       model fitting, prediction intervals, safety cap
  pipeline.py     per-SKU forecasts, CLI
  backtest.py     rolling-origin evaluation
ordering/
  demand.py       demand over any window of days, from monthly forecasts
  schedule.py     delivery calendars
  inventory.py    inventory snapshot (synthetic unless you supply one)
  recommend.py    order rules per depot priority
scripts/generate_synthetic_data.py
tests/
```

## Data

**Border crossings.** Monthly inbound bus passengers, pedestrians and
personal-vehicle passengers for one port, from the US Bureau of Transportation
Statistics border crossing dataset. Two problems are fixed in code, not
in the file: missing bus counts (Jun 2022, Oct 2023) are interpolated, and a
recording error (Oct 2023 pedestrians reported as 1,539 against ~230,000 in
neighbouring months) is flagged and interpolated. A month is flagged only if it
is both a robust-z outlier and more than 5x off its neighbours, so the real 2020
closure drop is kept.

**Sales.** 16 synthetic SKUs, daily, Jan 2019 – Sep 2025, one store
(`df_store_1`). Each SKU's demand follows one border series with an elasticity,
plus its own month-of-year pattern, trend, weekday pattern and autocorrelated
noise. The 2020 closure therefore shows up through the border data.

| Depot | SKU | Follows | Notes |
|---|---|---|---|
| Beverage & Perishable | Premium Tequila, Blended Whisky, Imported Beer | personal vehicles | December / Father's Day / summer peaks |
| | Bottled Water | pedestrians | summer peak |
| | Energy Drink | personal vehicles | growing ~18% a year |
| | Boxed Chocolates | pedestrians | December and Valentine's, 180-day shelf life |
| | Fresh Fruit Cup | pedestrians | summer, 5-day shelf life |
| Luxury | Fragrance A, Fragrance B | bus passengers | December / Mother's Day (May) |
| | Sunglasses X → Sunglasses Y | bus passengers | X declines from late 2022; Y launches Mar 2023 |
| | Designer Wallet | bus passengers | intermittent (about 1 a month) |
| Dry Goods & Tobacco | Cigarettes | pedestrians | slow decline |
| | Candy Tin, Neck Pillow | vehicles / bus | |
| | USB Cable | nothing | bought by locals and staff; control case |

## Forecasting

### Model-selection rules

The rules and thresholds from the original pipeline are kept, in the same
priority order:

1. No sales at all → zero. Fewer than 3 non-zero months → recent average.
   Sparsity > 0.27 → intermittent-demand model.
2. **Strong December** → seasonal model, with border data when available, and
   a relaxed safety cap (×1.5). This overrides everything below.
3. Strong summer → seasonal model (with border data if the border rule passes).
4. Border correlation > 0.47 → ARIMA with the border series as a regressor.
5. Autocorrelation > 0.55 → ARIMA (seasonal if seasonal strength > 0.5).
6. Seasonal strength > 0.83 → seasonal model.
7. Otherwise → exponential smoothing.
8. No sales in the last 6 months → forecast 0 (dead product).
9. Forecasts are capped by the original safety cap. December products get the
   permissive cap.

### What changed and why

| Original | Now | Why |
|---|---|---|
| Border correlation on raw levels | On first differences, and must be significant (p < 0.05). Lag-1 Granger p-value reported | Two series that both fell in 2020 correlate strongly whether or not one drives the other |
| All three border series as regressors | Only the best-correlated one | They move together; three collinear regressors on 30–80 points overfit |
| Future border = last month repeated | Same month last year, scaled by recent year-over-year change | Last month repeated ignores seasonality; plain last-year badly under-predicted the 2022 recovery |
| Autocorrelation by Pearson on shifted series | ACF plus Ljung-Box test (p < 0.05) | Tests whether the autocorrelation is real, not noise |
| Fixed SARIMA orders, e.g. (2,1,2)(1,1,1,12) | d from KPSS (ADF reported); D from STL seasonal strength ≥ 0.64; orders chosen by AICc, p+q ≤ 2, at most n/5 parameters; seasonal terms need 24+ months | The fixed order had about 7 parameters for a series with roughly 11 usable points after seasonal differencing |
| `enforce_stationarity=False` | Enforced | Unconstrained fits produced the explosive forecasts the safety checks were catching |
| Sparse → average of last 4 months | Croston with Syntetos-Boylan correction, smoothing chosen in-sample; demand class (smooth / erratic / intermittent / lumpy) reported | An average isn't designed for intermittent demand and its interval is wrong |
| Hand-written smoother with alpha = 0.3 | statsmodels ETS, parameters estimated, (A,N,N) or damped trend by AICc, model-based intervals | A fixed alpha is a guess; the old interval ignored the model |
| Negative forecast → whole-history mean | Clipped to 0 | On a declining product the whole-history mean is far too high |
| Explosive forecast → whole-history mean, silently | Last-12-month mean, and the fallback is noted in the output and on the page | Same reason, and fallbacks should be visible |
| Forecast capped but interval not | `lower ≤ forecast ≤ upper ≤ cap` always | A capped forecast could sit below its own lower bound |
| Bare `except:`, `fillna(method=...)`, one 1,400-line file, prints | Package, typed, logged, config file, tests | |

The December rule is still permissive. With only 2 Decembers of history, "the
maximum is in December" can fire on noise. It also fires on growing products,
whose latest December is naturally above their all-time average. The output
reports how many Decembers each call rested on (`seasonal_evidence_decembers`).
For the same reason, the summer rule fires on products still recovering from
2020.

### Backtest (synthetic data)

Rolling origin, cutoffs every 3 months from Mar 2022 to Jun 2025, horizons 1–3:

| | Value |
|---|---|
| MASE (scaled by in-sample seasonal-naive error) | 0.51 |
| Seasonal-naive forecast, same scale | 0.68 |
| Mean bias (units/month) | +0.9 |
| 90% interval coverage | 85% |

Coverage is below the nominal 90%, mainly because ARIMA intervals ignore
uncertainty in the estimated parameters. Weakest cases: the declining
sunglasses (the model keeps expecting a partial recovery) and the new SKU in its
first year. On your own data, rerun `python -m forecasting.backtest` and
compare against the original numbers before trusting either.

## Ordering

For each depot, the plan covers demand from the start of the forecast month
until the delivery after next: this order has to last until then. Monthly
forecasts are spread evenly over their days. Variance over a window is forecast
error, scaled with the window, plus Poisson day-to-day noise. Demand is
modelled as a negative binomial matched to that mean and variance, which
behaves correctly for slow luxury items selling 0–2 a week.

Stockout risk in the table is P(demand before the delivery after next > stock +
order). A note flags items likely to run out before this order even arrives.

### Judgment calls (all in `config/settings.yaml`)

| | Beverage & Perishable | Luxury | Dry Goods & Tobacco |
|---|---|---|---|
| Distance | 50 mi | 200 mi | 100 mi |
| Deliveries | Tue and Fri, 1-day lead | every other Wed, 4-day lead | Mon, 2-day lead |
| Truck | 6 pallets, $350 a trip | van, 2 pallets, $1,200, $25k insured value | 8 pallets, $500 |
| Priority | Keep alcohol stocked | At least one of every active SKU on hand | Lowest cost per unit delivered |
| Rule | Alcohol ordered up to the 99% demand quantile, other drinks 95%. Short-life perishables use the newsvendor quantity (margin / price) and are then cut back until expected spoilage is ≤ 3% of expected sales within shelf life. Long-life perishables (> 30 days) use 95%. When the cooler (1.5 pallets) or truck is full, non-alcohol is cut first and alcohol last. Remaining alcohol risk above 1% is flagged. | Order up to the 90% quantile, never below 1 unit. Stop reordering after 6 months without sales. SKU cycling below. Load kept under the insured value and van space. | Skip the truck if every SKU will last to the next one at 95%. If a truck is needed but the load is under 0.5 pallets, order two weeks at once, but only when shelf space allows, so the next trip can be skipped. |

**Store:** 2 pallets of overflow space, for alcohol and luxury only (never
perishables). Stock above an item's shelf maximum goes there if space is free.
Otherwise the order is trimmed.

**Luxury SKU cycling.** A SKU is phased out when all of these hold:
- it was classed Dying or Dead in 2 consecutive monthly runs;
- its last 6 months are below 80% of the same months a year earlier, so an ordinary seasonal dip doesn't count;
- a sibling in the same family is Rising, or is Stable and outselling it.

Because the Rising/Dying label flips from month to month on small counts, the
decision is sticky. Once it fires, the SKU stays phased out for up to 12 months
while it is still down year over year. A phased-out SKU is not reordered; the
page says to keep one on display, move the rest to overflow and mark it down.

### Using real data

- **Sales:** same columns as `df_store_1_daily_sales.csv` (`date, store, sku, description, category, depot, units`).
- **Products:** `products.csv` (case pack, cases per pallet, cost, price, shelf maximum, cooler flag, shelf life, family).
- **Inventory:** `data/inventory_snapshot.csv` with `asof, sku, on_hand, overflow_units, on_order`. `asof` is the first day of the month being planned. Without it, a synthetic snapshot of 40–130% of one replenishment cycle is used, and the page says so.

## Limitations

- Sales history is treated as demand. Days when an item was out of stock
  understate demand; real data should be corrected for that first.
- Monthly forecasts are spread evenly within the month. There are no
  weekday or holiday-week effects in ordering.
- Depot settings, truck sizes, costs and shelf capacities from real client could not be used. 
