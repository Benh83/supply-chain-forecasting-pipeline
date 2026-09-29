"""Streamlit app: forecast chart plus ordering suggestions.

    streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from forecasting.data import complete_series
from forecasting.pipeline import forecast_many, load_all
from ordering import recommend
from ordering.inventory import load_inventory

st.set_page_config(page_title="Forecast and ordering", layout="centered")

cfg, monthly, border, products = load_all()
last_month = monthly["yearmonth"].max()


@st.cache_data(show_spinner=False, max_entries=64)
def depot_forecasts(depot: str, cutoff: str):
    skus = list(products.index[products["depot"] == depot])
    return {f.sku: f for f in forecast_many(skus, pd.Timestamp(cutoff), cfg, monthly, border)}


st.subheader("Demand forecast and ordering")

left, right = st.columns(2)
sku = left.selectbox("Product", list(products.index),
                     format_func=lambda s: f"{products.loc[s, 'description']} ({s})")
months = pd.date_range("2020-01-01", last_month, freq="MS")[::-1]
default = list(months).index(last_month - pd.DateOffset(months=3))
cutoff = right.selectbox("Forecast from", months, index=default,
                         format_func=lambda m: m.strftime("%b %Y"),
                         help="Last month of history. The forecast covers the next three months.")
show_actual = st.checkbox("Show actual demand", value=True)

depot = products.loc[sku, "depot"]
with st.spinner("Running forecasts..."):
    forecasts = depot_forecasts(depot, cutoff.strftime("%Y-%m-%d"))

if sku not in forecasts:
    st.write(f"No sales history for this product up to {cutoff:%b %Y}.")
else:
    fc = forecasts[sku]
    start = cutoff - pd.DateOffset(months=35)
    hist = fc.history[fc.history.index >= start]
    frames = [pd.DataFrame({"month": hist.index, "units": hist.values, "series": "History"})]
    fut = pd.DataFrame({
        "month": [cutoff + pd.DateOffset(months=h.horizon) for h in fc.horizons],
        "units": [h.forecast for h in fc.horizons],
        "lower": [h.lower for h in fc.horizons],
        "upper": [h.upper for h in fc.horizons],
    })
    # join the forecast line to the last observed month
    joint = pd.concat([pd.DataFrame({"month": [cutoff], "units": [hist.iloc[-1]]}), fut[["month", "units"]]])
    frames.append(joint.assign(series="Forecast"))

    actual = complete_series(monthly, sku, last_month, start=fc.history.index[0])
    actual = actual[(actual.index > cutoff) & (actual.index <= fut["month"].max())]
    if show_actual and not actual.empty:
        joint = pd.concat([pd.Series([hist.iloc[-1]], index=[cutoff]), actual])
        frames.append(pd.DataFrame({"month": joint.index, "units": joint.values, "series": "Actual"}))

    lines = pd.concat(frames)
    colors = alt.Scale(domain=["History", "Forecast", "Actual"],
                       range=["#333333", "#1f5fbf", "#d9480f"])
    base = alt.Chart(lines).encode(
        x=alt.X("month:T", title=None),
        y=alt.Y("units:Q", title="Units per month"),
        color=alt.Color("series:N", scale=colors, legend=alt.Legend(title=None, orient="top")),
    )
    band = alt.Chart(fut).mark_area(opacity=0.15, color="#1f5fbf").encode(
        x="month:T", y="lower:Q", y2="upper:Q")
    chart = (band + base.mark_line(point=True, strokeWidth=1.5)).properties(height=320)
    st.altair_chart(chart, width="stretch")

    st.write(f"Model: {fc.horizons[0].model}. Selected because: {'; '.join(fc.choice.reasons)}. "
             f"Recent trend: {fc.trend}.")
    table = pd.DataFrame({
        "Month": fut["month"].dt.strftime("%b %Y"),
        "Forecast": fut["units"].round(0).astype(int),
        "90% low": fut["lower"].round(0).astype(int),
        "90% high": fut["upper"].round(0).astype(int),
    })
    if show_actual:
        table["Actual"] = [int(actual[m]) if m in actual.index else None for m in fut["month"]]
    st.dataframe(table, hide_index=True)
    notes = {h.note for h in fc.horizons if h.note}
    for note in notes:
        st.write(f"Note: {note}")

# --- Ordering ---------------------------------------------------------------
inventory, synthetic = load_inventory(cfg["paths"]["inventory"], products, monthly, cutoff, cfg)
plan = recommend(depot, cutoff, forecasts, products, inventory, monthly, cfg)

st.subheader(f"Ordering: {plan.name}")
st.write(f"Planning as of {plan.asof:%b %d, %Y}. Order by {plan.order_by:%a %b %d} "
         f"for delivery {plan.delivery:%a %b %d}. Next delivery {plan.next_delivery:%a %b %d}.")
if plan.decision == "Order":
    st.write(f"Load: {plan.pallets:.2f} of {plan.truck_pallets:g} pallets ({plan.fill_pct:.0f}%), "
             f"{plan.units:,} units, ${plan.load_value:,.0f} at cost. "
             f"Trip ${plan.trip_cost:,.0f} (${plan.cost_per_unit:.2f} per unit).")
else:
    st.write(f"Decision: {plan.decision}.")
for note in plan.notes:
    st.write(note)
st.dataframe(plan.lines.rename(columns=lambda c: c.replace("_", " ").capitalize()),
             hide_index=True)
if synthetic:
    st.caption("Inventory is a synthetic snapshot. Add data/inventory_snapshot.csv to use real counts.")
