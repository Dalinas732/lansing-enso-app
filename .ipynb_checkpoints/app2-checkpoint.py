# app.py
import streamlit as st
import pandas as pd
import numpy as np
from statistics import mode
import plotly.graph_objects as go
import plotly.express as px

# ---------- Data loader ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/full_data_encoded.csv", low_memory=False)

try:
    df_raw = load_data()
    st.success("✅ Loaded data/full_data_encoded.csv successfully.")
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()


st.title("Temperature Patterns and ENSO Phases (1960–2024)")

st.markdown("""
The following three plots explore how **ENSO (El Niño–Southern Oscillation)** affects Michigan’s temperatures.
ENSO is a recurring climate pattern driven by temperature shifts in the tropical Pacific Ocean that influences weather worldwide.
- **El Niño** phases (negative values) often bring **warmer** winters to the Great Lakes region.
- **La Niña** phases (positive values) tend to bring **cooler** conditions.
Together, these plots visualize how local temperatures vary under different ENSO phases:
1. **Absolute Monthly High Temperatures (1960–2024)** — Shows seasonal cycles across all years.
2. **Temperature Anomalies by ENSO Phase** — Highlights deviations from average patterns.
3. **Winter Temperature Distributions (Violin Plot)** — Compares the range and variability of winter highs across ENSO categories.
""")
def plot_temp_anom(df_raw):
    # =========================
    # Load & basic preparation
    # =========================
    # Keep only what's needed for temps
    df = df_raw[["Year", "Month", "high"]].dropna().copy()
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    
    # Order months for x-axis
    month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
    
    # If duplicates exist per Year×Month, average them
    df = df.groupby(["Year", "Month"], as_index=False, observed=False)["high"].mean()
    
    # =========================
    # Build monthly 1960–2000 baseline and anomalies
    # =========================
    baseline_df = df[(df["Year"] >= 1960) & (df["Year"] <= 2000)]
    baseline_by_month = (baseline_df.groupby("Month", observed=False)["high"]
                         .mean().reindex(month_order))
    
    years_all = sorted(df["Year"].unique())
    series_by_year_abs = {}   # absolute temps
    series_by_year_dev = {}   # anomalies (Δ vs baseline)
    complete_years = []
    for y in years_all:
        s = df[df["Year"] == y].set_index("Month").reindex(month_order)["high"]
        if s.notna().all():  # only keep complete years (12 months)
            series_by_year_abs[y] = s.values.tolist()
            series_by_year_dev[y] = (s - baseline_by_month).values.tolist()
            complete_years.append(y)
    years = complete_years
    
    # =========================
    # Compute ENSO code per Season_Year (DJF)
    # =========================
    enso_df = df_raw[["Year", "Month", "ENSO_encoded"]].dropna().copy()
    enso_df["ENSO_encoded"] = pd.to_numeric(enso_df["ENSO_encoded"], errors="coerce")
    enso_df["Month"] = pd.Categorical(enso_df["Month"], categories=month_order, ordered=True)
    
    winter_months = ["DEC", "JAN", "FEB"]
    enso_winter = enso_df[enso_df["Month"].isin(winter_months)].copy()
    
    # Season_Year: DEC counts toward the next year's winter
    enso_winter["Season_Year"] = enso_winter.apply(
        lambda r: int(r["Year"]) + 1 if r["Month"] == "DEC" else int(r["Year"]),
        axis=1
    )
    
    def safe_mode(vals):
        vals = [v for v in vals if pd.notna(v)]
        if not vals:
            return np.nan
        try:
            return mode(vals)
        except Exception:
            # tie-breaker: closest to neutral, then by value
            return sorted(vals, key=lambda x: (abs(x), x))[0]
    
    # One ENSO code per Season_Year (mode over DJF)
    enso_per_year = (enso_winter.groupby("Season_Year")["ENSO_encoded"]
                     .apply(lambda s: safe_mode(list(s.values)))
                     .to_dict())
    
    # Map each plotted calendar year to its DJF Season_Year ENSO
    enso_for_plotted_year = {y: enso_per_year.get(y, np.nan) for y in years}
    
    # =========================
    # Colors (fully opaque, saturated)
    # =========================
    enso_order = [-4, -3, -2, -1, 0, 1, 2, 3]
    enso_color_map = {
        -4: "#8B0000",  # dark red
        -3: "#C62828",
        -2: "#E53935",
        -1: "#FF1744",
         0: "#2E7D32",  # dark gray for neutral highlight
         1: "#1E88E5",
         2: "#1565C0",
         3: "#0D47A1"   # dark blue
    }
    
    # =========================
    # Build the figure (anomalies)
    # =========================
    fig = go.Figure()
    
    # 1) Background gray lines
    for y in years:
        fig.add_trace(go.Scatter(
            x=month_order, y=series_by_year_dev[y],
            mode="lines",
            line=dict(color="#E0E0E0", width=1),
            hoverinfo="skip",
            showlegend=False,
            name=f"{y} (bg)"
        ))
    
    # 2) Highlight layer (one per year), hidden by default
    for y in years:
        enso_code = enso_for_plotted_year[y]
        color = enso_color_map.get(int(enso_code), "#424242") if pd.notna(enso_code) else "#424242"
        fig.add_trace(go.Scatter(
            x=month_order, y=series_by_year_dev[y],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=6, color=color, line=dict(width=0.5, color="black")),
            opacity=1.0,
            name=str(y),
            text=[f"Year {y} | ENSO {enso_code}"]*12,
            hovertemplate="%{text}<br>Month: %{x}<br>ΔHigh: %{y:.1f}°F<extra></extra>",
            visible=False,
            showlegend=False
        ))
    
    # Reference zero line
    fig.add_trace(go.Scatter(
        x=month_order, y=[0]*12,
        mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        name="Baseline (0)",
        hoverinfo="skip",
        showlegend=True
    ))
    
    # Visibility helper
    n = len(years)
    def visibility_for_enso(selected_enso):
        vis = [True]*n + [False]*n + [True]  # gray + highlights off + zero line
        if selected_enso is None:
            return vis
        for i, y in enumerate(years):
            val = enso_for_plotted_year[y]
            if pd.notna(val) and int(val) == selected_enso:
                vis[n + i] = True
        return vis
    
    # =========================
    # Slider steps (with descriptive ENSO labels)
    # =========================
    
    enso_labels = {
        -4: "Very Strong El Niño",
        -3: "Strong El Niño",
        -2: "Moderate El Niño",
        -1: "Weak El Niño",
         0: "Neutral",
         1: "Weak La Niña",
         2: "Moderate La Niña",
         3: "Strong La Niña"
    }
    
    steps = [{
        "method": "update",
        "label": "All Years",
        "args": [
            {"visible": visibility_for_enso(None)},
            {"title": "Monthly High Temperature Anomalies — All Years (Gray)"}
        ]
    }]
    
    for e in enso_order:
        label = enso_labels.get(e, str(e))
        steps.append({
            "method": "update",
            "label": label,
            "args": [
                {"visible": visibility_for_enso(e)},
                {"title": f"Monthly High Temperature Anomalies — Highlight: {label}"}
            ]
        })
    
    # Axes (use helpers to avoid bad property paths)
    fig.update_xaxes(
        title_text="Month",
        title_font=dict(color="#000000"),
        tickfont=dict(color="#000000"),
        showgrid=True,
        gridcolor="#e0e0e0",
        zerolinecolor="#000000",
        linecolor="#000000",
        categoryorder="array",
        categoryarray=month_order
    )
    
    fig.update_yaxes(
        title_text="Δ High Temperature (°F) vs 1960–2000",
        title_font=dict(color="#000000"),
        tickfont=dict(color="#000000"),
        showgrid=True,
        gridcolor="#e0e0e0",
        zerolinecolor="#000000",
        linecolor="#000000"
    )
    
    # Layout (bright white)
    fig.update_layout(
        title="Monthly High Temperature Anomalies (Δ vs 1960–2000) — Highlight by ENSO Phase",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        width=1150,
        height=650,
        sliders=[{
            "active": 0,
            "y": -0.06,
            "pad": {"t": 30, "b": 10},
            "len": 0.98,
            "currentvalue": {"prefix": "Filter ENSO: ", "visible": True},
            "steps": steps
        }]
    )
    return fig

#### Figure 2
def plot_absolute_temperature(df_raw):
    # =========================
    # Load & basic preparation
    # =========================
    df = df_raw[["Year", "Month", "high"]].dropna()
    df["high"] = pd.to_numeric(df["high"], errors="coerce")

    # Month order for the x-axis
    month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)

    df = df.groupby(["Year", "Month"], as_index=False, observed=False)["high"].mean()

    # =========================
    # ENSO processing
    # =========================
    enso_df = df_raw[["Year", "Month", "ENSO_encoded"]].dropna()
    enso_df["ENSO_encoded"] = pd.to_numeric(enso_df["ENSO_encoded"], errors="coerce")
    enso_df["Month"] = pd.Categorical(enso_df["Month"], categories=month_order, ordered=True)

    winter_months = ["DEC", "JAN", "FEB"]
    enso_winter = enso_df[enso_df["Month"].isin(winter_months)].copy()
    enso_winter["Season_Year"] = enso_winter.apply(
        lambda r: int(r["Year"]) + 1 if r["Month"] == "DEC" else int(r["Year"]), axis=1
    )

    def safe_mode(vals):
        vals = [v for v in vals if pd.notna(v)]
        if not vals:
            return np.nan
        try:
            return mode(vals)
        except Exception:
            return sorted(vals, key=lambda x: (abs(x), x))[0]

    enso_per_year = (enso_winter.groupby("Season_Year")["ENSO_encoded"]
                     .apply(lambda s: safe_mode(list(s.values)))
                     .to_dict())

    # =========================
    # Build time series
    # =========================
    years_all = sorted(df["Year"].unique())
    series_by_year = {}
    complete_years = []
    for y in years_all:
        s = df[df["Year"] == y].set_index("Month").reindex(month_order)["high"]
        if s.notna().all():
            series_by_year[y] = s.values.tolist()
            complete_years.append(y)

    years = complete_years
    enso_for_plotted_year = {y: enso_per_year.get(y, np.nan) for y in years}

    # =========================
    # Colors and slider labels
    # =========================
    enso_order = [-4, -3, -2, -1, 0, 1, 2, 3]
    enso_color_map = {
        -4: "#8B0000",
        -3: "#C62828",
        -2: "#E53935",
        -1: "#FF1744",
         0: "#228B22",  # green for neutral
         1: "#1E88E5",
         2: "#1565C0",
         3: "#0D47A1"
    }

    enso_labels = {
        -4: "Very Strong El Niño",
        -3: "Strong El Niño",
        -2: "Moderate El Niño",
        -1: "Weak El Niño",
         0: "Neutral",
         1: "Weak La Niña",
         2: "Moderate La Niña",
         3: "Strong La Niña"
    }

    # =========================
    # Plot
    # =========================
    fig = go.Figure()

    # Background gray
    for y in years:
        fig.add_trace(go.Scatter(
            x=month_order, y=series_by_year[y],
            mode="lines",
            line=dict(color="#E0E0E0", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Highlight layer
    for y in years:
        enso_code = enso_for_plotted_year[y]
        color = enso_color_map.get(int(enso_code), "#424242") if pd.notna(enso_code) else "#424242"
        fig.add_trace(go.Scatter(
            x=month_order, y=series_by_year[y],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=6, color=color, line=dict(width=0.5, color="black")),
            opacity=1.0,
            name=str(y),
            text=[f"Year {y} | {enso_labels.get(int(enso_code), 'Neutral')}"]*12,
            hovertemplate="%{text}<br>Month: %{x}<br>High: %{y:.1f}°F<extra></extra>",
            visible=False,
            showlegend=False
        ))

    # Visibility helper
    n = len(years)
    def visibility_for_enso(selected_enso):
        vis = [True]*n + [False]*n
        if selected_enso is None:
            return vis
        for i, y in enumerate(years):
            val = enso_for_plotted_year[y]
            if pd.notna(val) and int(val) == selected_enso:
                vis[n + i] = True
        return vis

    steps = []
    steps.append({
        "method": "update",
        "label": "All Years",
        "args": [{"visible": visibility_for_enso(None)}]
    })
    for e in enso_order:
        steps.append({
            "method": "update",
            "label": enso_labels[e],
            "args": [{"visible": visibility_for_enso(e)}]
        })

    fig.update_layout(
        title="Average Monthly High Temperatures (1960–2024)",
        xaxis=dict(title="Month", categoryorder="array", categoryarray=month_order),
        yaxis=dict(title="Average High Temperature (°F)"),
        template="plotly_white",
        width=1150,
        height=650,
        sliders=[{
            "active": 0,
            "y": -0.08,
            "pad": {"t": 30, "b": 10},
            "len": 0.98,
            "steps": steps
        }]
    )

    return fig



#### Violin PLots
def plot_violin_enso(df):
    import plotly.express as px
    import pandas as pd

    # To handle winter seasons properly (Dec belongs to the *next year's* winter)
    df["Season_Year"] = df.apply(
        lambda row: row["Year"] + 1 if row["Month"] == "DEC" else row["Year"], axis=1
    )

    # Keep only winter months (December, January, February)
    winter_months = ["DEC", "JAN", "FEB"]
    df_winter = df[df["Month"].isin(winter_months)]

    # Average only the high temps per DJF season, keeping ENSO constant
    winter_avg = (
        df_winter.groupby("Season_Year", as_index=False)
          .agg({
              "high": "mean",
              "ENSO_encoded": "first"   # ENSO is same across DJF, so just take the first value
          })
    )

    # Define order and color map
    order = [-4, -3, -2, -1, 0, 1, 2, 3]
    winter_avg["ENSO_encoded"] = pd.Categorical(
        winter_avg["ENSO_encoded"],
        categories=order,
        ordered=True
    )

    enso_map = {
        -4: "#b2182b", -3: "#d6604d", -2: "#f4a582", -1: "#fddbc7",
         0: "#f0f0f0",  1: "#d1e5f0",  2: "#92c5de",  3: "#2166ac"
    }
    enso_map = {**enso_map, **{str(k): v for k, v in enso_map.items()}}

    # Create violin plot
    fig = px.violin(
        winter_avg,
        x="ENSO_encoded",
        y="high",
        color="ENSO_encoded",
        category_orders={"ENSO_encoded": order},
        color_discrete_map=enso_map,
        box=True,
        points="all",
        hover_data={"high":":.1f", "Season_Year": True, "ENSO_encoded": False}
    )

    # Make violins thicker, more visible, and fully opaque
    fig.update_traces(
        width=0.9,
        meanline_visible=True,
        opacity=1.0,         # full opacity for clarity
        pointpos=0.0,        # points centered inside
        jitter=0.05,         # small horizontal spread to avoid overlap
        marker=dict(size=6, line=dict(width=0.5, color="black"))
    )
    # Bright, readable layout
    fig.update_layout(
        template="plotly_white",
        violingap=0.05,
        violinmode="overlay",
        width=1000,
        height=700,
        title="Average Winter (DJF) High Temperatures vs ENSO Phase",
        font=dict(size=14, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(font=dict(color="black", size=12))
    )
    
    # X axis: show ENSO phase names instead of numbers, and make labels dark
    fig.update_xaxes(
        title_text="ENSO Phase",
        tickmode="array",
        tickvals=[-4, -3, -2, -1, 0, 1, 2, 3],
        ticktext=[
            "Very Strong El Niño",
            "Strong El Niño",
            "Moderate El Niño",
            "Weak El Niño",
            "Neutral",
            "Weak La Niña",
            "Moderate La Niña",
            "Strong La Niña"
        ],
        tickfont=dict(color="black", size=12),
        title_font=dict(color="black", size=14)
    )

    # Y axis: dark labels/title
    fig.update_yaxes(
        title_text="Average Winter High Temperature (°F)",
        tickfont=dict(color="black", size=12),
        title_font=dict(color="black", size=14)
    )

    return fig




# ---- Absolute Temperature Plot ----
st.markdown("## 🌡️ Absolute Temperature by Month (1960–2024)")
st.markdown("""
This chart shows **actual observed average high temperatures** (not anomalies)
for every month from 1960–2024.  
Use the slider to highlight years associated with different **ENSO phases**.
""")


fig2 = plot_absolute_temperature(df_raw)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Use the slider to highlight specific ENSO phases and see how absolute temperatures respond.")


st.set_page_config(page_title="Monthly High Temp Anomalies", layout="wide", initial_sidebar_state="expanded")
# Force light theme
st.markdown("""
### 🌡️ Monthly High Temperature Anomalies — ENSO Phase Comparison

This visualization shows how **monthly high temperature anomalies** (°F relative to the 1960–2024 baseline) vary by **ENSO phase**  
(El Niño / La Niña / Neutral). Use the slider below to highlight specific ENSO categories and explore their influence on temperature trends.

""")
fig1 = plot_temp_anom(df_raw)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Use the slider to highlight years by ENSO phase.")

# ---- Violin Plot ----
st.markdown("## 🎻 Winter High Temperatures by ENSO Phase (Violin Plot)")
st.markdown("""
This visualization shows how **average winter (DJF) high temperatures** vary across
different **ENSO phases**.  
Each violin’s width reflects how frequently temperatures occur at different levels.
""")

fig3 = plot_violin_enso(df_raw)
st.plotly_chart(fig3, use_container_width=True)
