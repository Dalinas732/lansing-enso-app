# app.py
import streamlit as st
import pandas as pd
import numpy as np
from statistics import mode
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Michigan Temperature & ENSO Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# --- Custom Light Theme CSS ---
st.markdown("""
<style>
/* Warm beige page background */
.stApp {
    background-color: #f7f3e9;
    color: #1a1a1a !important;
}

/* Override Streamlit's dark theme font colors */
html, body, [class*="st-"], [data-testid="stMarkdownContainer"], p, span, div {
    color: #1a1a1a !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #1a1a1a !important;
    font-weight: 700;
}

/* Tabs styling */
[data-baseweb="tab-list"] {
    background-color: #efe9dc;
    border-radius: 8px;
    padding: 6px;
}
[data-baseweb="tab"] {
    color: #333 !important;
    font-weight: 600;
}
[data-baseweb="tab"][aria-selected="true"] {
    background-color: #f9f5ea !important;
    border-radius: 8px;
}

/* Comfortable layout padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

/* Paragraph text */
p {
    font-size: 1.1rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
### 🌎 Lansing Temperature Trends and ENSO Impacts (1960–2024)

This dashboard explores long-term temperature patterns in **Lansing, Michigan**, using historical weather records from 1960 to 2024.  
It highlights how **average temperatures** and the **number of freeze days** have changed over time, and examines how these patterns relate to  
**ENSO (El Niño–Southern Oscillation)** phases — a recurring climate pattern in the Pacific Ocean that influences seasonal weather across North America.
""")

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
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1a1a1a"),
        xaxis=dict(gridcolor="rgba(180,180,180,0.3)"),
        yaxis=dict(gridcolor="rgba(180,180,180,0.3)")
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

# Annual Montly temp
def plot_annual_temp(df):
    # Keep only what's needed
    df = df[["Year", "Month", "high"]].dropna()
    
    # Month order (ensure correct x-axis order)
    month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
    
    # If duplicates exist per Year×Month, average them (retain current pandas behavior)
    df = df.groupby(["Year", "Month"], as_index=False, observed=False)["high"].mean()
    
    # Keep only years that have all 12 months (comment out this block to allow gaps)
    years_all = sorted(df["Year"].unique())
    years = []
    for y in years_all:
        sub = df[df["Year"] == y].set_index("Month").reindex(month_order)
        if sub["high"].notna().all():
            years.append(y)
    
    # Build series: year -> list of highs in month order
    series_by_year = {
        y: df[df["Year"] == y].set_index("Month").reindex(month_order)["high"].tolist()
        for y in years
    }
    
    # --- Figure ---
    fig = go.Figure()
    
    # 1) Add the HIGHLIGHT trace FIRST (this will be updated by frames)
    default_year = max(years)
    fig.add_trace(
        go.Scatter(
            x=month_order,
            y=series_by_year[default_year],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=6),
            name=str(default_year),
            text=[default_year]*12,
            hovertemplate="Year: %{text}<br>Month: %{x}<br>High: %{y:.1f}°F<extra></extra>",
            showlegend=False,
        )
    )
    
    # 2) Add gray background lines (static; hover off to reduce clutter)
    for y in years:
        fig.add_trace(
            go.Scatter(
                x=month_order,
                y=series_by_year[y],
                mode="lines",
                line=dict(color="lightgray", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    
    # 3) Frames that update ONLY the first trace (index 0 = highlight)
    frames = []
    for y in years:
        frames.append(
            go.Frame(
                name=str(y),
                data=[go.Scatter(x=month_order, y=series_by_year[y], text=[y]*12)],
                traces=[0],  # update the highlight trace only
            )
        )
    fig.frames = frames
    
    # Slider steps
    steps = []
    for y in years:
        steps.append({
            "args": [[str(y)],
                     {"frame": {"duration": 0, "redraw": True},
                      "mode": "immediate",
                      "transition": {"duration": 0}}],
            "label": str(y),
            "method": "animate"
        })
    
    # Layout
    fig.update_layout(
        title="Monthly Average Highs by Year (All Years in Gray; Use Slider to Highlight One)",
        xaxis=dict(title="Month", categoryorder="array", categoryarray=month_order),
        yaxis=dict(title="Average High Temperature (°F)"),
        template="plotly_white",
        width=1100,
        height=600,
        sliders=[{
            "active": years.index(default_year),
            "y": -0.06,
            "pad": {"t": 30, "b": 10},
            "len": 0.98,
            "currentvalue": {"prefix": "Year: ", "visible": True},
            "steps": steps
        }],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 1.02, "y": 1.12,
            "buttons": [
                {"label": "Play",
                 "method": "animate",
                 "args": [None, {"frame": {"duration": 350, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 150}}]},
                {"label": "Pause",
                 "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]}
            ]
        }]
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1a1a1a"),
        xaxis=dict(gridcolor="rgba(180,180,180,0.3)"),
        yaxis=dict(gridcolor="rgba(180,180,180,0.3)")
        )
    return fig
def plot_annual_anom(df):
    month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
    df = df.groupby(["Year","Month"], as_index=False, observed=False)["high"].mean()
    
    # Baseline 1960–2000 (monthly means)
    baseline = (df[(df["Year"] >= 1960) & (df["Year"] <= 2000)]
                .groupby("Month", observed=False)["high"]
                .mean().reindex(month_order))
    
    # Keep complete years
    years_all = sorted(df["Year"].unique())
    years = []
    by_year = {}
    for y in years_all:
        s = df[df["Year"] == y].set_index("Month").reindex(month_order)["high"]
        if s.notna().all():
            years.append(y)
            by_year[y] = s
    
    # Build deviation series: each year minus the baseline per month
    dev_by_year = {y: (by_year[y] - baseline).tolist() for y in years}
    
    # --- Figure with anomalies ---
    fig = go.Figure()
    
    # Highlight trace FIRST (will be updated by frames)
    default_year = max(years)
    fig.add_trace(go.Scatter(
        x=month_order, y=dev_by_year[default_year],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=6),
        name=str(default_year),
        text=[default_year]*12,
        hovertemplate="Year: %{text}<br>Month: %{x}<br>ΔHigh: %{y:.1f}°F<extra></extra>",
        showlegend=False
    ))
    
    # Gray background lines
    for y in years:
        fig.add_trace(go.Scatter(
            x=month_order, y=dev_by_year[y],
            mode="lines",
            line=dict(color="lightgray", width=1),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Zero line (reference)
    fig.add_trace(go.Scatter(
        x=month_order, y=[0]*12,
        mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        name="Baseline (0)",
        hoverinfo="skip",
        showlegend=True
    ))
    
    # Frames (update only the first trace = highlight)
    frames = []
    for y in years:
        frames.append(go.Frame(
            name=str(y),
            data=[go.Scatter(x=month_order, y=dev_by_year[y], text=[y]*12)],
            traces=[0]
        ))
    fig.frames = frames
    
    # Slider
    steps = [{
        "args": [[str(y)], {"frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}}],
        "label": str(y),
        "method": "animate"
    } for y in years]
    
    fig.update_layout(
        title="Monthly High Temperature Anomalies vs 1960–2024 Baseline (All Years in Gray; Slider Highlights One)",
        xaxis=dict(title="Month", categoryorder="array", categoryarray=month_order),
        yaxis=dict(title="Δ High Temperature (°F) vs 1960–2000"),
        template="plotly_white",
        width=1100, height=600,
        sliders=[{
            "active": years.index(default_year),
            "y": -0.06, "len": 0.98,
            "currentvalue": {"prefix": "Year: ", "visible": True},
            "steps": steps
        }]
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1a1a1a"),
        xaxis=dict(gridcolor="rgba(180,180,180,0.3)"),
        yaxis=dict(gridcolor="rgba(180,180,180,0.3)")
    )
    return fig


#  Time series of Energy Data
def time_series(df):
    figs = []
    # 1. Prepare the time series
    # -------------------------------
    # Assuming df has columns: "Month", "residential million kilowatthours"
    
    df["Month"] = pd.to_datetime(df["Month"], format="%b %Y")
    df = df.sort_values("Month")                # make sure it's chronological
    df.set_index("Month", inplace=True)
    
    y = df["residential million kilowatthours"].astype(float)
    
    # -------------------------------
    # 2. Seasonal decomposition
    # -------------------------------
    result = seasonal_decompose(
        y,
        model="additive",
        period=12,                 # monthly data, yearly seasonality
        extrapolate_trend="freq"
    )
    
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid
    deseasonalized = y - seasonal
    
    df_clean = pd.DataFrame({
        "Original": y,
        "Trend": trend,
        "Seasonal": seasonal,
        "Residual": resid,
        "Deseasonalized": deseasonalized,
    })
    
    # 3. Decomposition plots
    # -------------------------------
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(df_clean.index, df_clean["Original"])
    plt.title("Original Michigan Residential Energy Demand")
    plt.ylabel("M kWh")
    plt.grid(True, alpha=0.6)
    
    plt.subplot(4, 1, 2)
    plt.plot(df_clean.index, df_clean["Trend"])
    plt.title("Trend")
    plt.ylabel("M kWh")
    plt.grid(True, alpha=0.6)
    
    plt.subplot(4, 1, 3)
    plt.plot(df_clean.index, df_clean["Seasonal"])
    plt.title("Seasonal Component")
    plt.ylabel("M kWh")
    plt.grid(True, alpha=0.6)
    
    plt.subplot(4, 1, 4)
    plt.plot(df_clean.index, df_clean["Residual"])
    plt.title("Residual (y - trend - seasonal)")
    plt.ylabel("M kWh")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.6)
    
    plt.tight_layout()
    st.pyplot(plt.gcf())      # ⬅️ instead of plt.show()

    # -------------------------------
    # 4. Winter-month data + Winter_Year
    # -------------------------------
    winter_df = df_clean[df_clean.index.month.isin([12, 1, 2])].copy()
    
    # Dec belongs to the *next* winter; Jan/Feb to current year
    winter_df["Winter_Year"] = np.where(
        winter_df.index.month == 12,
        winter_df.index.year + 1,
        winter_df.index.year
    )
    
    
    # 5. Winter anomalies (z-score plot)
    mu = winter_df["Deseasonalized"].mean()
    sigma = winter_df["Deseasonalized"].std()
    
    winter_df["Winter_Anomaly"] = (winter_df["Deseasonalized"] - mu) / sigma
    
    threshold = 1.2
    strong_winter_months = winter_df[winter_df["Winter_Anomaly"] > threshold]
    # -------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(winter_df.index, winter_df["Winter_Anomaly"], "-o", label="Winter Anomaly")
    plt.axhline(threshold, linestyle="--", color="red",
                label=f"Strong Winter Threshold ({threshold:.1f}σ)")
    plt.title("Winter Energy Demand Anomalies (Deseasonalized & Standardized)")
    plt.ylabel("Anomaly (Z-score)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())      # ⬅️ instead of plt.show()
    
    
    # 6. Winter severity by true winter year
    winter_severity = winter_df.groupby("Winter_Year")["Deseasonalized"].mean()
    # -------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(winter_severity.index, winter_severity.values, "-o")
    plt.title("Winter Severity Index (Mean Deseasonalized Demand per Winter)")
    plt.xlabel("Winter Year (Dec–Feb grouped)")
    plt.ylabel("Mean Deseasonalized Demand (M kWh)")
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt.gcf())      # ⬅️ instead of plt.show()
    
    
    # 7. Highlight strong winter months on original series
    # -------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_clean.index, df_clean["Original"], color="lightgray", label="Original Demand")
    plt.scatter(
        strong_winter_months.index,
        strong_winter_months["Original"],
        color="red",
        s=60,
        label="Unusually Strong Winter Month"
    )
    plt.title("Original Demand with Strong Winter Months Highlighted")
    plt.ylabel("Million kilowatthours")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())      # ⬅️ instead of plt.show()

### Machine Learning Methood 1
def ml_1(df):
    """
    Machine Learning Model #1:
    Predict winter anomaly class from ENSO index using a Random Forest.
    Displays a discrete-color scatter plot in Streamlit.
    """

    st.subheader("Machine Learning Model 1: ENSO → Winter Severity Prediction")

    # ----------------------------------------------------
    # 1) Build winter_summary from df
    # ----------------------------------------------------
    # Expect df_clean or similar: must have columns:
    # ["Winter_Year", "ENSO_encoded", "Anomaly_Class"]
    # If your df is named differently, adjust the selections.

    winter_summary = df.copy()

    class_mapping = {
        "-2: Far Below Normal": -2,
        "-1: Below Normal": -1,
        "0: Normal": 0,
        "+1: Above Normal": 1,
        "+2: Far Above Normal": 2
    }

    if 'Anomaly_Class_Encoded' not in winter_summary.columns:
        winter_summary['Anomaly_Class_Encoded'] = winter_summary['Anomaly_Class'].map(class_mapping)

    # ----------------------------------------------------
    # 2) Prepare data for ML
    # ----------------------------------------------------
    X = winter_summary[['ENSO_encoded']].copy()
    y = winter_summary['Anomaly_Class_Encoded'].copy()

    # Drop missing rows
    mask = X['ENSO_encoded'].notna() & y.notna()
    X = X[mask]
    y = y[mask]
    winter_summary = winter_summary[mask].copy()

    # ----------------------------------------------------
    # 3) Train/test split
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # ----------------------------------------------------
    # 4) Train Random Forest
    # ----------------------------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)

    # ----------------------------------------------------
    # 5) Predict for all winters
    # ----------------------------------------------------
    winter_summary['RF_Predicted_Class'] = rf_model.predict(X)

    # ----------------------------------------------------
    # 6) Visualization
    # ----------------------------------------------------
    unique_classes = sorted(winter_summary['RF_Predicted_Class'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(8, 4))

    for cls in unique_classes:
        subset = winter_summary[winter_summary['RF_Predicted_Class'] == cls]
        ax.scatter(
            subset['ENSO_encoded'],
            subset['Anomaly_Class_Encoded'],
            color=color_map[cls],
            label=f"Demand Level {cls}",
            s=120,
            alpha=0.85,
            edgecolors='black'
        )
    
    ax.set_title(
        "True Anomaly Class vs ENSO\nColored by Predicted Demand Level",
        fontsize=14, weight='bold'
    )
    ax.set_xlabel("ENSO Encoded (El Niño → La Niña)", fontsize=12)
    
    # 🔥 Make the y-label highly visible
    ax.set_ylabel("True Anomaly Class (Encoded)", fontsize=12, labelpad=10)
    
    ax.grid(alpha=0.4)
    ax.legend(title="Predicted Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # ---- FIX FOR STREAMLIT ----
    fig.tight_layout()
    plt.subplots_adjust(left=0.15)   # ⬅️ Reserve space for left labels
    
    st.pyplot(fig)


# --- Tabs at the top ---
time_series_dashboard ,tab_dashboard, trends, tab_notes = st.tabs(["Forecasting Energy Demand","📊 ENSO Analysis", "📝 Trends","Notes and Methods"])

with time_series_dashboard:
    st.markdown("""
    The following plots are Time Series Analysis removing the seasonailty in the annual Michigan energy demand
    """)
    df_ts = pd.read_csv("data/Retail_sales_of_electricity_monthly_res.csv", skiprows=4)
    time_series(df_ts)
    st.markdown("""
    Using Random Forest Classifer to predict the energy demand based on ENSO alone. The Demand Level goes from below average to above average on the scale -2 to +2
    """)
    winter_summary_df = pd.read_csv("data/winter_summary_with_ENSO.csv", low_memory=False)
    ml_1(winter_summary_df)

with tab_dashboard:
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
    # ===== Everything you already built goes INSIDE this block =====
    # (Put your top overview, then Absolute, Anomaly, and Violin sections here)

    # Top overview
    st.title("Temperature Patterns and ENSO Phases (1960–2024)")
    st.markdown("""
    The following three plots explore how **ENSO (El Niño–Southern Oscillation)** affects Michigan’s temperatures.
    ENSO is a recurring climate pattern driven by temperature shifts in the tropical Pacific Ocean that influences weather worldwide.
    - **El Niño** phases often bring **warmer** winters to the Great Lakes region.
    - **La Niña** phases tend to bring **cooler** conditions.
    """)

    # --- Absolute Temperature plot (your function/section) ---
    st.markdown("## 🌡️ Absolute Temperature by Month (1960–2024)")
    st.markdown("""
    This chart shows **actual average monthly high temperatures** (not anomalies) for each year.
    Use the slider to highlight years associated with different **ENSO phases**.
    """)
    fig_abs = plot_absolute_temperature(df_raw)     # <- your helper
    st.plotly_chart(fig_abs, use_container_width=True)

    st.markdown("---")  # thin line divider
    st.markdown("<br>", unsafe_allow_html=True)  # small vertical gap

    
    # --- Anomaly plot (your existing 'fig' code that builds the anomaly figure) ---
    st.markdown("## 🌊 ENSO Temperature Anomalies by Year")
    st.markdown("""
    This plot shows **anomalies** (deviations from 1960–2000 monthly averages).
    Use the slider to highlight a specific ENSO phase.
    """)
    # Make sure you create `fig` (the anomaly figure) BEFORE this line:
    fig2 = plot_temp_anom(df_raw)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Use the slider to highlight years by ENSO phase.")
    st.markdown("---")  # thin line divider
    st.markdown("<br>", unsafe_allow_html=True)  # small vertical gap

    # --- Violin plot ---
    st.markdown("## 🎻 Winter High Temperatures by ENSO Phase (Violin Plot)")
    st.markdown("""
    Average **winter (DJF)** high temperatures grouped by ENSO phase.
    Each violin shows the distribution and variability; points are individual winters.
    """)
    fig_violin = plot_violin_enso(df_raw)   # or plot_violin_enso(df_raw) depending on your function
    st.plotly_chart(fig_violin, use_container_width=True)

    
    st.markdown("## ENSO Phase Variance")

    #st.subheader("Temperature Trends for each month")

    # Load & resize while keeping aspect ratio
    img = Image.open("images/ENSO_ocs.PNG")
    target_width = 1600
    h = int(img.height * (target_width / img.width))
    img_small = img.resize((target_width, h))
    
    # Center it on the page
    left, mid, right = st.columns([1,3,1])
    with mid:
        st.image(img_small, caption="ENSO Phase Over Time.")




with trends:
    st.title("Temperature Patterns (1960–2024)")
    st.markdown("""
    This page explores long-term climate patterns in Lansing, Michigan, focusing on two key indicators: overall temperature trends and the number of
    freeze days per year. By examining how average temperatures have shifted over time alongside the frequency of days below freezing, we can better
    understand the region’s changing seasonal patterns and the broader signals of climate variability and warming in the Great Lakes area.
    """)

    st.markdown("## 📈 Lansing Temperature Trends (static images)")

    st.subheader("Temperature Trends for each month")

    # Load & resize while keeping aspect ratio
    img = Image.open("images/all_year_trends.PNG")
    target_width = 1600
    h = int(img.height * (target_width / img.width))
    img_small = img.resize((target_width, h))
    
    # Center it on the page
    left, mid, right = st.columns([1,3,1])
    with mid:
        st.image(img_small, caption="Average monthly temperature over time.")
    

    
    st.subheader("Number of Freeze Days Each Month Per Year")

    # Load & resize while keeping aspect ratio
    img = Image.open("images/freeze_days_months.PNG")
    target_width = 1600
    h = int(img.height * (target_width / img.width))
    img_small = img.resize((target_width, h))
    
    # Center it on the page
    left, mid, right = st.columns([1,3,1])
    with mid:
        st.image(img_small, caption="Number of freeze days per year each month (≤ 32°F)")

    
    st.subheader("Number of Freeze Days Per Year")
    st.caption("Below is a visualization showing how the number of ≤32°F days has changed over time.")
    
    # Load & resize while keeping aspect ratio
    img = Image.open("images/freeze_days.PNG")
    target_width = 1500
    h = int(img.height * (target_width / img.width))
    img_small = img.resize((target_width, h))
    
    # Center it on the page
    left, mid, right = st.columns([1,3,1])
    with mid:
        st.image(img_small, caption="Number of freeze days per year (≤ 32°F)")
     


    # --- Absolute Temperature plot (your function/section) ---
    st.markdown("## 🌡️ Absolute Temperature by Month (1960–2024)")
    st.markdown("""
    This chart shows **actual average monthly high temperatures** (not anomalies) for each year.
    Use the slider to highlight different years*.
    """)
    fig = plot_annual_temp(df_raw)     # <- your helper
    st.plotly_chart(fig, use_container_width=True)

    # --- Absolute Temperature plot (your function/section) ---
    st.markdown("## 🌡️ Temperature Anomaly by Month (1960–2024)")
    st.markdown("""
    This chart shows anomalies for each year.
    Use the slider to highlight different years
    """)
    fig = plot_annual_anom(df_raw)     # <- your helper
    st.plotly_chart(fig, use_container_width=True)


with tab_notes:
    # Put assignment notes, methodology, references, or TODOs here
    st.markdown("""
    **Notes / Methods**
    Data source: (1) Lansing Weather Station (2) El Nino-Southern Oscillation (ENSO) Data set (3) Michigan Energy database
    
    This app explores lansing weather trends, how the ENSO phase affects its winters. And if it's possible to forecast energy demand purely based on ENSO prediction since this is forecasted months in advance before the season.
    
    I combined theses three data sources and spliced them to be the same dimensions. Unfortunetly the energy demand data spans the last 20 years so only 20 datapoints. Statistics are low but for proof of concept I still deploy machine learning models.

    """)
