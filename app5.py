import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Flight Delays", layout="wide")
st.title("Flight Delays — Visual Exploration and Route/Seasonal Insights")

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, low_memory=False)

    # Parse date and derive calendar helpers
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
        df["Year"] = df["FL_DATE"].dt.year
        df["Month"] = df["FL_DATE"].dt.month
        df["DayOfWeek"] = df["FL_DATE"].dt.dayofweek + 1  # 1..7

    # Route helper
    if "ROUTE" not in df.columns and {"ORIGIN", "DEST"}.issubset(df.columns):
        df["ROUTE"] = df["ORIGIN"].astype(str).str.strip() + "-" + df["DEST"].astype(str).str.strip()

    # Departure hour
    if "DEP_HOUR" not in df.columns:
        if "DEP_TIME_STR" in df.columns:
            def hour_from_str(s):
                try:
                    return int(str(s).split(":")[0])
                except:
                    return np.nan
            df["DEP_HOUR"] = df["DEP_TIME_STR"].apply(hour_from_str)
        elif "DEP_TIME" in df.columns:
            def hhmm_to_hour(v):
                try:
                    v = int(float(v))
                    return v // 100
                except:
                    return np.nan
            df["DEP_HOUR"] = df["DEP_TIME"].apply(hhmm_to_hour)

    # Normalize reason, coerce numerics
    if "DELAY_REASON" in df.columns:
        df["DELAY_REASON"] = df["DELAY_REASON"].fillna("No Delay").replace({"": "No Delay"})
    for c in ["ARR_DELAY", "DEP_DELAY", "DISTANCE", "TAXI_OUT", "TAXI_IN", "CANCELLED", "ELAPSED_TIME", "CRS_ELAPSED_TIME", "AIR_TIME"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

DATA_PATH = "flight_delay_3m_clean_single_reason.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Missing {DATA_PATH}. Place the cleaned CSV in this folder.")
    st.stop()

df = load_data(DATA_PATH)

# ---------------- Sidebar: Filters ----------------
with st.sidebar:
    st.header("Filters")

    # Date range
    if "FL_DATE" in df.columns:
        min_d, max_d = df["FL_DATE"].min(), df["FL_DATE"].max()
        date_range = st.date_input("Date range", (min_d, max_d))
    else:
        date_range = None

    # Year and Month filters (checkbox groups)
    years = sorted([int(y) for y in df.get("Year", pd.Series(dtype=int)).dropna().unique().tolist()])
    if years:
        colY1, colY2 = st.columns(2)
        all_y = colY1.checkbox("All years", value=True)
        none_y = colY2.checkbox("No years", value=False, key="none_years")
        if none_y:
            sel_years = []
        elif all_y:
            sel_years = years
        else:
            cols = st.columns(2)
            sel_years = []
            for i, y in enumerate(years):
                if cols[i % 2].checkbox(str(y), value=False, key=f"Y_{y}"):
                    sel_years.append(y)
    else:
        sel_years = []

    months = list(range(1, 13)) if "Month" in df.columns else []
    if months:
        colM1, colM2 = st.columns(2)
        all_m = colM1.checkbox("All months", value=True)
        none_m = colM2.checkbox("No months", value=False, key="none_months")
        if none_m:
            sel_months = []
        elif all_m:
            sel_months = months
        else:
            cols = st.columns(2)
            sel_months = []
            for i, m in enumerate(months):
                if cols[i % 2].checkbox(f"M{m}", value=False, key=f"M_{m}"):
                    sel_months.append(m)
    else:
        sel_months = []

    st.markdown("---")

    # Airlines checkboxes
    airlines = sorted(df.get("AIRLINE", pd.Series(dtype=str)).dropna().unique().tolist())
    colA1, colA2 = st.columns(2)
    all_a = colA1.checkbox("All airlines", value=True)
    none_a = colA2.checkbox("No airlines", value=False, key="none_airlines")
    if none_a:
        sel_airlines = []
    elif all_a:
        sel_airlines = airlines
    else:
        cols = st.columns(2)
        sel_airlines = []
        for i, a in enumerate(airlines):
            if cols[i % 2].checkbox(a, value=False, key=f"A_{a}"):
                sel_airlines.append(a)

    # Reasons checkboxes
    reasons = sorted(df.get("DELAY_REASON", pd.Series(dtype=str)).dropna().unique().tolist())
    colR1, colR2 = st.columns(2)
    all_r = colR1.checkbox("All reasons", value=True)
    none_r = colR2.checkbox("No reasons", value=False, key="none_reasons")
    if none_r:
        sel_reasons = []
    elif all_r:
        sel_reasons = reasons
    else:
        cols = st.columns(2)
        sel_reasons = []
        for i, r in enumerate(reasons):
            if cols[i % 2].checkbox(r, value=False, key=f"R_{r}"):
                sel_reasons.append(r)

    show_no_delay = st.checkbox("Include 'No Delay'", value=True)

    st.markdown("---")

    # Hour/Distance sliders
    if "DEP_HOUR" in df.columns and df["DEP_HOUR"].notna().any():
        hmin = int(max(0, np.nanmin(df["DEP_HOUR"])))
        hmax = int(min(23, np.nanmax(df["DEP_HOUR"])))
    else:
        hmin, hmax = 0, 23
    sel_hour_range = st.slider("Departure Hour Range", min_value=0, max_value=23, value=(hmin, hmax))

    if "DISTANCE" in df.columns and df["DISTANCE"].notna().any():
        dmin, dmax = float(df["DISTANCE"].min()), float(df["DISTANCE"].max())
        sel_dist = st.slider("Distance Range (miles)", min_value=0.0, max_value=max(1.0, dmax), value=(max(0.0, dmin), dmax))
    else:
        sel_dist = (0.0, 1e9)

    # Optional routes subset select (if too many routes)
    routes = sorted(df.get("ROUTE", pd.Series(dtype=str)).dropna().unique().tolist())
    routes_display = routes[:500]
    sel_routes = st.multiselect("Routes (subset)", routes_display, default=[])

# ---------------- Apply Filters ----------------
f = df.copy()
if date_range and "FL_DATE" in f.columns and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["FL_DATE"] >= start) & (f["FL_DATE"] <= end)]
if sel_years and "Year" in f.columns:
    f = f[f["Year"].isin(sel_years)]
if sel_months and "Month" in f.columns:
    f = f[f["Month"].isin(sel_months)]
if sel_airlines and "AIRLINE" in f.columns:
    f = f[f["AIRLINE"].isin(sel_airlines)]
if sel_reasons and "DELAY_REASON" in f.columns:
    f = f[f["DELAY_REASON"].isin(sel_reasons)]
if not show_no_delay and "DELAY_REASON" in f.columns:
    f = f[f["DELAY_REASON"] != "No Delay"]
if "DEP_HOUR" in f.columns:
    f = f[(f["DEP_HOUR"] >= sel_hour_range[0]) & (f["DEP_HOUR"] <= sel_hour_range[1])]
if "DISTANCE" in f.columns:
    f = f[(f["DISTANCE"] >= sel_dist[0]) & (f["DISTANCE"] <= sel_dist[1])]
if sel_routes and "ROUTE" in f.columns:
    f = f[f["ROUTE"].isin(sel_routes)]

# ---------------- KPIs ----------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Flights", f"{len(f):,}")
with k2:
    st.metric("On-Time %", f"{(f['ARR_DELAY'] <= 0).mean() * 100:.1f}%" if len(f) else "0%")
with k3:
    avg_arr = f["ARR_DELAY"].mean() if "ARR_DELAY" in f.columns else np.nan
    st.metric("Avg ARR Delay", f"{avg_arr:.1f}" if pd.notna(avg_arr) else "NA")
with k4:
    canc_rate = f["CANCELLED"].mean() if "CANCELLED" in f.columns else np.nan
    st.metric("Cancellation %", f"{canc_rate * 100:.2f}%" if pd.notna(canc_rate) else "NA")

# ---------------- Tabs ----------------
tab_overview, tab_time, tab_airlines, tab_routes, tab_maps, tab_correlation = st.tabs(
    ["Overview", "Time & Season", "Airlines", "Routes", "Maps", "Correlation"]
)

# -------- Overview --------
with tab_overview:
    left, mid, right = st.columns(3)
    # Delay reason share
    if "DELAY_REASON" in f.columns and not f.empty:
        share = f["DELAY_REASON"].value_counts().reset_index(name="Count")
        if "index" in share.columns:
            share = share.rename(columns={"index": "DELAY_REASON"})
        left.plotly_chart(px.pie(share, names="DELAY_REASON", values="Count", hole=0.35,
                                 title="Delay Reason Share"), use_container_width=True)
        mid.plotly_chart(px.bar(share.head(10), x="DELAY_REASON", y="Count",
                                title="Top Reasons (Count)"), use_container_width=True)
    # Top airlines bar
    if "AIRLINE" in f.columns and not f.empty:
        airlines_ct = f["AIRLINE"].value_counts().reset_index(name="Flights")
        if "index" in airlines_ct.columns:
            airlines_ct = airlines_ct.rename(columns={"index": "AIRLINE"})
        right.plotly_chart(px.bar(airlines_ct.head(10), x="AIRLINE", y="Flights",
                                  title="Top Airlines"), use_container_width=True)

# -------- Time & Season --------
with tab_time:
    if {"FL_DATE", "ARR_DELAY"}.issubset(f.columns) and not f.empty:
        daily_avg = f.groupby("FL_DATE")["ARR_DELAY"].mean().reset_index()
        st.plotly_chart(px.line(daily_avg, x="FL_DATE", y="ARR_DELAY",
                                title="Daily Avg Arrival Delay"), use_container_width=True)
    if {"Month", "ARR_DELAY"}.issubset(f.columns) and not f.empty:
        mavg = f.groupby("Month")["ARR_DELAY"].mean().reset_index()
        st.plotly_chart(px.line(mavg, x="Month", y="ARR_DELAY", markers=True,
                                title="Monthly Avg Arrival Delay"), use_container_width=True)
    if {"DEP_HOUR", "ARR_DELAY"}.issubset(f.columns) and not f.empty:
        hour_avg = f.groupby("DEP_HOUR")["ARR_DELAY"].mean().reset_index()
        st.plotly_chart(px.line(hour_avg, x="DEP_HOUR", y="ARR_DELAY", markers=True,
                                title="Avg Delay by Hour"), use_container_width=True)
    if {"DayOfWeek", "ARR_DELAY"}.issubset(f.columns) and not f.empty:
        dow = f.groupby("DayOfWeek")["ARR_DELAY"].mean().reset_index()
        dow["Day"] = dow["DayOfWeek"].map({1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"})
        st.plotly_chart(px.bar(dow, x="Day", y="ARR_DELAY", title="Avg Delay by Day"),
                        use_container_width=True)
    if {"DEP_HOUR", "DayOfWeek", "ARR_DELAY"}.issubset(f.columns) and not f.empty:
        heat = f.groupby(["DEP_HOUR", "DayOfWeek"])["ARR_DELAY"].mean().reset_index()
        heat["Day"] = heat["DayOfWeek"].map({1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"})
        st.plotly_chart(px.density_heatmap(heat, x="DEP_HOUR", y="Day", z="ARR_DELAY",
                                           color_continuous_scale="RdYlGn_r", title="Hour × Day Avg Delay"),
                        use_container_width=True)
    # Stacked area over time by reason
    if {"FL_DATE", "DELAY_REASON"}.issubset(f.columns) and not f.empty:
        series = f.groupby(["FL_DATE", "DELAY_REASON"]).size().reset_index(name="Flights")
        st.plotly_chart(px.area(series, x="FL_DATE", y="Flights", color="DELAY_REASON",
                                title="Reasons Over Time (Stacked Area)"),
                        use_container_width=True)
    # 100% stacked reason mix by hour
    if {"DEP_HOUR", "DELAY_REASON"}.issubset(f.columns) and not f.empty:
        hour_reason = pd.crosstab(f["DEP_HOUR"], f["DELAY_REASON"], normalize="index") * 100
        st.plotly_chart(px.bar(hour_reason, x=hour_reason.index, y=list(hour_reason.columns),
                               barmode="relative", title="Reason Mix by Hour (100% Stacked)"),
                        use_container_width=True)
    # ECDF of ARR_DELAY
    if "ARR_DELAY" in f.columns and f["ARR_DELAY"].notna().any():
        edf = f[["ARR_DELAY"]].dropna().sort_values("ARR_DELAY")
        edf["ECDF"] = np.arange(1, len(edf) + 1) / len(edf)
        st.plotly_chart(px.line(edf, x="ARR_DELAY", y="ECDF", title="ECDF of Arrival Delay"),
                        use_container_width=True)

# -------- Airlines --------
with tab_airlines:
    sel_topN = st.slider("Top N airlines", min_value=5, max_value=20, value=10, step=1)
    if "AIRLINE" in f.columns and not f.empty:
        topA = f["AIRLINE"].value_counts().head(sel_topN).index
        sub = f[f["AIRLINE"].isin(topA)].copy()
        if {"AIRLINE", "DELAY_REASON", "ARR_DELAY"}.issubset(sub.columns) and not sub.empty:
            piv = sub.groupby(["AIRLINE", "DELAY_REASON"])["ARR_DELAY"].mean().reset_index()
            st.plotly_chart(px.bar(piv, x="AIRLINE", y="ARR_DELAY", color="DELAY_REASON",
                                   barmode="stack", title="Avg Delay by Airline & Reason"),
                            use_container_width=True)
            st.plotly_chart(px.violin(sub, x="AIRLINE", y="ARR_DELAY", points="outliers",
                                      box=True, color="AIRLINE", title="Violin: Arr Delay by Airline"),
                            use_container_width=True)
            st.plotly_chart(px.box(sub, x="AIRLINE", y="ARR_DELAY", color="DELAY_REASON",
                                   points="all", title="Box: Arr Delay by Airline & Reason"),
                            use_container_width=True)
            # Parallel categories: Airline → Reason → Status
            sub["STATUS"] = np.where(sub["ARR_DELAY"] > 0, "Delayed", "On Time")
            pc = px.parallel_categories(sub, dimensions=["AIRLINE", "DELAY_REASON", "STATUS"],
                                        color=sub["ARR_DELAY"], color_continuous_scale="RdYlGn_r",
                                        title="Parallel Categories: Airline → Reason → Status")
            st.plotly_chart(pc, use_container_width=True)

# -------- Routes --------
with tab_routes:
    # Ensure ROUTE exists
    if "ROUTE" not in f.columns and {"ORIGIN", "DEST"}.issubset(f.columns):
        f["ROUTE"] = f["ORIGIN"].astype(str).str.strip() + "-" + f["DEST"].astype(str).str.strip()

    if "ROUTE" in f.columns and not f.empty:
        rc = f["ROUTE"].value_counts().reset_index(name="Flights")
        if "index" in rc.columns:
            rc = rc.rename(columns={"index": "ROUTE"})
        if not rc.empty:
            st.plotly_chart(px.bar(rc.head(15), x="Flights", y="ROUTE", orientation="h",
                                   title="Top Routes by Flights"), use_container_width=True)

    if {"ORIGIN", "DEST", "ARR_DELAY"}.issubset(f.columns) and not f.empty:
        pivot = f.pivot_table(index="ORIGIN", columns="DEST", values="ARR_DELAY", aggfunc="mean")
        if pivot.size:
            st.plotly_chart(px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                                      title="Avg Delay by Origin-Dest"), use_container_width=True)
    else:
        st.info("Routes require ORIGIN, DEST, ARR_DELAY columns.")

# -------- Maps --------
with tab_maps:
    m1, m2 = st.columns(2)

    # State choropleth (no lat/lon needed)
    if "STATE" not in f.columns:
        if "ORIGIN_CITY" in f.columns:
            f["STATE"] = f["ORIGIN_CITY"].str.extract(r",\s*([A-Z]{2})$")
        else:
            f["STATE"] = np.nan

    state_avg = f.groupby("STATE", dropna=True)["ARR_DELAY"].mean().reset_index()
    if "STATE" in state_avg.columns and not state_avg.empty:
        m1.plotly_chart(px.choropleth(state_avg, locations="STATE", locationmode="USA-states",
                                      color="ARR_DELAY", scope="usa", color_continuous_scale="RdYlGn_r",
                                      title="Avg Arrival Delay by State"), use_container_width=True)
    else:
        m1.info("No state codes available for choropleth.")

    # Airport bubble map (requires LAT/LON)
    if {"LAT", "LON"}.issubset(f.columns):
        df_map = f.dropna(subset=["LAT", "LON"]).copy()
        if not df_map.empty:
            air_stats = (df_map.groupby(["ORIGIN", "LAT", "LON", "DELAY_REASON"])
                               .agg(avg_arr=("ARR_DELAY", "mean"), flights=("ARR_DELAY", "size"))
                               .reset_index())
            m2.plotly_chart(px.scatter_geo(air_stats, lat="LAT", lon="LON",
                                           color="DELAY_REASON", size="flights", size_max=20,
                                           hover_name="ORIGIN", scope="usa", projection="albers usa",
                                           title="Airport Delays by Reason (Bubble Size = Flights)"),
                            use_container_width=True)
        else:
            m2.info("LAT/LON present but no rows after filters.")
    else:
        m2.info("LAT/LON not found. Merge an airport reference (IATA → LAT/LON) to enable bubble map.")

    st.markdown("Optional: Route lines map (requires airports_ref.csv with IATA → LAT/LON).")
    with st.expander("Show route lines code (optional)"):
        st.code("""
# Prepare routes and join coordinates, then render geo lines
routes = (f.groupby(["ORIGIN","DEST"])
            .agg(flights=("ARR_DELAY","size"), avg_arr=("ARR_DELAY","mean"))
            .reset_index()
            .sort_values("flights", ascending=False)
            .head(100))
air_ref = pd.read_csv("airports_ref.csv")  # Columns: IATA, LAT, LON
air_o = air_ref.rename(columns={"IATA":"ORIGIN","LAT":"OLAT","LON":"OLON"})
air_d = air_ref.rename(columns={"IATA":"DEST","LAT":"DLAT","LON":"DLON"})
routes = routes.merge(air_o[["ORIGIN","OLAT","OLON"]], on="ORIGIN", how="left") \\
               .merge(air_d[["DEST","DLAT","DLON"]], on="DEST",  how="left") \\
               .dropna(subset=["OLAT","OLON","DLAT","DLON"])
fig = go.Figure()
for _, r in routes.iterrows():
    fig.add_trace(go.Scattergeo(
        lon=[r["OLON"], r["DLON"]],
        lat=[r["OLAT"], r["DLAT"]],
        mode="lines",
        line=dict(width=max(1, r["flights"]/routes["flights"].max()*5), color="crimson"),
        hovertext=f'{r["ORIGIN"]}→{r["DEST"]} | Flights: {r["flights"]} | Avg: {r["avg_arr"]:.1f}m',
        hoverinfo="text"
    ))
fig.update_geos(scope="usa", projection_type="albers usa", showcountries=True)
fig.update_layout(title="Top Routes by Flights (Line Width)")
fig.show()
        """, language="python")

# -------- Correlation --------
with tab_correlation:
    cand = ["ARR_DELAY","DEP_DELAY","TAXI_OUT","TAXI_IN","DISTANCE","ELAPSED_TIME","CRS_ELAPSED_TIME","AIR_TIME"]
    corr_cols = [c for c in cand if c in f.columns]
    if corr_cols:
        num = f[corr_cols].select_dtypes(include=np.number).dropna()
        if num.shape[1] >= 2 and not num.empty:
            corr = num.corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                      title="Correlation Heatmap (numeric features)"),
                            use_container_width=True)
        else:
            st.info("Correlation needs at least two numeric columns with data after filters.")
    else:
        st.info("No numeric columns available for correlation.")

# -------- Extra: Distribution tab (optional, uncomment to add a tab) --------
# tab_dist = st.tabs(["Distribution"])[0]
# with tab_dist:
#     if "ARR_DELAY" in f.columns and not f.empty:
#         st.plotly_chart(px.histogram(f, x="ARR_DELAY", nbins=60, color="DELAY_REASON",
#                                      title="Arrival Delay Histogram"),
#                         use_container_width=True)
#     if {"DISTANCE","ARR_DELAY"}.issubset(f.columns) and not f.empty:
#         sample = f.sample(min(30000, len(f)), random_state=42)
#         st.plotly_chart(px.density_heatmap(sample, x="DISTANCE", y="ARR_DELAY",
#                                            nbinsx=40, nbinsy=40, color_continuous_scale="Viridis",
#                                            title="Distance vs Arrival Delay (2D Histogram)"),
#                         use_container_width=True)
#     if "TAXI_OUT" in f.columns and f["TAXI_OUT"].notna().any():
#         st.plotly_chart(px.histogram(f, x="TAXI_OUT", nbins=40, title="Taxi-Out Time"),
#                         use_container_width=True)
