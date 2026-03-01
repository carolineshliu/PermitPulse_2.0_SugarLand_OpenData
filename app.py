import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PermitPulse 2.0 — Sugar Land", layout="wide")

# -----------------------------
# Load and clean permits
# -----------------------------
@st.cache_data
def load_permits(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1252")
    # strip extra spaces from column names
    df.columns = df.columns.str.strip()
    
    # Parse dates
    date_cols = ["Applied Date", "Issued Date", "Expire Date", "Completed Date"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Convert numeric columns
    for c in ["Permit Valuation", "Permit Square Feet"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Clean ZIPs
    def clean_zip(z):
        if pd.isna(z):
            return np.nan
        z = str(z).strip()
        m = re.match(r"(\d{5})", z)
        return m.group(1) if m else np.nan
    if "Zip Code" in df.columns:
        df["zip5"] = df["Zip Code"].apply(clean_zip)
    else:
        df["zip5"] = np.nan

    # Days to issue/complete
    if "Issued Date" in df.columns and "Applied Date" in df.columns:
        df["days_to_issue"] = (df["Issued Date"] - df["Applied Date"]).dt.days
    else:
        df["days_to_issue"] = np.nan
    if "Completed Date" in df.columns and "Issued Date" in df.columns:
        df["days_to_complete"] = (df["Completed Date"] - df["Issued Date"]).dt.days
    else:
        df["days_to_complete"] = np.nan

    # Clean string columns
    str_cols = ["Permit Status", "Type", "Workclass", "City", "State", "Address", "Contact Company Name"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    
    return df

# -----------------------------
# Load traffic info/events
# -----------------------------
@st.cache_data
def load_traffic_information(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

@st.cache_data
def load_traffic_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["start_time", "end_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# -----------------------------
# Normalization helper
# -----------------------------
def normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

# -----------------------------
# Load data
# -----------------------------
st.title("PermitPulse 2.0 — Development × Traffic Intelligence")
st.caption("Sugar Land open data integration: Building Permits + Traffic Information + Traffic Events.")

with st.sidebar:
    st.header("Data files")
    permits_file = st.text_input("Permits CSV path", r"C:\Users\caroline\Downloads\PermitPulse_2_Full_Submission_Package\PermitPulse_2.0_SugarLand_OpenData\data\permits.csv")
    traffic_info_file = st.text_input("Traffic Information CSV path", r"C:\Users\caroline\Downloads\PermitPulse_2_Full_Submission_Package\PermitPulse_2.0_SugarLand_OpenData\data\traffic_information.csv")
    traffic_events_file = st.text_input("Traffic Events CSV path", r"C:\Users\caroline\Downloads\PermitPulse_2_Full_Submission_Package\PermitPulse_2.0_SugarLand_OpenData\data\traffic_events.csv")
    st.caption("Replace the traffic CSVs with official portal exports; keep/rename columns to match template headers.")

permits = load_permits(permits_file)
tinfo = load_traffic_information(traffic_info_file)
tevents = load_traffic_events(traffic_events_file)

# -----------------------------
# Sidebar Filters
# -----------------------------
with st.sidebar:
    st.header("Filters")
    # ZIP filter
    zips = sorted([z for z in permits["zip5"].dropna().unique().tolist()])
    zip_sel = st.multiselect("ZIP", options=zips, default=zips if zips else [])
    
    # Date filter
    if "Applied Date" in permits.columns and permits["Applied Date"].notna().any():
        min_date = permits["Applied Date"].min()
        max_date = permits["Applied Date"].max()
        date_range = st.date_input(
            "Applied date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
    else:
        date_range = (None, None)

# -----------------------------
# Apply filters
# -----------------------------
p = permits.copy()
if zip_sel:
    p = p[p["zip5"].isin(zip_sel)]
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if "Applied Date" in p.columns:
        p = p[(p["Applied Date"] >= start) & (p["Applied Date"] <= end)]

# -----------------------------
# Metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Permits (filtered)", f"{len(p):,}")
c2.metric("Median days to issue", f"{p['days_to_issue'].median(skipna=True):.0f}")
c3.metric("P90 days to issue", f"{p['days_to_issue'].quantile(0.9):.0f}")
c4.metric("Traffic events (rows)", f"{len(tevents):,}")

st.divider()

# -----------------------------
# Permit & Traffic charts
# -----------------------------
left, right = st.columns([2, 1])
with left:
    st.subheader("Permit volume over time")
    if "Applied Date" in p.columns:
        g = p.dropna(subset=["Applied Date"]).copy()
        g["month"] = g["Applied Date"].dt.to_period("M").dt.to_timestamp()
        monthly = g.groupby("month").size().reset_index(name="permits")
        st.line_chart(monthly.set_index("month")["permits"], height=260)
with right:
    st.subheader("Traffic events over time")
    if "start_time" in tevents.columns:
        h = tevents.dropna(subset=["start_time"]).copy()
        h["day"] = h["start_time"].dt.to_period("D").dt.to_timestamp()
        daily = h.groupby("day").size().reset_index(name="events")
        st.line_chart(daily.set_index("day")["events"], height=260)
    else:
        st.info("Traffic events CSV missing start_time.")

st.divider()

# -----------------------------
# Urban Stress Score (ZIP-level)
# -----------------------------
st.subheader("Urban Stress Score (ZIP-level)")
dev = p.groupby("zip5").agg(
    permit_count=("Permit Number", "count"),
    valuation_sum=("Permit Valuation", "sum"),
).reset_index()
dev["dpi"] = normalize(dev["permit_count"]) * 0.6 + normalize(dev["valuation_sum"]) * 0.4

if "zip5" in tevents.columns:
    ts = tevents.groupby("zip5").size().reset_index(name="event_count")
else:
    ts = pd.DataFrame({"zip5": dev["zip5"], "event_count": np.repeat(len(tevents), len(dev))})
ts["tsi"] = normalize(ts["event_count"])

score = dev.merge(ts[["zip5","event_count","tsi"]], on="zip5", how="left")
score["urban_stress"] = score["dpi"] * score["tsi"]

a, b = st.columns(2)
with a:
    st.write("**Top ZIPs by Urban Stress**")
    st.dataframe(score.sort_values("urban_stress", ascending=False).head(10), use_container_width=True, height=360)
with b:
    st.write("**Components**")
    st.dataframe(score.sort_values("dpi", ascending=False).head(10)[["zip5","permit_count","valuation_sum","dpi","event_count","tsi","urban_stress"]],
                 use_container_width=True, height=360)

st.divider()

# -----------------------------
# Delay Watch
# -----------------------------
st.subheader("Delay Watch — unusually slow permits to issue")
slow = p.dropna(subset=["days_to_issue"]).copy()
slow = slow[slow["days_to_issue"] >= 0]
if len(slow):
    grp = slow.groupby(["Type", "Workclass"])["days_to_issue"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: np.median(np.abs(s - np.median(s))) if len(s) else np.nan)
    slow["robust_z"] = (slow["days_to_issue"] - med) / (1.4826 * mad.replace(0, np.nan))
    flagged = slow.sort_values(["robust_z", "days_to_issue"], ascending=False).head(50)
    cols = ["Permit Number","Applied Date","Issued Date","days_to_issue","robust_z","Type","Workclass","Permit Status","Address","zip5"]
    st.dataframe(flagged[cols], use_container_width=True, height=420)
else:
    st.info("No issue-time data available under current filters.")

st.divider()

# -----------------------------
# Export CSV
# -----------------------------
st.subheader("Export")
csv = p.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered permits CSV", data=csv, file_name="permitpulse_filtered.csv", mime="text/csv")

st.write("Traffic Events CSV preview")
st.write(tevents.head(10))
st.write("Columns in traffic events CSV:", tevents.columns.tolist())
st.write("Number of valid start_time rows:", tevents["start_time"].notna().sum())

st.subheader("Permit Type Distribution")
type_counts = p["Type"].value_counts().head(10)
st.bar_chart(type_counts)

statuses = sorted(p["Permit Status"].dropna().unique())
status_sel = st.sidebar.multiselect("Permit Status", options=statuses, default=statuses)
p = p[p["Permit Status"].isin(status_sel)]
