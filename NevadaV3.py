import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import pydeck as pdk
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------
# BASIC CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Nevada Irrigation & ML Yield Dashboard",
    layout="wide"
)

st.title("Nevada Irrigation Water Management Dashboard (Crop-Specific ML Yield)")

# -----------------------------------------------------
# NEVADA COUNTY CENTROIDS
# -----------------------------------------------------
NEVADA_COUNTIES = [
    {"county": "Clark",        "lat": 36.17, "lon": -115.14},
    {"county": "Washoe",       "lat": 39.53, "lon": -119.81},
    {"county": "Carson City",  "lat": 39.16, "lon": -119.77},
    {"county": "Douglas",      "lat": 38.91, "lon": -119.63},
    {"county": "Lyon",         "lat": 39.02, "lon": -119.17},
    {"county": "Churchill",    "lat": 39.47, "lon": -118.78},
    {"county": "Pershing",     "lat": 40.44, "lon": -118.40},
    {"county": "Humboldt",     "lat": 40.97, "lon": -117.73},
    {"county": "Elko",         "lat": 40.83, "lon": -115.76},
    {"county": "White Pine",   "lat": 39.25, "lon": -114.90},
    {"county": "Lander",       "lat": 39.88, "lon": -117.01},
    {"county": "Eureka",       "lat": 39.52, "lon": -116.21},
    {"county": "Nye",          "lat": 38.0,  "lon": -116.7},
    {"county": "Esmeralda",    "lat": 37.80, "lon": -117.60},
    {"county": "Mineral",      "lat": 38.50, "lon": -118.60},
    {"county": "Lincoln",      "lat": 37.60, "lon": -114.88},
    {"county": "Storey",       "lat": 39.39, "lon": -119.53},
]
county_df = pd.DataFrame(NEVADA_COUNTIES)

# -----------------------------------------------------
# CROP / SOIL / IRRIGATION PARAMETERS
# -----------------------------------------------------
CROP_PARAMS = {
    "Alfalfa (hay)": {
        "root_depth_m": 1.5,
        "readily_available_fraction": 0.5,
        "kc": 1.15
    },
    "Corn (grain)": {
        "root_depth_m": 1.2,
        "readily_available_fraction": 0.5,
        "kc": 1.15
    },
    "Small grains (wheat/barley)": {
        "root_depth_m": 1.0,
        "readily_available_fraction": 0.45,
        "kc": 1.05
    },
    "Pasture / mixed grass": {
        "root_depth_m": 0.8,
        "readily_available_fraction": 0.4,
        "kc": 0.95
    },
    "Vegetables (generic)": {
        "root_depth_m": 0.6,
        "readily_available_fraction": 0.35,
        "kc": 1.05
    }
}

# Default efficiencies by system (used as slider defaults)
IRRIGATION_EFF_DEFAULT = {
    "Flood / surface": 0.55,
    "Center pivot": 0.75,
    "Drip / micro": 0.90
}

SOIL_TAW_MM = {
    "Sand": 80,
    "Loamy sand": 100,
    "Sandy loam": 120,
    "Loam": 150,
    "Silt loam": 170,
    "Clay loam": 180,
    "Clay": 200
}

# -----------------------------------------------------
# CLIMATE FUNCTIONS
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_daily_climate_open_meteo(lat, lon, start_date, end_date):
    """Fetch daily precip + ET0 from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,et0_fao_evapotranspiration",
        "timezone": "America/Los_Angeles",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "precip_mm": data["daily"]["precipitation_sum"],
        "et0_mm": data["daily"]["et0_fao_evapotranspiration"]
    })
    return df


def build_simple_et0_pattern(start_date, end_date):
    """Simple demo ET0 pattern: sinusoidal ET0, light random rainfall."""
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    # 5 ± 1.5 mm/day ET0
    t = np.linspace(0, 2 * np.pi, n)
    et0 = 5.0 + 1.5 * np.sin(t)
    # Occasional small rainfall
    rng = np.random.default_rng(123)
    precip = rng.choice([0, 0, 0, 3, 5, 8], size=n, p=[0.6, 0.2, 0.05, 0.1, 0.03, 0.02])

    df = pd.DataFrame({
        "date": dates,
        "precip_mm": precip,
        "et0_mm": et0
    })
    return df


def read_climate_csv(file, start_date, end_date):
    """
    Read uploaded CSV with columns:
    date, precip_mm, et0_mm
    """
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    if "precip_mm" not in df.columns or "et0_mm" not in df.columns:
        raise ValueError("CSV must contain 'precip_mm' and 'et0_mm' columns.")

    # Filter to requested period if possible
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    if mask.any():
        df = df.loc[mask].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_climate_df(option, lat, lon, start_date, end_date, uploaded_file):
    """Wrapper that returns climate df according to selected option."""
    if option.startswith("Open-Meteo"):
        return fetch_daily_climate_open_meteo(lat, lon, start_date, end_date)
    elif option.startswith("Simple ET"):
        return build_simple_et0_pattern(start_date, end_date)
    else:  # upload CSV
        if uploaded_file is None:
            raise ValueError("Please upload a daily climate CSV file.")
        return read_climate_csv(uploaded_file, start_date, end_date)

# -----------------------------------------------------
# WATER BALANCE & ML
# -----------------------------------------------------
def run_water_balance(
    climate_df,
    crop,
    soil,
    irr_efficiency,
    rainfall_effectiveness,
    strategy
):
    """
    Simple soil-water balance with fixed Kc.
    Returns daily dataframe including irrigation events.
    """
    params = CROP_PARAMS[crop]
    root_depth_m = params["root_depth_m"]
    frac_raw = params["readily_available_fraction"]
    kc = params["kc"]

    taw_per_m = SOIL_TAW_MM[soil]  # mm/m
    TAW = root_depth_m * taw_per_m  # mm
    RAW = frac_raw * TAW

    # Strategy factor: allow more depletion before irrigating
    if strategy.startswith("Full irrigation"):
        trigger_factor = 1.0
    elif "mild deficit" in strategy:
        trigger_factor = 1.4
    else:  # severe deficit
        trigger_factor = 1.8

    trigger_depletion = min(TAW * 0.9, RAW * trigger_factor)

    df = climate_df.copy().reset_index(drop=True)
    df["day"] = np.arange(len(df))

    sw = TAW  # start at field capacity
    irrigations = []
    sw_list = []
    depletions = []

    for _, row in df.iterrows():
        et_c = row["et0_mm"] * kc
        precip = row["precip_mm"] * rainfall_effectiveness

        # soil water balance
        sw = sw + precip - et_c
        if sw > TAW:
            sw = TAW

        depletion = TAW - sw
        irrigation_mm = 0.0

        # Irrigate when depletion exceeds threshold
        if depletion > trigger_depletion:
            net_req = depletion
            gross = net_req / max(irr_efficiency, 1e-6)
            irrigation_mm = gross
            sw += net_req
            depletion = TAW - sw

        irrigations.append(irrigation_mm)
        sw_list.append(sw)
        depletions.append(depletion)

    df["et_c_mm"] = df["et0_mm"] * kc
    df["soil_water_mm"] = sw_list
    df["depletion_mm"] = depletions
    df["irrigation_mm"] = irrigations

    return df


@st.cache_resource(show_spinner=False)
def build_synthetic_ml_models():
    """
    Build simple crop-specific RandomForest models on synthetic data.
    """
    models = {}
    rng = np.random.default_rng(42)

    for crop in CROP_PARAMS.keys():
        n = 800
        total_irr = rng.uniform(0, 800, n)
        total_precip = rng.uniform(0, 450, n)
        seasonal_et0 = rng.uniform(700, 1400, n)
        deficit = rng.uniform(0, 1, n)

        if "Alfalfa" in crop:
            base = 10
            irr_coeff = 0.004
            precip_coeff = 0.003
        elif "Corn" in crop:
            base = 8
            irr_coeff = 0.005
            precip_coeff = 0.004
        elif "Small grains" in crop:
            base = 5
            irr_coeff = 0.0035
            precip_coeff = 0.0035
        elif "Pasture" in crop:
            base = 6
            irr_coeff = 0.003
            precip_coeff = 0.003
        else:  # vegetables
            base = 15
            irr_coeff = 0.006
            precip_coeff = 0.0045

        noise = rng.normal(0, 0.5, n)
        yield_t_ha = (
            base
            + irr_coeff * total_irr
            + precip_coeff * total_precip
            - 3.0 * deficit
            + noise
        )
        yield_t_ha = np.maximum(yield_t_ha, 0.1)

        X = np.column_stack([total_irr, total_precip, seasonal_et0, deficit])
        y = yield_t_ha

        rf = RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            max_depth=12,
            min_samples_leaf=3
        )
        rf.fit(X, y)
        models[crop] = rf

    return models


def predict_yield_for_scenario(models, crop, wb_df):
    """Aggregate water-balance outputs for ML prediction."""
    total_irr_mm = wb_df["irrigation_mm"].sum()
    total_precip_mm = wb_df["precip_mm"].sum()
    seasonal_et0_mm = wb_df["et0_mm"].sum()
    deficit_index = (wb_df["depletion_mm"] > wb_df["depletion_mm"].max() * 0.5).mean()

    model = models[crop]
    X = np.array([[total_irr_mm, total_precip_mm, seasonal_et0_mm, deficit_index]])
    pred_yield = model.predict(X)[0]

    return {
        "pred_yield_t_ha": float(pred_yield),
        "total_irr_mm": float(total_irr_mm),
        "total_precip_mm": float(total_precip_mm),
        "seasonal_et0_mm": float(seasonal_et0_mm),
        "deficit_index": float(deficit_index)
    }

# -----------------------------------------------------
# SIDEBAR LAYOUT (LEFT SIDE)
# -----------------------------------------------------
st.sidebar.markdown("### 1. Location & Crop Setup")

state = st.sidebar.selectbox("Select state", ["Nevada"], index=0)

# County selector (Nevada only)
county_choice = st.sidebar.selectbox(
    "County (Nevada)",
    options=county_df["county"].tolist(),
    index=2  # default Carson City
)
county_row = county_df[county_df["county"] == county_choice].iloc[0]

# Lat / Lon, prefilled from county but editable (GPS override)
lat = st.sidebar.number_input(
    "Latitude (°N)",
    value=float(county_row["lat"]),
    format="%.4f"
)
lon = st.sidebar.number_input(
    "Longitude (°E)",
    value=float(county_row["lon"]),
    format="%.4f"
)

season_year = st.sidebar.slider("Season year", 2020, 2035, 2024)

planting_date = st.sidebar.date_input(
    "Planting / emergence date",
    value=dt.date(season_year, 4, 15)
)

crop_choice = st.sidebar.selectbox(
    "Crop",
    options=list(CROP_PARAMS.keys()),
    index=1  # default Corn (grain)
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Soil & SSURGO (beta)")

soil_choice = st.sidebar.selectbox(
    "Dominant soil type (manual)",
    options=list(SOIL_TAW_MM.keys()),
    index=3  # Loam
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Climate Data / ETo options")

climate_option = st.sidebar.radio(
    "Climate / ET₀ source",
    [
        "Open-Meteo (automatic)",
        "Simple ET₀ pattern (demo)",
        "Upload daily climate CSV"
    ],
    index=0
)

uploaded_climate_file = None
if climate_option.startswith("Upload"):
    uploaded_climate_file = st.sidebar.file_uploader(
        "Upload daily climate CSV (date, precip_mm, et0_mm)",
        type=["csv"]
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 4. Irrigation system & strategy")

irr_sys_choice = st.sidebar.selectbox(
    "Irrigation system",
    options=list(IRRIGATION_EFF_DEFAULT.keys()),
    index=1  # Center pivot
)

default_eff = IRRIGATION_EFF_DEFAULT[irr_sys_choice]
irr_strategy = st.sidebar.selectbox(
    "Irrigation strategy",
    options=[
        "Full irrigation (no intentional stress)",
        "Allow mild deficit (demo)",
        "Severe deficit (demo)"
    ],
    index=0
)

irr_efficiency = st.sidebar.slider(
    "Irrigation application efficiency",
    min_value=0.40,
    max_value=0.95,
    value=float(default_eff),
    step=0.01
)

rainfall_effectiveness = st.sidebar.slider(
    "Rainfall effectiveness",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.05
)

season_length_days = st.sidebar.slider(
    "Season length (days)",
    min_value=60,
    max_value=250,
    value=140,
    step=10
)

run_button = st.sidebar.button("Run irrigation simulation")

# -----------------------------------------------------
# MAIN LAYOUT: BIG MAP + SCENARIO SUMMARY
# -----------------------------------------------------
map_col, info_col = st.columns([1.8, 1])

with map_col:
    st.subheader("Nevada county map (medium dots)")

    map_df = county_df.copy()
    map_df["selected"] = map_df["county"] == county_choice
    map_df["color_r"] = np.where(map_df["selected"], 255, 0)
    map_df["color_g"] = np.where(map_df["selected"], 140, 100)
    map_df["color_b"] = np.where(map_df["selected"], 0, 200)
    map_df["radius"] = np.where(map_df["selected"], 15000, 10000)  # medium dots

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="[color_r, color_g, color_b]",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=39.0,
        longitude=-117.0,
        zoom=5.5,
        pitch=0
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip={"text": "{county}"}
    )

    st.pydeck_chart(r, use_container_width=True, height=550)

    # Long description moved to bottom of map
    st.markdown(
        """
This prototype dashboard is designed for Nevada Extension and other stakeholders.

- Select a county or enter GPS coordinates.  
- Choose crop, soil, irrigation system, and planting date.  
- The app:  
  - Pulls daily climate (ET₀ & rainfall) via Open-Meteo or other options.  
  - Runs a simple soil-water balance to schedule irrigations.  
  - Uses crop-specific ML models (Random Forest) to estimate yield from water balance indicators.  
  - (Built-in synthetic models – you can later swap with real trial data.)  

Extension agents can quickly choose a county (map or dropdown) or use field GPS coordinates directly from a farm visit.
"""
    )

with info_col:
    st.subheader("Scenario summary (inputs)")
    st.markdown(
        f"""
- **Crop:** {crop_choice}  
- **Soil:** {soil_choice}  
- **Irrigation system:** {irr_sys_choice}  
- **Irrigation strategy:** {irr_strategy}  
- **Season year:** {season_year}  
- **Planting date:** {planting_date.isoformat()}  
- **Season length:** {season_length_days} days  
- **Location:** {lat:.3f}°N, {lon:.3f}°E  
- **County (for map):** {county_choice}
"""
    )

# -----------------------------------------------------
# RUN MODEL WHEN BUTTON CLICKED
# -----------------------------------------------------
if run_button:
    with st.spinner("Running irrigation simulation and ML yield model..."):
        start_date = planting_date
        end_date = planting_date + dt.timedelta(days=season_length_days)

        # Climate
        try:
            climate_df = get_climate_df(
                option=climate_option,
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
                uploaded_file=uploaded_climate_file
            )
        except Exception as e:
            st.error(f"Error loading climate data: {e}")
            st.stop()

        # Water balance
        wb_df = run_water_balance(
            climate_df=climate_df,
            crop=crop_choice,
            soil=soil_choice,
            irr_efficiency=irr_efficiency,
            rainfall_effectiveness=rainfall_effectiveness,
            strategy=irr_strategy
        )

        # ML yield
        models = build_synthetic_ml_models()
        ml_results = predict_yield_for_scenario(models, crop_choice, wb_df)

    st.markdown("---")
    st.header("Irrigation & yield results for selected scenario")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Predicted yield", f"{ml_results['pred_yield_t_ha']:.1f} t/ha")
    kpi2.metric("Total irrigation", f"{ml_results['total_irr_mm']:.0f} mm")
    kpi3.metric("Total rainfall", f"{ml_results['total_precip_mm']:.0f} mm")
    kpi4.metric("Seasonal ET₀", f"{ml_results['seasonal_et0_mm']:.0f} mm")

    st.markdown(
        f"**Deficit index (0 = no stress, 1 = severe seasonal stress):** "
        f"{ml_results['deficit_index']:.2f}"
    )

    st.subheader("Daily irrigation and water balance timeline")

    chart_df = wb_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])

    fig1 = px.bar(
        chart_df,
        x="date",
        y=["irrigation_mm", "precip_mm"],
        labels={"value": "mm/day", "date": "Date", "variable": "Water component"},
        title="Daily irrigation and rainfall"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(
        chart_df,
        x="date",
        y=["soil_water_mm", "depletion_mm"],
        labels={"value": "mm", "date": "Date", "variable": "Soil-water state"},
        title="Soil-water content and depletion"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Irrigation event table")
    irr_events = chart_df[chart_df["irrigation_mm"] > 0].copy()
    irr_events = irr_events[["date", "irrigation_mm", "precip_mm", "et0_mm", "et_c_mm"]]
    irr_events["date"] = irr_events["date"].dt.date

    if len(irr_events) == 0:
        st.warning("No irrigation events triggered under this scenario.")
    else:
        st.dataframe(
            irr_events.rename(columns={
                "date": "Date",
                "irrigation_mm": "Irrigation (mm)",
                "precip_mm": "Rain (mm)",
                "et0_mm": "ET₀ (mm)",
                "et_c_mm": "ETc (mm)"
            }),
            use_container_width=True
        )

else:
    st.info("Set up your scenario on the left and click **Run irrigation simulation**.")
