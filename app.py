import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Plotly import with fallback
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False
    import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# ========================== SETUP ==========================
st.set_page_config(page_title="Ferry Demand Forecasting", page_icon="⛴️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1b3a5c; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #1f8a8c; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ========================== DATA LOADING ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("Toronto_Island_Ferry_Tickets.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp")
    df = df.rename(columns={"Sales Count": "sales", "Redemption Count": "redemptions"})
    df = df[["Timestamp", "sales", "redemptions"]].set_index("Timestamp")
    full_grid = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df_full = df.reindex(full_grid).fillna(0)
    df_full.index.name = "Timestamp"
    return df_full

@st.cache_data
def engineer_features(df):
    feat = df.copy()
    feat["hour"] = feat.index.hour
    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
    feat["hour_sin"] = np.sin(2*np.pi*feat["hour"]/24)
    feat["hour_cos"] = np.cos(2*np.pi*feat["hour"]/24)
    feat["dow_sin"] = np.sin(2*np.pi*feat["dow"]/7)
    feat["dow_cos"] = np.cos(2*np.pi*feat["dow"]/7)
    feat["month_sin"] = np.sin(2*np.pi*feat["month"]/12)
    feat["month_cos"] = np.cos(2*np.pi*feat["month"]/12)
    for lag in [1, 2, 4, 8, 96, 672]:
        feat[f"sales_lag_{lag}"] = feat["sales"].shift(lag)
        feat[f"redemptions_lag_{lag}"] = feat["redemptions"].shift(lag)
    for window in [4, 8, 16, 96]:
        feat[f"sales_roll_mean_{window}"] = feat["sales"].shift(1).rolling(window).mean()
        feat[f"sales_roll_std_{window}"] = feat["sales"].shift(1).rolling(window).std()
        feat[f"sales_roll_max_{window}"] = feat["sales"].shift(1).rolling(window).max()
    HORIZONS = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}
    for name, steps in HORIZONS.items():
        feat[f"target_sales_{name}"] = feat["sales"].shift(-steps)
    return feat, HORIZONS

FEATURE_COLS = [
    "hour", "dow", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8", "sales_lag_96", "sales_lag_672",
    "redemptions_lag_1", "redemptions_lag_2", "redemptions_lag_4", "redemptions_lag_8",
    "sales_roll_mean_4", "sales_roll_std_4", "sales_roll_max_4",
    "sales_roll_mean_8", "sales_roll_std_8", "sales_roll_max_8",
    "sales_roll_mean_16", "sales_roll_std_16", "sales_roll_max_16",
    "sales_roll_mean_96", "sales_roll_std_96", "sales_roll_max_96",
]

# ========================== MODEL TRAINING ==========================
@st.cache_resource
def train_models():
    df_full = load_data()
    feat, HORIZONS = engineer_features(df_full)
    end_ts = feat.index.max()
    test_start = end_ts - pd.Timedelta(days=14)
    train_start = test_start - pd.Timedelta(days=365)

    models = {}
    predictions = {}
    test_data = {}
    performance = {}

    for hname, steps in HORIZONS.items():
        target_col = f"target_sales_{hname}"
        cols_needed = FEATURE_COLS + [target_col]
        sub = feat.loc[train_start:end_ts, cols_needed].dropna()
        train, test = sub.loc[:test_start], sub.loc[test_start:]
        X_train, y_train = train[FEATURE_COLS], train[target_col]
        X_test, y_test = test[FEATURE_COLS], test[target_col]

        test_data[hname] = {"X_test": X_test, "y_test": y_test, "index": test.index}

        lr = LinearRegression().fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42).fit(X_train, y_train)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.5, random_state=42).fit(X_train, y_train)
        xgbm = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=42, n_jobs=-1).fit(X_train, y_train)

        models[hname] = {"Linear Regression": lr, "Random Forest": rf, "Gradient Boosting": gb, "XGBoost": xgbm}

        preds = {}
        for name, model in models[hname].items():
            y_pred = model.predict(X_test)
            preds[name] = np.clip(y_pred, 0, None)
        predictions[hname] = preds

        perf = {}
        for name, y_pred in preds.items():
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            perf[name] = {"MAE": mae, "RMSE": rmse}
        performance[hname] = perf

    return models, predictions, test_data, performance, HORIZONS, train_start, test_start

# ========================== UI ==========================
def main():
    st.markdown('<div class="main-header">⛴️ Ferry Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Toronto Island Ferry Ticketing Data · 2015–2025</div>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Controls")

    with st.spinner("Loading data and training models (first run may take 2-3 minutes)..."):
        models, predictions, test_data, performance, HORIZONS, train_start, test_start = train_models()

    model_options = ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)

    horizon_map = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}
    horizon_names = list(horizon_map.keys())
    selected_horizon = st.sidebar.selectbox("Forecast Horizon", horizon_names)
    steps = horizon_map[selected_horizon]

    date_input = st.sidebar.date_input("Date", value=test_start.date(), min_value=test_start.date(), max_value=test_start.date() + timedelta(days=13))
    time_input = st.sidebar.time_input("Time (15-min interval)", value=datetime.strptime("12:00", "%H:%M").time())
    selected_timestamp = pd.Timestamp(datetime.combine(date_input, time_input)).floor("15min")

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Forecast Origin:** {selected_timestamp.strftime('%Y-%m-%d %H:%M')}")

    df_full = load_data()
    actual_value = df_full.loc[selected_timestamp, "sales"] if selected_timestamp in df_full.index else None

    # Predict
    feat, _ = engineer_features(df_full)
    row = feat.loc[selected_timestamp:selected_timestamp, FEATURE_COLS].dropna()
    forecast_value = None
    if len(row) > 0:
        model = models[selected_horizon][selected_model]
        forecast_value = max(0, model.predict(row)[0])
    else:
        st.sidebar.warning("Insufficient history for this timestamp. Choose a later time.")

    # Metrics
    st.header("📊 Forecast Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Selected Time", selected_timestamp.strftime("%Y-%m-%d %H:%M"))
    c2.metric("📈 Forecasted Sales", f"{forecast_value:.0f}" if forecast_value is not None else "N/A")
    if actual_value is not None:
        c3.metric("✅ Actual Sales", f"{actual_value:.0f}")
        if forecast_value is not None:
            error = forecast_value - actual_value
            st.metric("Error", f"{error:+.0f}", delta=f"{error:+.0f}", delta_color="inverse")
    else:
        c3.metric("✅ Actual Sales", "Not available (outside test period)")

    # Forecast vs Actual plot (last 7 days)
    st.subheader("📈 Forecast vs Actual (Last 7 Days)")
    end_vis = test_start + pd.Timedelta(days=13)
    start_vis = end_vis - pd.Timedelta(days=7)
    vis_series = df_full.loc[start_vis:end_vis]

    pred_series = predictions[selected_horizon][selected_model]
    test_idx = test_data[selected_horizon]["index"]
    pred_df = pd.DataFrame({"pred": pred_series}, index=test_idx)
    pred_aligned = pred_df.reindex(vis_series.index)
    actual_vis = vis_series["sales"]

    if USE_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_vis.index, y=actual_vis, mode='lines', name='Actual', line=dict(color='#1b3a5c', width=2)))
        fig.add_trace(go.Scatter(x=pred_aligned.index, y=pred_aligned["pred"], mode='lines', name=f'{selected_model} Forecast', line=dict(color='#e07a3e', width=2, dash='dash')))
        fig.update_layout(title=f'{selected_model} - {selected_horizon} Horizon', xaxis_title='Time', yaxis_title='Sales', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(actual_vis.index, actual_vis, label='Actual', color='#1b3a5c', linewidth=2)
        ax.plot(pred_aligned.index, pred_aligned["pred"], label=f'{selected_model} Forecast', color='#e07a3e', linewidth=2, linestyle='--')
        ax.set_title(f'{selected_model} - {selected_horizon} Horizon')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig)

    # Performance
    st.subheader("📊 Model Performance on Test Set")
    perf = performance[selected_horizon][selected_model]
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{perf['MAE']:.2f}")
    col2.metric("RMSE", f"{perf['RMSE']:.2f}")

    # Model comparison
    st.subheader("📊 Model Comparison (MAE by Horizon)")
    comp_data = []
    for hname in horizon_names:
        for mname in model_options:
            if hname in performance and mname in performance[hname]:
                comp_data.append({"Horizon": hname, "Model": mname, "MAE": performance[hname][mname]["MAE"]})
    if comp_data:
        comp_df = pd.DataFrame(comp_data).pivot(index="Model", columns="Horizon", values="MAE")
        st.dataframe(comp_df.style.format("{:.2f}"))

    st.sidebar.markdown("---")
    st.sidebar.info("Data: Toronto Island Ferry (2015-2025). Models trained on 1 year, tested on last 14 days.")

if __name__ == "__main__":
    main()