import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Ferry Demand Forecasting",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1b3a5c;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #1f8a8c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 5px solid #1f8a8c;
    }
    .model-badge {
        display: inline-block;
        background-color: #e07a3e;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 1. Data Loading & Caching
# ============================================================================
@st.cache_data
def load_data():
    """Load and clean the ferry ticketing data."""
    try:
        df = pd.read_csv("Toronto_Island_Ferry_Tickets.csv")
    except FileNotFoundError:
        st.error("Data file 'Toronto_Island_Ferry_Tickets.csv' not found. Please place it in the same directory as this app.")
        st.stop()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp")
    df = df.rename(columns={"Sales Count": "sales", "Redemption Count": "redemptions"})
    df = df[["Timestamp", "sales", "redemptions"]].set_index("Timestamp")
    
    # Reindex to 15-min grid
    full_grid = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df_full = df.reindex(full_grid)
    df_full.index.name = "Timestamp"
    df_full["sales"] = df_full["sales"].fillna(0)
    df_full["redemptions"] = df_full["redemptions"].fillna(0)
    return df_full

@st.cache_data
def engineer_features(df):
    """Create features and targets."""
    feat = df.copy()
    # Temporal encodings
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
    
    # Lags
    for lag in [1, 2, 4, 8, 96, 672]:
        feat[f"sales_lag_{lag}"] = feat["sales"].shift(lag)
        feat[f"redemptions_lag_{lag}"] = feat["redemptions"].shift(lag)
    
    # Rolling stats
    for window in [4, 8, 16, 96]:
        feat[f"sales_roll_mean_{window}"] = feat["sales"].shift(1).rolling(window).mean()
        feat[f"sales_roll_std_{window}"] = feat["sales"].shift(1).rolling(window).std()
        feat[f"sales_roll_max_{window}"] = feat["sales"].shift(1).rolling(window).max()
    
    # Multi-horizon targets
    HORIZONS = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}
    for name, steps in HORIZONS.items():
        feat[f"target_sales_{name}"] = feat["sales"].shift(-steps)
    
    return feat, HORIZONS

# Feature columns used for ML models
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

# ============================================================================
# 2. Model Training & Caching
# ============================================================================
@st.cache_resource
def train_models():
    """Train all ML models and return them along with test data and predictions."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import xgboost as xgb
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    
    df_full = load_data()
    feat, HORIZONS = engineer_features(df_full)
    
    # Define train/test split (last 14 days for test)
    end_ts = feat.index.max()
    test_start = end_ts - pd.Timedelta(days=14)
    train_start = test_start - pd.Timedelta(days=365)
    
    # Prepare dictionaries
    models = {}
    predictions = {}
    test_data = {}
    performance = {}
    
    # ML models per horizon
    for hname, steps in HORIZONS.items():
        target_col = f"target_sales_{hname}"
        cols_needed = FEATURE_COLS + [target_col]
        sub = feat.loc[train_start:end_ts, cols_needed].dropna()
        train, test = sub.loc[:test_start], sub.loc[test_start:]
        X_train, y_train = train[FEATURE_COLS], train[target_col]
        X_test, y_test = test[FEATURE_COLS], test[target_col]
        
        # Store test data for later
        test_data[hname] = {"X_test": X_test, "y_test": y_test, "index": test.index}
        
        # Train models
        lr = LinearRegression().fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42).fit(X_train, y_train)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.5, random_state=42).fit(X_train, y_train)
        xgbm = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=42, n_jobs=-1).fit(X_train, y_train)
        
        models[hname] = {
            "Linear Regression": lr,
            "Random Forest": rf,
            "Gradient Boosting": gb,
            "XGBoost": xgbm
        }
        
        # Predictions for evaluation
        preds = {}
        for name, model in models[hname].items():
            y_pred = model.predict(X_test)
            preds[name] = np.clip(y_pred, 0, None)
        predictions[hname] = preds
        
        # Performance metrics
        perf = {}
        for name, y_pred in preds.items():
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            perf[name] = {"MAE": mae, "RMSE": rmse}
        performance[hname] = perf
    
    # SARIMA (simplified: pre-fit on entire train set, not rolling)
    # For demo, we'll just use a pre-defined model; but we'll implement a function to generate forecasts on the fly.
    # We'll store a pre-fit model for quick predictions.
    sarima_models = {}
    sales_series = df_full["sales"].asfreq("15min").fillna(0)
    train_series = sales_series.loc[train_start:test_start]
    try:
        sarima = SARIMAX(train_series, order=(2,1,2), enforce_stationarity=False, enforce_invertibility=False)
        sarima_fit = sarima.fit(disp=False, maxiter=50)
        sarima_models["fit"] = sarima_fit
        sarima_models["last_train_time"] = test_start
    except Exception as e:
        st.warning(f"SARIMA training failed: {e}")
        sarima_models = None
    
    # Prophet (pre-fit on last year)
    prophet_model = None
    try:
        p_train = sales_series.loc[train_start:test_start].reset_index()
        p_train.columns = ["ds", "y"]
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                    changepoint_prior_scale=0.05, interval_width=0.9)
        m.fit(p_train)
        prophet_model = m
    except Exception as e:
        st.warning(f"Prophet training failed: {e}")
        prophet_model = None
    
    return models, predictions, test_data, performance, sarima_models, prophet_model, HORIZONS, train_start, test_start

# ============================================================================
# 3. Prediction Functions
# ============================================================================
def predict_ml(model, features_df):
    """Predict using a trained ML model."""
    return model.predict(features_df[FEATURE_COLS])

# ============================================================================
# 4. Streamlit UI
# ============================================================================
def main():
    st.markdown('<div class="main-header">⛴️ Ferry Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Toronto Island Ferry Ticketing Data · 2015–2025</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Load data and models
    with st.spinner("Loading data and training models... (this may take a few minutes on first run)"):
        models, predictions, test_data, performance, sarima_models, prophet_model, HORIZONS, train_start, test_start = train_models()
    
    # Model selection
    model_options = ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]
    # Add SARIMA and Prophet if available
    if sarima_models is not None:
        model_options.append("SARIMA")
    if prophet_model is not None:
        model_options.append("Prophet")
    
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    # Horizon selection
    horizon_map = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}
    horizon_names = list(horizon_map.keys())
    selected_horizon = st.sidebar.selectbox("Forecast Horizon", horizon_names)
    steps = horizon_map[selected_horizon]
    
    # Date/time selector
    date_col1, date_col2 = st.sidebar.columns(2)
    with date_col1:
        date_input = st.date_input("Date", value=test_start.date(), min_value=test_start.date(), max_value=test_start.date() + timedelta(days=13))
    with date_col2:
        time_input = st.time_input("Time (15-min interval)", value=datetime.strptime("12:00", "%H:%M").time())
    
    # Combine to timestamp
    selected_timestamp = pd.Timestamp(datetime.combine(date_input, time_input))
    # Round to nearest 15-min
    selected_timestamp = selected_timestamp.floor("15min")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Forecast Origin:**")
    st.sidebar.write(selected_timestamp.strftime("%Y-%m-%d %H:%M"))
    
    # Fetch actual data for the selected timestamp (if in test set)
    df_full = load_data()
    actual_value = None
    if selected_timestamp in df_full.index:
        actual_value = df_full.loc[selected_timestamp, "sales"]
    
    # Get forecast
    forecast_value = None
    if selected_model in ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]:
        # Use ML model
        feat, _ = engineer_features(df_full)
        row = feat.loc[selected_timestamp:selected_timestamp, FEATURE_COLS].dropna()
        if len(row) == 0:
            st.sidebar.warning("Insufficient history for this timestamp (need lags and rolling stats). Try a later time.")
        else:
            model = models[selected_horizon][selected_model]
            pred = predict_ml(model, row)
            forecast_value = max(0, pred[0])
    elif selected_model == "SARIMA":
        @st.cache_data(show_spinner=False)
        def get_sarima_forecast(origin, steps):
            try:
                sales_series = df_full["sales"].asfreq("15min").fillna(0)
                hist = sales_series.loc[:origin]
                if len(hist) < 100:
                    return None
                model = SARIMAX(hist, order=(2,1,2), enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False, maxiter=50)
                fc = fit.get_forecast(steps=steps).predicted_mean
                return fc.iloc[-1] if len(fc) > 0 else None
            except:
                return None
        forecast_value = get_sarima_forecast(selected_timestamp, steps)
    elif selected_model == "Prophet":
        @st.cache_data(show_spinner=False)
        def get_prophet_forecast(origin, steps):
            try:
                sales_series = df_full["sales"].asfreq("15min").fillna(0)
                hist = sales_series.loc[:origin]
                if len(hist) < 100:
                    return None
                df_prophet = hist.reset_index()
                df_prophet.columns = ["ds", "y"]
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                            changepoint_prior_scale=0.05)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=steps, freq="15min")
                forecast = m.predict(future)
                return forecast["yhat"].iloc[-1] if len(forecast) > 0 else None
            except:
                return None
        forecast_value = get_prophet_forecast(selected_timestamp, steps)
    
    # Display results
    st.header("📊 Forecast Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Selected Time", selected_timestamp.strftime("%Y-%m-%d %H:%M"))
    with col2:
        st.metric("📈 Forecasted Sales", f"{forecast_value:.0f}" if forecast_value is not None else "N/A")
    with col3:
        if actual_value is not None:
            st.metric("✅ Actual Sales", f"{actual_value:.0f}")
            if forecast_value is not None:
                error = forecast_value - actual_value
                st.metric("Error", f"{error:+.0f}", delta=f"{error:+.0f}", delta_color="inverse")
        else:
            st.metric("✅ Actual Sales", "Not available (outside test period)")
    
    # Visualization: forecast vs actual over a recent window
    st.subheader("📈 Forecast vs Actual (Last 7 days)")
    end_vis = test_start + pd.Timedelta(days=13)
    start_vis = end_vis - pd.Timedelta(days=7)
    vis_series = df_full.loc[start_vis:end_vis]
    
    if selected_model in ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]:
        pred_series = predictions[selected_horizon][selected_model]
        test_idx = test_data[selected_horizon]["index"]
        pred_df = pd.DataFrame({"pred": pred_series}, index=test_idx)
        pred_aligned = pred_df.reindex(vis_series.index)
        actual_vis = vis_series["sales"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_vis.index, y=actual_vis, mode='lines', name='Actual', line=dict(color='#1b3a5c', width=2)))
        fig.add_trace(go.Scatter(x=pred_aligned.index, y=pred_aligned["pred"], mode='lines', name=f'{selected_model} Forecast', line=dict(color='#e07a3e', width=2, dash='dash')))
        fig.update_layout(title=f'{selected_model} - {selected_horizon} Horizon',
                          xaxis_title='Time', yaxis_title='Sales',
                          hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    elif selected_model in ["SARIMA", "Prophet"]:
        st.info(f"{selected_model} dynamic forecasts not precomputed for all timestamps. Please select a specific timestamp for forecast.")
    else:
        st.info("Select a model to see forecast vs actual.")
    
    # Performance metrics
    st.subheader("📊 Model Performance on Test Set")
    if selected_model in ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]:
        perf = performance[selected_horizon][selected_model]
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{perf['MAE']:.2f}")
        col2.metric("RMSE", f"{perf['RMSE']:.2f}")
    else:
        st.info(f"{selected_model} performance metrics not precomputed.")
    
    # Confidence intervals for Prophet
    if selected_model == "Prophet" and prophet_model is not None:
        st.subheader("📊 Forecast with Confidence Intervals")
        @st.cache_data(show_spinner=False)
        def get_prophet_forecast_with_ci(origin, steps):
            try:
                sales_series = df_full["sales"].asfreq("15min").fillna(0)
                hist = sales_series.loc[:origin]
                if len(hist) < 100:
                    return None
                df_prophet = hist.reset_index()
                df_prophet.columns = ["ds", "y"]
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                            changepoint_prior_scale=0.05, interval_width=0.9)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=steps, freq="15min")
                forecast = m.predict(future)
                return forecast
            except:
                return None
        forecast_df = get_prophet_forecast_with_ci(selected_timestamp, 8)
        if forecast_df is not None:
            fig = go.Figure()
            actual_hist = df_full.loc[selected_timestamp:selected_timestamp + pd.Timedelta(minutes=15*8)]
            fig.add_trace(go.Scatter(x=actual_hist.index, y=actual_hist["sales"], mode='lines', name='Actual', line=dict(color='#1b3a5c')))
            fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode='lines', name='Forecast', line=dict(color='#e07a3e')))
            fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"], mode='lines', name='Upper CI', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_lower"], mode='lines', name='Lower CI', line=dict(width=0), fill='tonexty', fillcolor='rgba(224,122,62,0.2)', showlegend=False))
            fig.update_layout(title='Prophet Forecast with 90% CI',
                              xaxis_title='Time', yaxis_title='Sales',
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate Prophet forecast for this timestamp.")
    
    # Model comparison table
    st.subheader("📊 Model Comparison (MAE by Horizon)")
    comp_data = []
    for hname in horizon_names:
        for mname in model_options:
            if mname in ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]:
                if hname in performance and mname in performance[hname]:
                    mae = performance[hname][mname]["MAE"]
                    comp_data.append({"Horizon": hname, "Model": mname, "MAE": mae})
    if comp_data:
        comp_df = pd.DataFrame(comp_data)
        pivot = comp_df.pivot(index="Model", columns="Horizon", values="MAE")
        st.dataframe(pivot.style.format("{:.2f}"))
    
    st.sidebar.markdown("---")
    st.sidebar.info("Data from Toronto Island Ferry Ticketing (2015-2025). Models trained on 1 year of historical data, tested on last 14 days.")
    st.sidebar.markdown("**Note:** SARIMA and Prophet forecasts are computed on-the-fly and may take a few seconds.")
    
if __name__ == "__main__":
    main()