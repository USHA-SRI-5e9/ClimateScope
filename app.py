import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="ClimateScope", layout="wide")

USERS_FILE = "users.csv"
DATA_FILE = "data/climate_data.csv"

# ---------------- AUTH UTILS ---------------- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=["username", "password"])
    return pd.read_csv(USERS_FILE)

def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    users.loc[len(users)] = [username, hash_password(password)]
    users.to_csv(USERS_FILE, index=False)
    return True

def authenticate(username, password):
    users = load_users()
    return not users[
        (users["username"] == username) &
        (users["password"] == hash_password(password))
    ].empty

# ---------------- SESSION ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ---------------- #
def login_page():
    st.title("🌍 ClimateScope – Visualizing Global Weather Trends")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            if save_user(new_user, new_pass):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists")

# ---------------- DATA CLEANING ---------------- #
def load_and_clean_data():
    df = pd.read_csv(DATA_FILE)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "datetime" not in df.columns:
        st.error("❌ 'datetime' column not found.")
        st.stop()

    df = df.dropna(how="all")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    numeric_cols = [
        "temperature_celsius",
        "precip_mm",
        "humidity",
        "wind_kph"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    return df

# ---------------- DASHBOARD ---------------- #
def dashboard():

    st.sidebar.title("🌡 ClimateScope")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Statistical Analysis",
            "Extreme Events",
            "Line & Distribution Charts",
            "Correlation Heatmap",
            "Choropleth Map",
            "Country Similarity",
            "Climate Ranking"
        ]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    df = load_and_clean_data()

    # ======================================================
    # OVERVIEW
    # ======================================================
    if menu == "Overview":

        st.title("📊 Climate Overview")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # ======================================================
    # STATISTICAL ANALYSIS
    # ======================================================
    elif menu == "Statistical Analysis":

        st.title("📈 Statistical Analysis")

        st.subheader("Skewness")
        skew_value = df["temperature_celsius"].skew()
        st.write(f"Skewness: {skew_value:.3f}")

        st.subheader("Z-Score Anomalies")
        df["z_score"] = (
            (df["temperature_celsius"] - df["temperature_celsius"].mean())
            / df["temperature_celsius"].std()
        )
        z_anomalies = df[abs(df["z_score"]) > 3]
        st.dataframe(z_anomalies.head(20))

        st.subheader("IQR Outliers")
        Q1 = df["temperature_celsius"].quantile(0.25)
        Q3 = df["temperature_celsius"].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        iqr_outliers = df[
            (df["temperature_celsius"] < lower) |
            (df["temperature_celsius"] > upper)
        ]

        st.dataframe(iqr_outliers.head(20))

    # ======================================================
    # EXTREME EVENTS
    # ======================================================
    elif menu == "Extreme Events":

        st.title("🔥 Extreme Temperature Events")

        threshold = df["temperature_celsius"].mean() + df["temperature_celsius"].std()
        extreme_df = df[df["temperature_celsius"] > threshold]

        st.write(f"Threshold: {threshold:.2f} °C")
        st.dataframe(extreme_df.head(20))

        extreme_count = (
            extreme_df.groupby("country")
            .size()
            .reset_index(name="extreme_days")
            .sort_values("extreme_days", ascending=False)
        )

        fig_bar = px.bar(
            extreme_count.head(10),
            x="country",
            y="extreme_days"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            extreme_count.head(5),
            names="country",
            values="extreme_days"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ======================================================
    # LINE & DISTRIBUTION CHARTS
    # ======================================================
    elif menu == "Line & Distribution Charts":

        st.title("📊 Climate Visualizations")

        yearly_temp = (
            df.groupby(["year", "country"])["temperature_celsius"]
            .mean()
            .reset_index()
        )

        fig_line = px.line(
            yearly_temp,
            x="year",
            y="temperature_celsius",
            color="country"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        fig_hist = px.histogram(
            df,
            x="temperature_celsius",
            nbins=30
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_violin = px.violin(
            df,
            x="country",
            y="temperature_celsius",
            box=True
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ======================================================
    # CORRELATION HEATMAP
    # ======================================================
    elif menu == "Correlation Heatmap":

        st.title("📊 Correlation Heatmap")

        corr = df[
            ["temperature_celsius", "precip_mm", "humidity", "wind_kph"]
        ].corr()

        fig_heatmap = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ======================================================
    # CHOROPLETH MAP
    # ======================================================
    elif menu == "Choropleth Map":

        st.title("🌍 Global Temperature Choropleth Map")

        country_avg = (
            df.groupby("country")["temperature_celsius"]
            .mean()
            .reset_index()
        )

        fig_map = px.choropleth(
            country_avg,
            locations="country",
            locationmode="country names",
            color="temperature_celsius",
            color_continuous_scale="RdYlBu_r"
        )

        st.plotly_chart(fig_map, use_container_width=True)

    # ======================================================
    # COUNTRY SIMILARITY
    # ======================================================
    elif menu == "Country Similarity":

        st.title("🌍 Country Climate Similarity")

        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity

        country_features = df.groupby("country").agg({
            "temperature_celsius": "mean",
            "precip_mm": "mean",
            "humidity": "mean",
            "wind_kph": "mean"
        }).reset_index()

        countries = country_features["country"].unique()

        col1, col2 = st.columns(2)

        with col1:
            country1 = st.selectbox("Select Country 1", countries)

        with col2:
            country2 = st.selectbox("Select Country 2", countries, index=1)

        if country1 != country2:

            scaler = StandardScaler()
            scaled = scaler.fit_transform(country_features.iloc[:, 1:])

            scaled_df = pd.DataFrame(
                scaled,
                index=country_features["country"],
                columns=country_features.columns[1:]
            )

            vec1 = scaled_df.loc[country1].values.reshape(1, -1)
            vec2 = scaled_df.loc[country2].values.reshape(1, -1)

            similarity = cosine_similarity(vec1, vec2)[0][0]
            similarity_percent = round(similarity * 100, 2)

            st.success(f"Similarity: {similarity_percent}%")

        else:
            st.warning("Please select two different countries")

    # ======================================================
    # CLIMATE RANKING
    # ======================================================
    elif menu == "Climate Ranking":

        st.title("🏆 Climate Intensity Ranking")

        country_features = df.groupby("country").agg({
            "temperature_celsius": "mean",
            "precip_mm": "mean",
            "humidity": "mean",
            "wind_kph": "mean"
        }).reset_index()

        country_features["climate_score"] = (
            country_features["temperature_celsius"] * 0.4 +
            country_features["precip_mm"] * 0.3 +
            country_features["humidity"] * 0.2 +
            country_features["wind_kph"] * 0.1
        )

        ranking_df = country_features.sort_values("climate_score", ascending=False)

        st.dataframe(
            ranking_df[["country", "climate_score"]].head(10),
            use_container_width=True
        )
# ---------------- MAIN ---------------- #
if st.session_state.logged_in:
    dashboard()
else:
    login_page()