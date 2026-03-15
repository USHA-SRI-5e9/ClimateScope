import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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

    if not os.path.exists(DATA_FILE):
        st.error("Climate dataset not found.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "datetime" not in df.columns:
        st.error("❌ 'datetime' column not found.")
        st.stop()

    df = df.dropna(how="all")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    numeric_cols = ["temperature_celsius", "precip_mm", "humidity", "wind_kph"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    return df


# ---------------- DASHBOARD ---------------- #
def dashboard():

    st.sidebar.title("🌡 ClimateScope")
    st.write("Milestone 3 Version")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Statistical Analysis",
            "Extreme Events",
            "Line & Distribution Charts",
            "Correlation Heatmap",
            "Choropleth Map",
            "Seasonal Heatmap",
            "Wind Analysis",
            "Country Similarity",
            "Climate Ranking",
            "Anomaly Visualization"
        ]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    df = load_and_clean_data()

    # ---------------- GLOBAL FILTERS ---------------- #
    st.sidebar.subheader("Filters")

    countries = st.sidebar.multiselect(
        "Select Countries",
        df["country"].unique(),
        default=df["country"].unique()[:5]
    )

    if countries:
        df = df[df["country"].isin(countries)]

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    year_range = st.sidebar.slider(
        "Select Year Range",
        min_year,
        max_year,
        (min_year, max_year)
    )

    df = df[
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1])
    ]

    # ---------------- DASHBOARD METRICS ---------------- #
    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Temperature", round(df["temperature_celsius"].mean(), 2))
    col2.metric("Avg Humidity", round(df["humidity"].mean(), 2))
    col3.metric("Avg Wind Speed", round(df["wind_kph"].mean(), 2))

    # ---------------- OVERVIEW ---------------- #
    if menu == "Overview":

        st.title("📊 Climate Overview")

        st.subheader("🔍 Key Climate Insights")

        hottest_country = df.groupby("country")["temperature_celsius"].mean().idxmax()
        coldest_country = df.groupby("country")["temperature_celsius"].mean().idxmin()
        most_rain = df.groupby("country")["precip_mm"].mean().idxmax()

        st.success(f"Hottest Country: {hottest_country}")
        st.info(f"Coldest Country: {coldest_country}")
        st.warning(f"Highest Rainfall Country: {most_rain}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # ---------------- STATISTICAL ANALYSIS ---------------- #
    elif menu == "Statistical Analysis":

        st.title("📈 Statistical Analysis")

        st.write("Skewness:", df["temperature_celsius"].skew())

        df["z_score"] = (
            df["temperature_celsius"] - df["temperature_celsius"].mean()
        ) / df["temperature_celsius"].std()

        st.subheader("Z-Score Anomalies")
        st.dataframe(df[abs(df["z_score"]) > 3].head(20))

    # ---------------- EXTREME EVENTS ---------------- #
    elif menu == "Extreme Events":

        st.title("🔥 Extreme Temperature Events")

        threshold = df["temperature_celsius"].mean() + df["temperature_celsius"].std()

        extreme_df = df[df["temperature_celsius"] > threshold]

        st.write("Threshold:", threshold)

        st.dataframe(extreme_df.head(20))

        extreme_count = extreme_df.groupby("country").size().reset_index(name="extreme_days")

        st.plotly_chart(px.bar(extreme_count, x="country", y="extreme_days"), use_container_width=True)

    # ---------------- LINE & DISTRIBUTION ---------------- #
    elif menu == "Line & Distribution Charts":

        st.title("📊 Climate Visualizations")

        yearly_temp = df.groupby(["year", "country"])["temperature_celsius"].mean().reset_index()

        st.plotly_chart(
            px.line(yearly_temp, x="year", y="temperature_celsius", color="country"),
            use_container_width=True
        )

        st.plotly_chart(px.histogram(df, x="temperature_celsius"), use_container_width=True)

        st.plotly_chart(px.violin(df, x="country", y="temperature_celsius"), use_container_width=True)

    # ---------------- CORRELATION HEATMAP ---------------- #
    elif menu == "Correlation Heatmap":

        st.title("📊 Correlation Heatmap")

        corr = df[["temperature_celsius", "precip_mm", "humidity", "wind_kph"]].corr()

        st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

    # ---------------- CHOROPLETH MAP ---------------- #
    elif menu == "Choropleth Map":

        st.title("🌍 Global Temperature Choropleth Map")

        country_avg = df.groupby("country")["temperature_celsius"].mean().reset_index()

        st.plotly_chart(
            px.choropleth(
                country_avg,
                locations="country",
                locationmode="country names",
                color="temperature_celsius",
                color_continuous_scale="RdYlBu_r"
            ),
            use_container_width=True
        )

    # ---------------- SEASONAL HEATMAP ---------------- #
    elif menu == "Seasonal Heatmap":

        st.title("🌡 Seasonal Temperature Heatmap")

        heat = df.groupby(["month", "country"])["temperature_celsius"].mean().reset_index()

        fig = px.density_heatmap(
            heat,
            x="month",
            y="country",
            z="temperature_celsius",
            color_continuous_scale="RdYlBu_r"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- WIND ANALYSIS ---------------- #
    elif menu == "Wind Analysis":

        st.title("💨 Wind Speed Analysis")

        fig = px.box(df, x="country", y="wind_kph", color="country")

        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            df,
            x="wind_kph",
            y="temperature_celsius",
            color="country"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- COUNTRY SIMILARITY ---------------- #
    elif menu == "Country Similarity":

        st.title("🌍 Country Climate Similarity")

        country_features = df.groupby("country").agg({
            "temperature_celsius":"mean",
            "precip_mm":"mean",
            "humidity":"mean",
            "wind_kph":"mean"
        }).reset_index()

        countries = country_features["country"].unique()

        col1, col2 = st.columns(2)

        with col1:
            country1 = st.selectbox("Country 1", countries)

        with col2:
            country2 = st.selectbox("Country 2", countries, index=1)

        if country1 != country2:

            scaler = StandardScaler()

            scaled_df = scaler.fit_transform(country_features.iloc[:,1:])

            similarity = cosine_similarity(
                [scaled_df[country_features[country_features["country"] == country1].index[0]]],
                [scaled_df[country_features[country_features["country"] == country2].index[0]]]
            )[0][0]

            st.success(f"Similarity: {round(similarity*100,2)}%")

    # ---------------- CLIMATE RANKING ---------------- #
    elif menu == "Climate Ranking":

        st.title("🏆 Climate Ranking")

        country_features = df.groupby("country").agg({
            "temperature_celsius":"mean",
            "precip_mm":"mean",
            "humidity":"mean",
            "wind_kph":"mean"
        }).reset_index()

        country_features["score"] = (
            country_features["temperature_celsius"]*0.4 +
            country_features["precip_mm"]*0.3 +
            country_features["humidity"]*0.2 +
            country_features["wind_kph"]*0.1
        )

        st.dataframe(country_features.sort_values("score", ascending=False).head(10))

    # ---------------- ANOMALY VISUALIZATION ---------------- #
    elif menu == "Anomaly Visualization":

        st.title("⚠ Climate Anomalies")

        df["z_score"] = (
            df["temperature_celsius"] - df["temperature_celsius"].mean()
        ) / df["temperature_celsius"].std()

        anomalies = df[abs(df["z_score"]) > 3]

        st.dataframe(anomalies.head(20))

        fig = px.scatter(
            anomalies,
            x="datetime",
            y="temperature_celsius",
            color="country"
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------- MAIN ---------------- #
if st.session_state.logged_in:
    dashboard()
else:
    login_page()