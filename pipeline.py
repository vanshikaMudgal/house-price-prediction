

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("🚀 Professional ML Pipeline Dashboard")

# ---------- FILE UPLOAD ----------
file = st.file_uploader("📂 Upload your CSV file")

if file:
    df = pd.read_csv(file)
else:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['median_house_value'] = data.target

# ---------- TARGET SELECT ----------
target = st.selectbox("🎯 Select Target Column", df.columns)

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data", "📊 EDA", "🧹 Cleaning", "📉 PCA", "🤖 Model"
])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

# ---------- TAB 2 ----------
with tab2:
    st.subheader("EDA - Interactive Graph")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("X-axis", df.columns)

    with col2:
        y_axis = st.selectbox("Y-axis", df.columns)

    fig = px.scatter(df, x=x_axis, y=y_axis)
    st.plotly_chart(fig)

# ---------- TAB 3 ----------
with tab3:
    st.subheader("Outlier Removal")

    X = df.drop(target, axis=1)
    y = df[target]

    iso = IsolationForest(contamination=0.05)
    outliers = iso.fit_predict(X)

    X = X[outliers == 1]
    y = y[outliers == 1]

    st.write("After removing outliers:", X.shape)

# ---------- TAB 4 ----------
with tab4:
    st.subheader("PCA Reduction")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    st.write("Reduced Shape:", X_pca.shape)

# ---------- TAB 5 ----------
with tab5:
    st.subheader("Model Training")

    model_choice = st.selectbox(
        "Choose Model",
        ["Linear Regression", "Random Forest"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("R2 Score", round(r2_score(y_test, y_pred), 3))
    st.success("Model trained successfully 🚀")