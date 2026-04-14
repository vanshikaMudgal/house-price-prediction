# import streamlit as st
# import pandas as pd
# import numpy as np

# from sklearn.ensemble import IsolationForest, RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# import seaborn as sns
# import matplotlib.pyplot as plt

# # ---------- PAGE CONFIG ----------
# st.set_page_config(page_title="ML Dashboard", layout="wide")

# st.title("🚀 AI/ML Pipeline Dashboard")

# # ---------- SIDEBAR ----------
# st.sidebar.header("⚙️ Controls")

# uploaded_file = st.sidebar.file_uploader("Upload CSV")

# model_choice = st.sidebar.selectbox(
#     "Choose Model",
#     ["Linear Regression", "Random Forest"]
# )

# show_eda = st.sidebar.checkbox("Show EDA", True)
# remove_outliers = st.sidebar.checkbox("Remove Outliers", True)

# # ---------- LOAD DATA ----------
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
# else:
#     from sklearn.datasets import fetch_california_housing
#     data = fetch_california_housing()
#     df = pd.DataFrame(data.data, columns=data.feature_names)
#     df['median_house_value'] = data.target

# # ---------- LAYOUT ----------
# col1, col2 = st.columns(2)

# # ---------- DATA PREVIEW ----------
# with col1:
#     st.subheader("📂 Dataset Preview")
#     st.dataframe(df.head())

# with col2:
#     st.subheader("📏 Dataset Info")
#     st.write("Shape:", df.shape)
#     st.write("Columns:", list(df.columns))

# # ---------- EDA ----------
# if show_eda:
#     st.subheader("📊 Correlation Heatmap")

#     fig, ax = plt.subplots(figsize=(8,5))
#     sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# # ---------- FEATURE ENGINEERING ----------
# df["rooms_per_household"] = df.iloc[:,2] / df.iloc[:,4]
# df["bedrooms_per_rooms"] = df.iloc[:,3] / df.iloc[:,2]

# # ---------- SPLIT ----------
# X = df.drop("median_house_value", axis=1)
# y = df["median_house_value"]

# # ---------- OUTLIER ----------
# if remove_outliers:
#     iso = IsolationForest(contamination=0.05)
#     outliers = iso.fit_predict(X)
#     X = X[outliers == 1]
#     y = y[outliers == 1]

# # ---------- SCALING ----------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ---------- PCA ----------
# pca = PCA(n_components=0.95)
# X_pca = pca.fit_transform(X_scaled)

# st.subheader("📉 PCA Reduction")
# st.write("Reduced Shape:", X_pca.shape)

# # ---------- TRAIN TEST ----------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_pca, y, test_size=0.2, random_state=42
# )

# # ---------- MODEL ----------
# if model_choice == "Linear Regression":
#     model = LinearRegression()
# else:
#     model = RandomForestRegressor()

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # ---------- METRICS ----------
# st.subheader("📈 Model Performance")

# col3, col4 = st.columns(2)

# with col3:
#     st.metric("R2 Score", round(r2_score(y_test, y_pred), 3))

# with col4:
#     st.metric("Model Used", model_choice)

# # ---------- FOOTER ----------
# st.success("✅ Pipeline executed successfully")




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