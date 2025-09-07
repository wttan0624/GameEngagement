# app.py - Enhanced Streamlit Dashboard for Engagement Prediction

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from xgboost import XGBClassifier


# -----------------------------
# Load Trained Model
# -----------------------------
model = XGBClassifier()
model.load_model("xgb_model.json")  # saved earlier

# -----------------------------
# Dashboard Title
# -----------------------------
st.set_page_config(page_title="Gaming Engagement Dashboard", layout="wide")
st.title("🎮 Gaming Engagement Prediction Dashboard")

# -----------------------------
# Sidebar - Player Input
# -----------------------------
st.sidebar.header("Player Data Input")

sessions = st.sidebar.number_input("Sessions per Week", min_value=1, max_value=100, value=10)
avg_duration = st.sidebar.number_input("Average Session Duration (minutes)", min_value=1, max_value=600, value=60)
player_level = st.sidebar.number_input("Player Level", min_value=1, max_value=100, value=5)
age = st.sidebar.number_input("Age", min_value=10, max_value=80, value=20)

# Auto calculate Total Weekly Playtime (hours)
total_playtime = (sessions * avg_duration) / 60  

# Make input dataframe
X_input = pd.DataFrame({
    "TotalWeeklyPlaytime": [total_playtime],
    "SessionsPerWeek": [sessions],
    "AvgSessionDurationMinutes": [avg_duration],
    "PlayerLevel": [player_level],
    "Age": [age]
})

# -----------------------------
# Example Data for Visualisations
# (In real use, replace with real dataset)
# -----------------------------
df = pd.read_csv('online_gaming_behavior_dataset.csv')

# Add derived feature
df["TotalWeeklyPlaytime"] = df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"] / 60

# -----------------------------
# Layout: High-Level Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("📈 Avg Playtime (hrs)", f"{df['TotalWeeklyPlaytime'].mean():.1f}")
col2.metric("🕹️ Avg Sessions/Week", f"{df['SessionsPerWeek'].mean():.1f}")
col3.metric("👥 Player Count", f"{len(df)}")

# -----------------------------
# Engagement Distribution
# -----------------------------
st.subheader("📊 Engagement Distribution")
dist_fig = px.histogram(df, x="EngagementLevel", color="EngagementLevel")
st.plotly_chart(dist_fig, use_container_width=True)

# -----------------------------
# Feature Importance (placeholder example)
# -----------------------------
st.subheader("🔥 Feature Importance")
importance_values = model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": ["TotalWeeklyPlaytime", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "Age"],
    "Importance": importance_values 
}).sort_values("Importance", ascending=False)
imp_fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h")
st.plotly_chart(imp_fig, use_container_width=True)

# -----------------------------
# Feature Distributions by Engagement
# -----------------------------
st.subheader("📌 Feature Distributions by Engagement Level")
for feature in ["TotalWeeklyPlaytime", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "Age"]:
    fig = px.box(df, x="EngagementLevel", y=feature, color="EngagementLevel",
                 title=f"{feature} Distribution by Engagement Level")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Segmentation: Age vs. Engagement
# -----------------------------
st.subheader("👥 Age Distribution by Engagement")
age_fig = px.histogram(df, x="Age", color="EngagementLevel", barmode="overlay")
st.plotly_chart(age_fig, use_container_width=True)


# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Engagement"):
    prediction = model.predict(X_input)[0]
    engagement_levels = {0: "Low", 1: "Medium", 2: "High"}  # adjust to your encoding
    st.metric("Predicted Engagement Level", engagement_levels[prediction])
