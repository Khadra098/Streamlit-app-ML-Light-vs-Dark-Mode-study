import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("/Users/khadramahamoud/Documents/RESULTS/Final_Combined_All_Data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Feature engineering
df["time_diff"] = df["task_time_light_sec"] - df["task_time_dark_sec"]
df["typo_diff"] = df["typos_light"] - df["typos_dark"]

st.set_page_config(layout="wide")
st.title("📊 Study Dashboard: Light vs. Dark Mode Analysis")

with st.expander("📌 Study Overview"):
    st.markdown(f"**Participants:** {len(df)}")
    st.markdown("This dashboard explores user performance, comfort, satisfaction, and preferences in light vs. dark interface modes.")

# Task Performance
st.markdown("### ⏱️ Task Performance")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Average Completion Time**")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.bar(["Light Mode", "Dark Mode"], [df["task_time_light_sec"].mean(), df["task_time_dark_sec"].mean()], color=["#3498db", "#2ecc71"])
    ax1.set_ylabel("Time (seconds)")
    st.pyplot(fig1)
with col2:
    st.markdown("**Average Typos**")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(["Light Mode", "Dark Mode"], [df["typos_light"].mean(), df["typos_dark"].mean()], color=["#e67e22", "#9b59b6"])
    ax2.set_ylabel("Typos")
    st.pyplot(fig2)

# Comfort and Satisfaction
st.markdown("### 😌 Comfort & 😊 Satisfaction Comparison")
col3, col4 = st.columns(2)
with col3:
    st.markdown("**Comfort Scores**")
    df_melted = df.melt(value_vars=["comfort_light", "comfort_dark"], var_name="Mode", value_name="Score")
    df_melted["Mode"] = df_melted["Mode"].str.replace("comfort_", "").str.title()
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    sns.boxplot(data=df_melted, x="Mode", y="Score", palette="pastel", ax=ax3)
    st.pyplot(fig3)
with col4:
    st.markdown("**Satisfaction Scores**")
    df_melted2 = df.melt(value_vars=["satisfaction_light", "satisfaction_dark"], var_name="Mode", value_name="Score")
    df_melted2["Mode"] = df_melted2["Mode"].str.replace("satisfaction_", "").str.title()
    fig4, ax4 = plt.subplots(figsize=(4, 3))
    sns.boxplot(data=df_melted2, x="Mode", y="Score", palette="pastel", ax=ax4)
    st.pyplot(fig4)

# Demographic Preferences and Performance
st.markdown("### 👥 Demographic Preferences vs Performance")
col5, col6 = st.columns(2)
with col5:
    st.markdown("**Mode Preference by Age Group**")
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    age_mode = df.groupby("age_group")["preferred_mode"].value_counts().unstack().fillna(0)
    age_mode.plot(kind="bar", stacked=True, colormap="Set2", ax=ax5)
    ax5.set_ylabel("Participants")
    st.pyplot(fig5)

with col6:
    st.markdown("**Performance by Age Group**")
    fig6, ax6 = plt.subplots(figsize=(5, 3))
    df_age_perf = df.groupby("age_group")[["task_time_light_sec", "task_time_dark_sec"]].mean()
    df_age_perf.plot(kind="bar", colormap="coolwarm", ax=ax6)
    ax6.set_ylabel("Time (seconds)")
    st.pyplot(fig6)

# Performance vs Preference and Break Analysis side-by-side
st.markdown("### 🔄 Performance vs Preference & ☕ Breaks")
col7, col8 = st.columns(2)
with col7:
    st.markdown("**Performance Difference by Preferred Mode**")
    fig7, ax7 = plt.subplots(figsize=(5, 3))
    sns.boxplot(x="preferred_mode", y="time_diff", data=df, palette="Set3", ax=ax7)
    ax7.axhline(0, color='gray', linestyle='--')
    ax7.set_ylabel("Time Diff (Light - Dark)")
    st.pyplot(fig7)

with col8:
    st.markdown("**Breaks and Average Typos**")
    fig8, ax8 = plt.subplots(figsize=(5, 3))
    break_perf = df.groupby("breaks_taken")[["typos_light", "typos_dark"]].mean()
    break_perf.plot(kind="bar", colormap="Accent", ax=ax8)
    ax8.set_ylabel("Avg. Typos")
    st.pyplot(fig8)

st.markdown("---")
st.markdown("📌 Use this dashboard to uncover mode-related trends and guide interface design choices.")
