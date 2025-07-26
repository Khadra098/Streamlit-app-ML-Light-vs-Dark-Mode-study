# unified_app.py
import streamlit as st
import joblib
import os



# Load models and encoders
user_model = joblib.load("logreg_mode_predictor.pkl")
designer_model = joblib.load("designer_mode_predictor.pkl")
target_encoder = joblib.load("target_encoder.pkl")
user_encoders = joblib.load("label_encoders.pkl")
designer_encoders = joblib.load("designer_label_encoders.pkl")

# --- Sidebar Navigation ---

st.sidebar.title("🧭 Choose Prediction Type")
selection = st.sidebar.radio("Go to:", ["🏠 Welcome", "👤 Personal Prediction", "🎨 Designer Audience Prediction",  "📝 Submit Your Preference", "🔁 Retrain Model", "📊 Visualize Preferences", "📘 Study Summary & Results"])

# --- Welcome Page ---
if selection == "🏠 Welcome":
    st.title("🌗 Light or Dark Mode Preference Predictor")

    st.markdown("""
    👋 **Welcome to the Mode Preference App!**

    This interactive tool helps you **predict and understand whether users prefer Light Mode or Dark Mode** — based on their behavior, comfort, and demographics.

    ---
    ### 🧠 What You Can Do:
    """)

    st.markdown("""
    - 🔍 **Personal Prediction**  
      Find out your own mode preference based on screen time, comfort levels, and more.

    - 🎯 **Designer Audience Prediction**  
      Target your design better by predicting what a specific demographic might prefer.

    - 📝 **Submit Your Preference**  
      Help improve the model by submitting your personal preferences anonymously.

    - 🔁 **Retrain the Model**  
      Rebuild the model with all feedback collected so far — improve its accuracy with real user data.

    - 📊 **Visualize Preferences**  
      Explore how different user groups (age, gender, screen time) lean toward Light or Dark Mode with live charts.
    """)

    st.markdown("---")

    # Add buttons that link users to other pages (via a note)
    st.subheader("🚀 Ready to get started?")
    st.info("👉 Use the **left sidebar** to choose a feature.")

    st.markdown("Have fun exploring the world of user preferences — and thank you for helping make smarter, more accessible designs!")


# --- Personal Prediction ---
elif selection == "👤 Personal Prediction":
    st.title("👤 Personal Mode Preference Predictor")

    # Inputs
    age = st.selectbox("Age Group", user_encoders['age_group'].classes_)
    gender = st.selectbox("Gender", user_encoders['gender'].classes_)
    device_usage = st.selectbox("Device usage frequency", user_encoders['device_usage_frequency'].classes_)
    primary_use = st.selectbox("Primary digital use", user_encoders['primary_use'].classes_)
    usual_mode = st.selectbox("Current mode preference", user_encoders['usual_mode'].classes_)
    eye_strain_exp = st.selectbox("Eye strain experience", user_encoders['eye_strain_experience'].classes_)
    mode_factors = st.selectbox("What influences mode choice?", user_encoders['mode_choice_factors'].classes_)
    screen_time = st.selectbox("Daily screen time", user_encoders['daily_screen_time'].classes_)

    comfort_light = st.slider("Comfort in light mode", 1, 5, 3)
    comfort_dark = st.slider("Comfort in dark mode", 1, 5, 3)
    eye_strain_light = st.slider("Eye strain in light mode", 1, 5, 3)
    eye_strain_dark = st.slider("Eye strain in dark mode", 1, 5, 3)
    focus_light = st.slider("Focus in light mode", 1, 5, 3)
    focus_dark = st.slider("Focus in dark mode", 1, 5, 3)

    # Feature engineering
    features = [
        user_encoders['age_group'].transform([age])[0],
        user_encoders['gender'].transform([gender])[0],
        user_encoders['device_usage_frequency'].transform([device_usage])[0],
        user_encoders['primary_use'].transform([primary_use])[0],
        user_encoders['usual_mode'].transform([usual_mode])[0],
        user_encoders['eye_strain_experience'].transform([eye_strain_exp])[0],
        user_encoders['mode_choice_factors'].transform([mode_factors])[0],
        user_encoders['daily_screen_time'].transform([screen_time])[0],
        comfort_light, comfort_dark,
        eye_strain_light, eye_strain_dark,
        focus_light, focus_dark,
        comfort_dark - comfort_light,
        focus_dark - focus_light,
        eye_strain_dark - eye_strain_light
    ]

    if st.button("Predict Personal Preference"):
        prediction = user_model.predict([features])[0]
        result = target_encoder.inverse_transform([prediction])[0]
        st.success(f"🎯 You likely prefer: **{result.title()} Mode**")

# --- Designer Audience Prediction ---
elif selection == "🎨 Designer Audience Prediction":
    st.title("🎨 Predict Mode Preference for Your Target Audience")

    age = st.selectbox("Audience Age Group", designer_encoders['age_group'].classes_)
    gender = st.selectbox("Audience Gender", designer_encoders['gender'].classes_)
    device_usage = st.selectbox("Device usage frequency", designer_encoders['device_usage_frequency'].classes_)
    primary_use = st.selectbox("Primary digital use", designer_encoders['primary_use'].classes_)
    usual_mode = st.selectbox("Most used mode", designer_encoders['usual_mode'].classes_)
    eye_strain_exp = st.selectbox("Eye strain experience?", designer_encoders['eye_strain_experience'].classes_)
    mode_factors = st.selectbox("Main influence on mode choice", designer_encoders['mode_choice_factors'].classes_)
    screen_time = st.selectbox("Typical daily screen time", designer_encoders['daily_screen_time'].classes_)

    audience_features = [
        designer_encoders['age_group'].transform([age])[0],
        designer_encoders['gender'].transform([gender])[0],
        designer_encoders['device_usage_frequency'].transform([device_usage])[0],
        designer_encoders['primary_use'].transform([primary_use])[0],
        designer_encoders['usual_mode'].transform([usual_mode])[0],
        designer_encoders['eye_strain_experience'].transform([eye_strain_exp])[0],
        designer_encoders['mode_choice_factors'].transform([mode_factors])[0],
        designer_encoders['daily_screen_time'].transform([screen_time])[0],
    ]

    if st.button("Predict Audience Preference"):
        prediction = designer_model.predict([audience_features])[0]
        result = target_encoder.inverse_transform([prediction])[0]
        st.success(f"🎯 Your audience is likely to prefer: **{result.title()} Mode**")





elif selection == "📝 Submit Your Preference":
    import pandas as pd
    import os

    DATA_FILE = "user_feedback.csv"

    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=[
            "age_group", "gender", "device_usage_frequency", "primary_use",
            "usual_mode", "eye_strain_experience", "mode_choice_factors",
            "daily_screen_time", "comfort_light", "comfort_dark",
            "eye_strain_light", "eye_strain_dark", "focus_light", "focus_dark",
            "mode_preference"
        ]).to_csv(DATA_FILE, index=False)

    st.title("📝 Submit Your Mode Preference")

    with st.form("user_feedback_form"):
        age = st.selectbox("Audience Age Group", designer_encoders['age_group'].classes_)
        gender = st.selectbox("Gender", user_encoders['gender'].classes_)
        device_usage = st.selectbox("Device usage frequency", user_encoders['device_usage_frequency'].classes_)
        primary_use = st.selectbox("Primary digital use", user_encoders['primary_use'].classes_)
        usual_mode = st.selectbox("Current mode preference", user_encoders['usual_mode'].classes_)
        eye_strain_exp = st.selectbox("Eye strain experience", user_encoders['eye_strain_experience'].classes_)
        mode_factors = st.selectbox("What influences mode choice?", user_encoders['mode_choice_factors'].classes_)
        screen_time = st.selectbox("Daily screen time", user_encoders['daily_screen_time'].classes_)

        comfort_light = st.slider("Comfort in light mode", 1, 5, 3)
        comfort_dark = st.slider("Comfort in dark mode", 1, 5, 3)
        eye_strain_light = st.slider("Eye strain in light mode", 1, 5, 3)
        eye_strain_dark = st.slider("Eye strain in dark mode", 1, 5, 3)
        focus_light = st.slider("Focus in light mode", 1, 5, 3)
        focus_dark = st.slider("Focus in dark mode", 1, 5, 3)

        mode_preference = st.selectbox("Which mode do you prefer?", target_encoder.classes_)

        submitted = st.form_submit_button("Submit")

    if submitted:
        new_data = pd.DataFrame([{
            "age_group": age,
            "gender": gender,
            "device_usage_frequency": device_usage,
            "primary_use": primary_use,
            "usual_mode": usual_mode,
            "eye_strain_experience": eye_strain_exp,
            "mode_choice_factors": mode_factors,
            "daily_screen_time": screen_time,
            "comfort_light": comfort_light,
            "comfort_dark": comfort_dark,
            "eye_strain_light": eye_strain_light,
            "eye_strain_dark": eye_strain_dark,
            "focus_light": focus_light,
            "focus_dark": focus_dark,
            "mode_preference": mode_preference
        }])
        new_data.to_csv(DATA_FILE, mode="a", header=False, index=False)
        st.success("✅ Thank you! Your data has been saved.")



    





elif selection == "🔁 Retrain Model":
    import pandas as pd
    import joblib
    import os
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    st.title("🔁 Retrain Model")

    DATA_FILE = "user_feedback.csv"
    MODEL_FILE = "logreg_mode_predictor.pkl"
    ENCODERS_FILE = "label_encoders.pkl"
    TARGET_ENCODER_FILE = "target_encoder.pkl"

    if not os.path.exists(DATA_FILE) or pd.read_csv(DATA_FILE).empty:
        st.warning("⚠️ No data available. Please submit data first.")
    else:
        df = pd.read_csv(DATA_FILE)

        cat_cols = [
            "age_group", "gender", "device_usage_frequency", "primary_use",
            "usual_mode", "eye_strain_experience", "mode_choice_factors",
            "daily_screen_time"
        ]

        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        target_encoder = LabelEncoder()
        df["mode_preference"] = target_encoder.fit_transform(df["mode_preference"])

        # Feature engineering
        df["comfort_diff"] = df["comfort_dark"] - df["comfort_light"]
        df["focus_diff"] = df["focus_dark"] - df["focus_light"]
        df["eye_strain_diff"] = df["eye_strain_dark"] - df["eye_strain_light"]

        X = df.drop(columns=["mode_preference"])
        y = df["mode_preference"]

        if len(set(y)) < 2:
            st.error("❌ Not enough class diversity to train. Please collect more diverse data (both Light & Dark preferences).")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(encoders, ENCODERS_FILE)
        joblib.dump(target_encoder, TARGET_ENCODER_FILE)

        st.success("✅ Model retrained and saved!")




elif selection == "📊 Visualize Preferences":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("📊 Visualize Mode Preferences")

    st.markdown("""
    This page helps you understand **how different user groups prefer light or dark mode** based on demographics and usage behavior.
    
    - The graphs below are based on **new user feedback** submitted through this app.
    - You can also preview the original dataset used during model training.
    - Use this data to inform design choices and better understand your audience!
    """)

    original_path = "demo dataset.xlsx"
    new_data_path = "user_feedback.csv"

    # Load original data
    try:
        original_df = pd.read_excel(original_path)
        st.success("✅ Original dataset loaded.")
    except Exception as e:
        original_df = pd.DataFrame()
        st.warning(f"⚠️ Could not load original dataset: {e}")

    # Load new data
    try:
        new_df = pd.read_csv(new_data_path)
        st.success("✅ New user-submitted data loaded.")
    except Exception as e:
        new_df = pd.DataFrame()
        st.warning(f"⚠️ No new feedback collected yet: {e}")

    # Download buttons
    st.markdown("### 📥 Download Datasets")
    col1, col2 = st.columns(2)
    with col1:
        if not new_df.empty:
            csv = new_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download New Data (CSV)", csv, "new_user_data.csv", "text/csv")
    with col2:
        if not original_df.empty:
            csv_orig = original_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Original Data (CSV)", csv_orig, "original_data.csv", "text/csv")

    # Optional original data preview
    with st.expander("🔎 Preview Original Dataset"):
        if not original_df.empty:
            st.markdown("This dataset was used to train the model before user submissions began.")
            st.dataframe(original_df.head(20))
        else:
            st.info("No original dataset available.")

    # Visualize new data
    if new_df.empty:
        st.error("🚫 No new user feedback found to visualize.")
    else:
        st.markdown("## 📊 Visualizing New Feedback Data")

        # Overall preference
        with st.expander("🎯 Overall Mode Preference (Pie Chart)"):
            st.markdown("Shows the percentage of users who prefer light or dark mode.")
            mode_counts = new_df["mode_preference"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

        # Age group
        with st.expander("👵 Mode Preference by Age Group"):
            st.markdown("Compare preferences across different age groups.")
            age_mode = new_df.groupby(["age_group", "mode_preference"]).size().unstack().fillna(0)
            fig2, ax2 = plt.subplots()
            age_mode.plot(kind="bar", stacked=True, ax=ax2)
            plt.ylabel("Number of Users")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        # Gender
        with st.expander("🚻 Mode Preference by Gender"):
            st.markdown("Explore how gender relates to mode choice.")
            gender_mode = new_df.groupby(["gender", "mode_preference"]).size().unstack().fillna(0)
            fig3, ax3 = plt.subplots()
            gender_mode.plot(kind="bar", stacked=True, ax=ax3)
            plt.ylabel("Number of Users")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

        # Screen time
        with st.expander("⏱️ Mode Preference by Daily Screen Time"):
            st.markdown("Understand how usage habits impact visual mode preference.")
            screen_mode = new_df.groupby(["daily_screen_time", "mode_preference"]).size().unstack().fillna(0)
            fig4, ax4 = plt.subplots()
            screen_mode.plot(kind="bar", stacked=True, ax=ax4)
            plt.ylabel("Number of Users")
            plt.xticks(rotation=45)
            st.pyplot(fig4)





elif selection == "📘 Study Summary & Results":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("📘 Study Summary & Results")

    st.markdown("""
    ### 🧪 About This Study  
    This study explores whether people perform better at a task in **Light Mode** or **Dark Mode**.  
    It includes two parts:

    1. **Pre-task questionnaire** — Users shared their mode preferences and screen habits.
    2. **Task performance** — Users completed the same task in both light and dark modes.

    This page shows key findings from that combined dataset. 👇
    """)

    # Load data
    df = pd.read_csv("/Users/khadramahamoud/Documents/RESULTS/Final_Combined_All_Data.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Feature engineering
    df["time_diff"] = df["task_time_light_sec"] - df["task_time_dark_sec"]
    df["typo_diff"] = df["typos_light"] - df["typos_dark"]

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

