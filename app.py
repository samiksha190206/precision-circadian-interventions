import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Precision Circadian Interventions",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "users.db"

KMEANS_PATH = BASE_DIR / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
LOGISTIC_PATH = BASE_DIR / "logistic_model.pkl"
RF_PATH = BASE_DIR / "random_forest_model.pkl"
GB_PATH = BASE_DIR / "gradient_boosting_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.pkl"

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    logistic_model = joblib.load(LOGISTIC_PATH)
    random_forest_model = joblib.load(RF_PATH)
    gradient_boosting_model = joblib.load(GB_PATH)

    try:
        feature_columns = joblib.load(FEATURE_PATH)
    except:
        feature_columns = [
            "age",
            "sad_hopeless",
            "physical_activity",
            "social_media_use",
            "sleep_duration",
            "sex_1.0",
            "sex_2.0"
        ]

    return (
        kmeans,
        scaler,
        logistic_model,
        random_forest_model,
        gradient_boosting_model,
        feature_columns
    )

try:

    (
        kmeans,
        scaler,
        logistic_model,
        random_forest_model,
        gradient_boosting_model,
        feature_columns
    ) = load_models()

except Exception as e:

    st.error("❌ Unable to load the trained models.")
    st.error(str(e))
    st.stop()

# ============================================================
# DATABASE
# ============================================================

def create_database():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            date TEXT NOT NULL,
            age REAL,
            sex TEXT,
            sad_hopeless REAL,
            physical_activity REAL,
            social_media_use REAL,
            sleep_duration REAL,
            cluster INTEGER,
            prediction INTEGER,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()

create_database()

# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()

# ============================================================
# REGISTER USER
# ============================================================

def register_user(username, password):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (username, password, created_at)
            VALUES (?, ?, ?)
            """,
            (
                username,
                hash_password(password),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

        conn.commit()

        success = True

    except sqlite3.IntegrityError:

        success = False

    conn.close()

    return success

# ============================================================
# LOGIN USER
# ============================================================

def login_user(username, password):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM users
        WHERE username = ? AND password = ?
        """,
        (
            username,
            hash_password(password)
        )
    )

    user = cursor.fetchone()

    conn.close()

    return user is not None

# ============================================================
# SAVE DAILY RECORD
# ============================================================

def save_record(
    username,
    age,
    sex,
    sad_hopeless,
    physical_activity,
    social_media_use,
    sleep_duration,
    cluster,
    prediction,
    confidence
):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO daily_records
        (
            username,
            date,
            age,
            sex,
            sad_hopeless,
            physical_activity,
            social_media_use,
            sleep_duration,
            cluster,
            prediction,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            age,
            sex,
            sad_hopeless,
            physical_activity,
            social_media_use,
            sleep_duration,
            cluster,
            prediction,
            confidence
        )
    )

    conn.commit()

    conn.close()

# ============================================================
# GET USER HISTORY
# ============================================================

def get_history(username):

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            date,
            sleep_duration,
            physical_activity,
            social_media_use,
            sad_hopeless,
            cluster,
            prediction,
            confidence
        FROM daily_records
        WHERE username = ?
        ORDER BY date ASC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(username,)
    )

    conn.close()

    return df

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background:
        radial-gradient(
            circle at 10% 10%,
            rgba(88, 60, 160, 0.35),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(25, 90, 150, 0.30),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #070b18 0%,
            #10152b 50%,
            #090d1d 100%
        );
    }

    /* Main content */

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Headings */

    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    /* Cards */

    .custom-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(20, 27, 50, 0.78);
        border: 1px solid rgba(150, 150, 255, 0.16);
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }

    /* Hero */

    .hero {
        padding: 35px;
        border-radius: 24px;
        background:
        linear-gradient(
            135deg,
            rgba(54, 40, 120, 0.85),
            rgba(18, 58, 100, 0.82)
        );
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        opacity: 0.85;
    }

    /* Metric cards */

    .metric-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(24, 32, 58, 0.85);
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 130px;
    }

    .metric-title {
        font-size: 14px;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 800;
        margin-top: 10px;
    }

    /* Recommendation */

    .recommendation {
        padding: 17px;
        border-radius: 14px;
        background: rgba(24, 66, 55, 0.75);
        border-left: 5px solid #55d6a5;
        margin-bottom: 12px;
    }

    /* Login */

    .login-box {
        max-width: 550px;
        margin: auto;
        padding: 35px;
        border-radius: 25px;
        background: rgba(15, 22, 43, 0.88);
        border: 1px solid rgba(150,150,255,0.20);
        box-shadow: 0 15px 50px rgba(0,0,0,0.40);
    }

    .brand {
        text-align: center;
        font-size: 32px;
        font-weight: 800;
    }

    .brand-subtitle {
        text-align: center;
        opacity: 0.65;
        margin-bottom: 30px;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 12px;
        min-height: 45px;
        font-weight: 700;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #0c1228,
            #090d1c
        );
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ============================================================
# LOGIN / REGISTER PAGE
# ============================================================

if not st.session_state.logged_in:

    st.markdown("<br><br>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown(
            """
            <div class="login-box">

            <div class="brand">
            🌙 Precision Circadian
            </div>

            <div class="brand-subtitle">
            INTERVENTIONS
            <br>
            Understand Today • Improve Tomorrow
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        login_tab, register_tab = st.tabs(
            ["🔐 Sign In", "✨ Create Account"]
        )

        # ====================================================
        # LOGIN
        # ====================================================

        with login_tab:

            st.subheader("Welcome Back 👋")

            username = st.text_input(
                "Username",
                key="login_username"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password"
            )

            if st.button(
                "Sign In →",
                use_container_width=True,
                type="primary"
            ):

                if username.strip() == "" or password == "":

                    st.warning(
                        "Please enter username and password."
                    )

                elif login_user(username, password):

                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = "Dashboard"

                    st.success("Login successful!")

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

        # ====================================================
        # REGISTER
        # ====================================================

        with register_tab:

            st.subheader("Create Your Account ✨")

            new_username = st.text_input(
                "Choose Username",
                key="register_username"
            )

            new_password = st.text_input(
                "Create Password",
                type="password",
                key="register_password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password"
            )

            if st.button(
                "Create Account",
                use_container_width=True,
                type="primary"
            ):

                if (
                    new_username.strip() == ""
                    or new_password == ""
                ):

                    st.warning(
                        "Please fill in all fields."
                    )

                elif new_password != confirm_password:

                    st.error(
                        "Passwords do not match."
                    )

                elif len(new_password) < 4:

                    st.warning(
                        "Password should contain at least 4 characters."
                    )

                elif register_user(
                    new_username,
                    new_password
                ):

                    st.success(
                        "Account created successfully! "
                        "You can now sign in."
                    )

                else:

                    st.error(
                        "Username already exists."
                    )

    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
        text-align:center;
        padding:15px;
        ">

        <div style="font-size:45px;">
        🌙
        </div>

        <h2>
        Precision<br>
        Circadian
        </h2>

        <p style="opacity:0.65;">
        INTERVENTIONS
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.write(
        f"👋 **Hello, {st.session_state.username}**"
    )

    st.caption(
        "Your personal wellness dashboard"
    )

    st.divider()

    if st.button(
        "🏠 Dashboard",
        use_container_width=True
    ):
        st.session_state.page = "Dashboard"
        st.rerun()

    if st.button(
        "📋 Daily Check-in",
        use_container_width=True
    ):
        st.session_state.page = "Daily Check-in"
        st.rerun()

    if st.button(
        "📈 My Progress",
        use_container_width=True
    ):
        st.session_state.page = "My Progress"
        st.rerun()

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "Dashboard"

        st.rerun()

# ============================================================
# GET HISTORY
# ============================================================

history = get_history(
    st.session_state.username
)

# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        Your Circadian Rhythm, Your Well-being 🌙
        </div>

        <div class="hero-subtitle">
        AI-powered behavioral analysis and personalized
        daily recommendations.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        f"Good morning, {st.session_state.username} 👋"
    )

    st.write(
        "Track your daily habits and understand how "
        "your behavioral patterns may influence your well-being."
    )

    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    if len(history) > 0:

        latest = history.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-title">
                😴 Sleep
                </div>

                <div class="metric-value">
                {latest['sleep_duration']:.1f} hrs
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-title">
                🏃 Activity
                </div>

                <div class="metric-value">
                {latest['physical_activity']:.0f}/10
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-title">
                📱 Social Media
                </div>

                <div class="metric-value">
                {latest['social_media_use']:.0f}/10
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-title">
                🧠 Sad/Hopeless
                </div>

                <div class="metric-value">
                {latest['sad_hopeless']:.0f}/10
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c5:

            prediction_text = (
                "Higher Difficulty"
                if latest["prediction"] == 1
                else "Lower Difficulty"
            )

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-title">
                🧠 Prediction
                </div>

                <div class="metric-value">
                {prediction_text}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ====================================================
        # PROFILE
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="custom-card">

                <h3>🧠 Behavioral Profile</h3>

                <p>
                Your current behavioral pattern was assigned
                to:
                </p>

                <h2>
                Cluster {int(latest['cluster'])}
                </h2>

                <p style="opacity:0.7;">
                This profile is determined using K-Means
                clustering from your behavioral features.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="custom-card">

                <h3>🎯 Prediction Confidence</h3>

                <h1>
                {latest['confidence']:.1f}%
                </h1>

                <p style="opacity:0.7;">
                Confidence of the Random Forest prediction
                for your latest check-in.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "🌱 You haven't completed your first daily check-in yet."
        )

        st.markdown(
            """
            <div class="custom-card">

            <h2>Start Your Journey</h2>

            <p>
            Complete your first daily check-in so the system
            can analyze your current behavioral pattern and
            generate personalized recommendations.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.subheader("How Your System Works")

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="custom-card">

            <h2>📋 01</h2>

            <h3>Daily Input</h3>

            <p>
            Enter your current sleep, activity,
            social-media use and mood information.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            """
            <div class="custom-card">

            <h2>🤖 02</h2>

            <h3>ML Analysis</h3>

            <p>
            K-Means identifies your behavioral profile
            and Random Forest predicts mental-health difficulty.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c:

        st.markdown(
            """
            <div class="custom-card">

            <h2>💡 03</h2>

            <h3>Personalized Guidance</h3>

            <p>
            Recommendations are generated from your
            current daily measurements and ML profile.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# DAILY CHECK-IN
# ============================================================

elif st.session_state.page == "Daily Check-in":

    st.title("📋 Daily Check-in")

    st.write(
        "Enter today's information. The system will analyze "
        "your current behavioral pattern."
    )

    st.divider()

    col1, col2 = st.columns(2)

    # ========================================================
    # PERSONAL INFORMATION
    # ========================================================

    with col1:

        st.subheader("👤 Personal Information")

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=20
        )

        sex = st.selectbox(
            "Sex",
            [
                "Female",
                "Male"
            ]
        )

        st.subheader("🧠 Well-being")

        sad_hopeless = st.slider(
            "How often have you felt sad or hopeless today?",
            min_value=0,
            max_value=10,
            value=1
        )

        st.caption(
            "0 = Not at all | 5 = Moderate | 10 = Very frequent"
        )

    # ========================================================
    # DAILY HABITS
    # ========================================================

    with col2:

        st.subheader("🌿 Daily Habits")

        physical_activity = st.slider(
            "Physical Activity Level",
            min_value=0,
            max_value=10,
            value=5
        )

        st.caption(
            "0 = Very Low | 5 = Moderate | 10 = Very High"
        )

        social_media_use = st.slider(
            "Social Media Usage",
            min_value=0,
            max_value=10,
            value=5
        )

        st.caption(
            "0 = Very Low | 5 = Moderate | 10 = Very High"
        )

        sleep_duration = st.slider(
            "Sleep Duration (hours)",
            min_value=0.0,
            max_value=12.0,
            value=7.0,
            step=0.5
        )

    st.divider()

    if st.button(
        "🔍 Analyze My Day",
        type="primary",
        use_container_width=True
    ):

        # ====================================================
        # CREATE MODEL INPUT
        # ====================================================

        user_data = {
            "age": age,
            "sad_hopeless": sad_hopeless,
            "physical_activity": physical_activity,
            "social_media_use": social_media_use,
            "sleep_duration": sleep_duration,

            "sex_1.0":
                1 if sex == "Female" else 0,

            "sex_2.0":
                1 if sex == "Male" else 0
        }

        user_df = pd.DataFrame(
            [user_data]
        )

        # Make sure columns are in training order

        user_df = user_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ====================================================
        # SCALE USER DATA
        # ====================================================

        user_scaled = scaler.transform(
            user_df
        )

        # ====================================================
        # K-MEANS
        # ====================================================

        user_cluster = int(
            kmeans.predict(
                user_scaled
            )[0]
        )

        # ====================================================
        # RANDOM FOREST
        # ====================================================

        prediction = int(
            random_forest_model.predict(
                user_scaled
            )[0]
        )

        probability = (
            random_forest_model
            .predict_proba(user_scaled)[0]
        )

        confidence = (
            max(probability) * 100
        )

        # ====================================================
        # STORE TEMPORARY RESULT
        # ====================================================

        st.session_state.last_result = {
            "age": age,
            "sex": sex,
            "sad_hopeless": sad_hopeless,
            "physical_activity": physical_activity,
            "social_media_use": social_media_use,
            "sleep_duration": sleep_duration,
            "cluster": user_cluster,
            "prediction": prediction,
            "confidence": confidence
        }

        st.success(
            "✅ Analysis completed successfully!"
        )

    # ========================================================
    # SHOW RESULT
    # ========================================================

    if "last_result" in st.session_state:

        result = st.session_state.last_result

        st.divider()

        st.header("✨ Your Analysis")

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Behavioral Profile",
                f"Cluster {result['cluster']}"
            )

        with r2:

            prediction_text = (
                "Higher Difficulty"
                if result["prediction"] == 1
                else "Lower Difficulty"
            )

            st.metric(
                "Mental-Health Prediction",
                prediction_text
            )

        with r3:

            st.metric(
                "Prediction Confidence",
                f"{result['confidence']:.1f}%"
            )

        # ====================================================
        # PERSONALIZED RECOMMENDATIONS
        # ====================================================

        st.subheader(
            "💡 Personalized Recommendations"
        )

        recommendations = []

        # Sleep recommendation

        if result["sleep_duration"] < 6:

            recommendations.append(
                "😴 Your current sleep duration is low. "
                "Try to gradually increase your sleep and "
                "maintain a consistent bedtime and wake-up time."
            )

        elif result["sleep_duration"] < 7:

            recommendations.append(
                "😴 Your sleep duration is slightly below the "
                "recommended range. Try to extend your sleep "
                "and maintain a regular schedule."
            )

        else:

            recommendations.append(
                "😴 Your current sleep duration is adequate. "
                "Continue maintaining a regular sleep schedule."
            )

        # Activity recommendation

        if result["physical_activity"] < 4:

            recommendations.append(
                "🏃 Your physical activity level is low. "
                "Try adding light walking or another comfortable "
                "form of daily movement."
            )

        elif result["physical_activity"] < 7:

            recommendations.append(
                "🏃 Your physical activity level is moderate. "
                "Try to maintain regular daily movement."
            )

        else:

            recommendations.append(
                "🏃 Your activity level is strong. "
                "Continue maintaining regular physical activity."
            )

        # Social media recommendation

        if result["social_media_use"] >= 8:

            recommendations.append(
                "📱 Your social-media usage is high. "
                "Reducing screen exposure, especially before "
                "bedtime, may support better sleep habits."
            )

        elif result["social_media_use"] >= 5:

            recommendations.append(
                "📱 Your social-media usage is moderate. "
                "Avoid excessive use close to bedtime."
            )

        else:

            recommendations.append(
                "📱 Your social-media usage is relatively low. "
                "Continue maintaining healthy screen habits."
            )

        # Mood recommendation

        if result["sad_hopeless"] >= 7:

            recommendations.append(
                "💙 Your reported sadness/hopelessness is high. "
                "Consider talking to a trusted person or qualified "
                "mental-health professional if these feelings persist."
            )

        elif result["sad_hopeless"] >= 4:

            recommendations.append(
                "💙 Your reported sadness/hopelessness is moderate. "
                "Pay attention to changes in your mood and maintain "
                "healthy sleep, activity and social routines."
            )

        else:

            recommendations.append(
                "💙 Your reported sadness/hopelessness is currently low. "
                "Continue maintaining supportive daily habits."
            )

        # ====================================================
        # DISPLAY RECOMMENDATIONS
        # ====================================================

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation">
                {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.info(
            f"""
            🧠 Your current behavioral characteristics were
            assigned to **Cluster {result['cluster']}** by the
            K-Means model.

            The Random Forest model predicted:
            **{prediction_text}**
            with **{result['confidence']:.1f}% confidence**.
            """
        )

        # ====================================================
        # SAVE BUTTON
        # ====================================================

        if st.button(
            "💾 Save Today's Record",
            type="primary",
            use_container_width=True
        ):

            save_record(
                st.session_state.username,
                result["age"],
                result["sex"],
                result["sad_hopeless"],
                result["physical_activity"],
                result["social_media_use"],
                result["sleep_duration"],
                result["cluster"],
                result["prediction"],
                result["confidence"]
            )

            st.success(
                "✅ Today's record has been saved to your personal history!"
            )

            del st.session_state.last_result

# ============================================================
# PROGRESS PAGE
# ============================================================

elif st.session_state.page == "My Progress":

    st.title("📈 My Progress")

    st.write(
        "Track how your daily behavioral patterns change over time."
    )

    st.divider()

    if len(history) == 0:

        st.info(
            "🌱 No records yet. Complete your first Daily Check-in "
            "to start tracking your progress."
        )

    else:

        # ====================================================
        # SUMMARY
        # ====================================================

        latest = history.iloc[-1]

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Total Check-ins",
                len(history)
            )

        with c2:

            st.metric(
                "Average Sleep",
                f"{history['sleep_duration'].mean():.1f} hrs"
            )

        with c3:

            st.metric(
                "Average Activity",
                f"{history['physical_activity'].mean():.1f}/10"
            )

        st.divider()

        # ====================================================
        # SLEEP TREND
        # ====================================================

        st.subheader("😴 Sleep Trend")

        fig_sleep = px.line(
            history,
            x="date",
            y="sleep_duration",
            markers=True,
            title="Daily Sleep Duration"
        )

        fig_sleep.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_sleep,
            use_container_width=True
        )

        # ====================================================
        # ACTIVITY TREND
        # ====================================================

        st.subheader("🏃 Physical Activity")

        fig_activity = px.line(
            history,
            x="date",
            y="physical_activity",
            markers=True,
            title="Daily Physical Activity"
        )

        fig_activity.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_activity,
            use_container_width=True
        )

        # ====================================================
        # SOCIAL MEDIA
        # ====================================================

        st.subheader("📱 Social Media Usage")

        fig_social = px.line(
            history,
            x="date",
            y="social_media_use",
            markers=True,
            title="Daily Social Media Usage"
        )

        fig_social.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_social,
            use_container_width=True
        )

        # ====================================================
        # MOOD
        # ====================================================

        st.subheader("💙 Sadness / Hopelessness")

        fig_mood = px.line(
            history,
            x="date",
            y="sad_hopeless",
            markers=True,
            title="Daily Reported Sadness / Hopelessness"
        )

        fig_mood.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_mood,
            use_container_width=True
        )

        # ====================================================
        # RECORD TABLE
        # ====================================================

        st.subheader("📋 Your Daily Records")

        display_df = history.copy()

        display_df["prediction"] = display_df[
            "prediction"
        ].map({
            0: "Lower Difficulty",
            1: "Higher Difficulty"
        })

        display_df["confidence"] = (
            display_df["confidence"]
            .round(1)
            .astype(str)
            + "%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
