import streamlit as st
import pickle
import numpy as np
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- CUSTOM CSS FOR ATTRACTIVE LAYOUT & ANIMATIONS ---
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    .title-text {
        text-align: center;
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle-text {
        text-align: center;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Result Card Styling */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: white;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.8s ease-in-out;
    }
    .pass-card {
        background: linear-gradient(135deg, #10B981, #059669);
    }
    .fail-card {
        background: linear-gradient(135deg, #EF4444, #DC2626);
    }

    /* Keyframe animation for result display */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# --- HEADER SECTION ---
st.markdown("<h1 class='title-text'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Enter marks for each subject to predict the final Pass/Fail status.</p>", unsafe_allow_html=True)
st.divider()

# --- INPUT FORM ---
st.subheader("📊 Subject Marks")

col1, col2 = st.columns(2)

with col1:
    hindi = st.number_input("Hindi Marks", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    english = st.number_input("English Marks", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    science = st.number_input("Science Marks", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

with col2:
    maths = st.number_input("Maths Marks", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    history = st.number_input("History Marks", min_value=0.0, max_value=100.0, value=68.0, step=1.0)
    geography = st.number_input("Geography Marks", min_value=0.0, max_value=100.0, value=72.0, step=1.0)

# Automatically compute total marks as expected by your model
total = hindi + english + science + maths + history + geography
st.info(f"**Calculated Total Marks:** {total:.2f} / 600.00")

# --- PREDICTION SECTION ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Predict Result", use_container_width=True, type="primary"):
    
    # Animated Loading State
    with st.spinner("Analyzing performance data..."):
        time.sleep(1)  # Brief visual pause for smooth transition
        
        # Prepare feature vector matching feature_names_in_:
        # ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geograpgy', 'Total']
        features = np.array([[hindi, english, science, maths, history, geography, total]])
        
        prediction = model.predict(features)[0]

    # --- DISPLAY ANIMATED RESULT ---
    if prediction == 1 or str(prediction).lower() == 'pass':
        st.balloons()
        st.markdown(
            "<div class='result-card pass-card'>🎉 Result: PASSED</div>", 
            unsafe_allow_html=True
        )
    else:
        st.snow()
        st.markdown(
            "<div class='result-card fail-card'>⚠️ Result: FAILED</div>", 
            unsafe_allow_html=True
        )
