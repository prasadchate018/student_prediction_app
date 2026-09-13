import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Academic Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        transform: translateY(-2px);
    }
    .prediction-card {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Trained Model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Title and Subtitle
st.title("🎓 Academic Performance Classifier")
st.write("Enter the student's scores across subjects to predict the classification result.")

st.divider()

# Input Form
with st.form("prediction_form"):
    st.subheader("Subject Marks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hindi = st.number_input("Hindi Marks", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        english = st.number_input("English Marks", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        science = st.number_input("Science Marks", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        maths = st.number_input("Maths Marks", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

    with col2:
        history = st.number_input("History Marks", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        geography = st.number_input("Geography Marks", min_value=0.0, max_value=100.0, value=72.0, step=1.0)
        
        # Calculate Total automatically
        calculated_total = hindi + english + science + maths + history + geography
        total = st.number_input("Total Marks", value=calculated_total, disabled=True)

    submit_button = st.form_submit_button("🔮 Predict Result")

# Prediction Trigger
if submit_button:
    with st.spinner("Processing input and running model inference..."):
        # Features array corresponding to model trained inputs
        input_data = np.array([[hindi, english, science, maths, history, geography, calculated_total]])
        prediction = model.predict(input_data)[0]
    
    # Visual Effects & Display Result
    st.balloons()
    
    st.markdown(f"""
        <div class="prediction-card">
            <h3>Prediction Outcome</h3>
            <h1 style="color: #FF4B4B; margin: 0;">Class {prediction}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.success("Inference completed successfully!")
