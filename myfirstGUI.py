import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
print(tf.__version__)


# Step 1: Set Streamlit page config to fixed layout (not full width)
st.set_page_config(page_title="RC Shear Wall Prediction", layout="centered")

# Custom CSS to make the UI more beautiful and control the aspect ratio
st.markdown("""
    <style>
        .title {
            color: #4C9E9F;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 3px;
            
        }
        .st-header {
            background-color: #B2D0D0;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            color: #003E3E;
        }
        .stInput, .stSelectbox, .stNumberInput {
            margin: 5px 0;
            background-color: #ffffff;
            border-radius: 5px;
            border: 1px solid #ddd;
            padding: 10px;
            width: 60%;
        }

        /* More specific selectors for streamlit header customization */
        .stTextInput, .stNumberInput, .stSelectbox, .stCheckbox {
            font-size: 14px;
        }

        /* Styling the headers of the input fields specifically */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-size: 18px !important;
        }

        /* Add padding to the left and right of the input fields */
        .stColumns {
            padding-left: 20%;
            padding-right: 20%;
        }

        /* Create space between left and right columns */
        .column-gap {
            display: flex;
            gap: 70px;  /* Increase space between columns */
            justify-content: center;
        }

        /* Reduce the overall width for a better appearance */
        .main-content {
            width: 60%;
            margin: 0 auto;
            padding: 20px;
        }

        /* Result container and icon */
        .result-container {
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 18px;
            color: #4C9E9F;
        }
        .result-icon {
            margin-right: 10px;
            font-size: 18px;
        }
        .result {
            background-color: #ffffff;
            color: #4C9E9F;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
        }
    </style>
""", unsafe_allow_html=True)

# Step 2: Try loading pre-trained scaler
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    # If scaler file does not exist, create and save it
    default_data = np.array([[5.0, 0.3, 8.0, 25, 415, 1.0, 100.0, 50.0, 100.0, 1, 1],
                             [6.0, 0.4, 10.0, 30, 500, 1.5, 150.0, 70.0, 200.0, 0, 0]])
    scaler = StandardScaler()
    scaler.fit(default_data)
    joblib.dump(scaler, 'scaler.pkl')

# Step 3: Load the trained model
model_path = r"F:/Graphical User Interface/GUI2/saved_model/model_saved_model.keras"
model = tf.keras.models.load_model(model_path)

# Streamlit Interface
st.title('Enter your RC shear wall data:')

# Add container div to control the size of the page
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Create space between left and right columns
    st.markdown('<div class="column-gap">', unsafe_allow_html=True)

    # Layout: Two columns for Geometrical and Material Properties
    col1, col2 = st.columns(2)

    with col1:
        st.header("Geometrical Properties")
        wall_height = st.number_input("Wall Height (m)", min_value=1.0, max_value=20.0, value=5.0)
        wall_thickness = st.number_input("Wall Thickness (m)", min_value=0.1, max_value=5.0, value=0.3)
        wall_length = st.number_input("Wall Length (m)", min_value=3.0, max_value=20.0, value=8.0)

    with col2:
        st.header("Material Properties")
        concrete_grade = st.selectbox("Concrete Grade (MPa)", [25, 30, 35, 40])
        steel_grade = st.selectbox("Steel Grade (MPa)", [415, 500, 550])
        reinforcement_ratio = st.number_input("Reinforcement Ratio (%)", min_value=0.5, max_value=2.5, value=1.0)

    # Layout: Two columns for Load Parameters and Boundary Conditions
    col3, col4 = st.columns(2)

    with col3:
        st.header("Load Parameters")
        axial_load = st.number_input("Axial Load (kN)", min_value=10.0, max_value=500.0, value=100.0)
        shear_load = st.number_input("Shear Load (kN)", min_value=10.0, max_value=500.0, value=50.0)
        overturning_moment = st.number_input("Overturning Moment (kNm)", min_value=10.0, max_value=1000.0, value=100.0)

    with col4:
        st.header("Boundary Conditions")
        support_type = st.selectbox("Support Type", [1, 0], format_func=lambda x: "Fixed" if x == 1 else "Hinged")
        loading_type = st.selectbox("Loading Type", [1, 0], format_func=lambda x: "Monotonic" if x == 1 else "Cyclic")

    st.markdown('</div>', unsafe_allow_html=True)  # Close column gap div

    # Prepare input data for prediction
    input_data = np.array([[wall_height, wall_thickness, wall_length,
                            concrete_grade, steel_grade, reinforcement_ratio,
                            axial_load, shear_load, overturning_moment,
                            support_type, loading_type]])

    # Normalize the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display prediction in a more stylish way with icon
    st.subheader("Predicted Energy Dissipation Capacity (kN.m)")
    st.markdown(f"""
        <div class="result-container">
            <i class="fas fa-cogs result-icon"></i>
            <div class="result">{prediction[0][0]:.2f} kN.m</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Include Font Awesome icons
st.markdown('<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">', unsafe_allow_html=True)
