# import os
# import pickle
# import streamlit as st
# from streamlit_option_menu import option_menu

# # Set page configuration
# st.set_page_config(page_title="Health Assistant",
#                    layout="wide",
#                    page_icon="🧑‍⚕️")

    
# # getting the working directory of the main.py
# working_dir = os.path.dirname(os.path.abspath(__file__))

# # loading the saved models

# diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# # sidebar for navigation
# with st.sidebar:
#     selected = option_menu('Multiple Disease Prediction System',

#                            ['Diabetes Prediction',
#                             'Heart Disease Prediction',
#                             'Parkinsons Prediction'],
#                            menu_icon='hospital-fill',
#                            icons=['activity', 'heart', 'person'],
#                            default_index=0)


# # Diabetes Prediction Page
# if selected == 'Diabetes Prediction':

#     # page title
#     st.title('Diabetes Prediction using ML')

#     # getting the input data from the user
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         Pregnancies = st.text_input('Number of Pregnancies')

#     with col2:
#         Glucose = st.text_input('Glucose Level')

#     with col3:
#         BloodPressure = st.text_input('Blood Pressure value')

#     with col1:
#         SkinThickness = st.text_input('Skin Thickness value')

#     with col2:
#         Insulin = st.text_input('Insulin Level')

#     with col3:
#         BMI = st.text_input('BMI value')

#     with col1:
#         DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

#     with col2:
#         Age = st.text_input('Age of the Person')


#     # code for Prediction
#     diab_diagnosis = ''

#     # creating a button for Prediction

#     if st.button('Diabetes Test Result'):

#         user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
#                       BMI, DiabetesPedigreeFunction, Age]

#         user_input = [float(x) for x in user_input]

#         diab_prediction = diabetes_model.predict([user_input])

#         if diab_prediction[0] == 1:
#             diab_diagnosis = 'The person is diabetic'
#         else:
#             diab_diagnosis = 'The person is not diabetic'

#     st.success(diab_diagnosis)

# # Heart Disease Prediction Page
# if selected == 'Heart Disease Prediction':

#     # page title
#     st.title('Heart Disease Prediction using ML')

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.text_input('Age')

#     with col2:
#         sex = st.text_input('Sex')

#     with col3:
#         cp = st.text_input('Chest Pain types')

#     with col1:
#         trestbps = st.text_input('Resting Blood Pressure')

#     with col2:
#         chol = st.text_input('Serum Cholestoral in mg/dl')

#     with col3:
#         fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

#     with col1:
#         restecg = st.text_input('Resting Electrocardiographic results')

#     with col2:
#         thalach = st.text_input('Maximum Heart Rate achieved')

#     with col3:
#         exang = st.text_input('Exercise Induced Angina')

#     with col1:
#         oldpeak = st.text_input('ST depression induced by exercise')

#     with col2:
#         slope = st.text_input('Slope of the peak exercise ST segment')

#     with col3:
#         ca = st.text_input('Major vessels colored by flourosopy')

#     with col1:
#         thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

#     # code for Prediction
#     heart_diagnosis = ''

#     # creating a button for Prediction

#     if st.button('Heart Disease Test Result'):

#         user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

#         user_input = [float(x) for x in user_input]

#         heart_prediction = heart_disease_model.predict([user_input])

#         if heart_prediction[0] == 1:
#             heart_diagnosis = 'The person is having heart disease'
#         else:
#             heart_diagnosis = 'The person does not have any heart disease'

#     st.success(heart_diagnosis)

# # Parkinson's Prediction Page
# if selected == "Parkinsons Prediction":

#     # page title
#     st.title("Parkinson's Disease Prediction using ML")

#     col1, col2, col3, col4, col5 = st.columns(5)

#     with col1:
#         fo = st.text_input('MDVP:Fo(Hz)')

#     with col2:
#         fhi = st.text_input('MDVP:Fhi(Hz)')

#     with col3:
#         flo = st.text_input('MDVP:Flo(Hz)')

#     with col4:
#         Jitter_percent = st.text_input('MDVP:Jitter(%)')

#     with col5:
#         Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

#     with col1:
#         RAP = st.text_input('MDVP:RAP')

#     with col2:
#         PPQ = st.text_input('MDVP:PPQ')

#     with col3:
#         DDP = st.text_input('Jitter:DDP')

#     with col4:
#         Shimmer = st.text_input('MDVP:Shimmer')

#     with col5:
#         Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

#     with col1:
#         APQ3 = st.text_input('Shimmer:APQ3')

#     with col2:
#         APQ5 = st.text_input('Shimmer:APQ5')

#     with col3:
#         APQ = st.text_input('MDVP:APQ')

#     with col4:
#         DDA = st.text_input('Shimmer:DDA')

#     with col5:
#         NHR = st.text_input('NHR')

#     with col1:
#         HNR = st.text_input('HNR')

#     with col2:
#         RPDE = st.text_input('RPDE')

#     with col3:
#         DFA = st.text_input('DFA')

#     with col4:
#         spread1 = st.text_input('spread1')

#     with col5:
#         spread2 = st.text_input('spread2')

#     with col1:
#         D2 = st.text_input('D2')

#     with col2:
#         PPE = st.text_input('PPE')

#     # code for Prediction
#     parkinsons_diagnosis = ''

#     # creating a button for Prediction    
#     if st.button("Parkinson's Test Result"):

#         user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
#                       RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
#                       APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

#         user_input = [float(x) for x in user_input]

#         parkinsons_prediction = parkinsons_model.predict([user_input])

#         if parkinsons_prediction[0] == 1:
#             parkinsons_diagnosis = "The person has Parkinson's disease"
#         else:
#             parkinsons_diagnosis = "The person does not have Parkinson's disease"

#     st.success(parkinsons_diagnosis)


import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import shap
from io import BytesIO
import matplotlib.pyplot as plt

# Dummy models for demonstration (replace with your actual trained models)
diabetes_model = RandomForestClassifier()
heart_model = RandomForestClassifier()
parkinsons_model = RandomForestClassifier()

# Dummy data to fit the models (replace with your datasets)
diabetes_model.fit(np.random.rand(100, 6), np.random.randint(2, size=100))
heart_model.fit(np.random.rand(100, 13), np.random.randint(2, size=100))
parkinsons_model.fit(np.random.rand(100, 22), np.random.randint(2, size=100))

# Set page configuration
st.set_page_config(page_title="Health Prediction System", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
selected = st.sidebar.radio("Select a Page:", ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"])

# Dark mode toggle
dark_mode = st.sidebar.checkbox("Enable Dark Mode")

# Set theme based on dark mode
if dark_mode:
    st.markdown("""
    <style>
    body {background-color: #2e3b4e; color: white;}
    </style>
    """, unsafe_allow_html=True)

# Home Page
if selected == "Home":
    st.title("Welcome to the Health Prediction System")
    st.write("This application predicts Diabetes, Heart Disease, and Parkinson's Disease using machine learning models.")
    st.image("https://via.placeholder.com/800x300", caption="Health Monitoring Made Easy")
    st.write("**Navigate to a specific disease prediction page from the sidebar to begin.**")

# Helper function to validate inputs
def validate_input(value, feature, min_val, max_val):
    try:
        val = float(value)
        if min_val <= val <= max_val:
            return val, None
        else:
            return None, f"{feature} should be between {min_val} and {max_val}."
    except ValueError:
        return None, f"{feature} should be a numeric value."

# Helper function for confidence scores and SHAP values
import numpy as np
import shap
import matplotlib.pyplot as plt

def show_model_explanation(model, user_input):
    st.subheader("Model Confidence")
    prediction_proba = model.predict_proba([user_input])  # This is fine
    confidence_score = np.max(prediction_proba) * 100
    st.write(f"The model is **{confidence_score:.2f}%** confident about this prediction.")

    # Convert user_input to a 2D array if it's not already
    user_input_array = np.array([user_input])

    # Check if any NaN values are present and handle them
    if np.any(np.isnan(user_input_array)):
        st.error("Input contains NaN values. Please provide valid input.")
        return
    
    st.subheader("Feature Contribution")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input_array)  # Ensure user_input is 2D array

    # Generate and display the SHAP plot
    shap.initjs()  # Initialize SHAP JavaScript visualization
    st.pyplot(shap.summary_plot(shap_values, user_input_array))  # SHAP summary plot
    plt.clf()  # Clear the plot to prevent reuse in the next section


# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, help="Number of times pregnant.")
    glucose = st.number_input("Glucose", min_value=0, help="Plasma glucose concentration.")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, help="Diastolic blood pressure (mm Hg).")
    bmi = st.number_input("BMI", min_value=0.0, help="Body Mass Index.")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, help="Likelihood of diabetes based on family history.")
    age = st.number_input("Age", min_value=0, step=1, help="Age in years.")

    # Prediction
    if st.button("Predict Diabetes"):
        user_input = [pregnancies, glucose, blood_pressure, bmi, diabetes_pedigree, age]
        diabetes_prediction = diabetes_model.predict([user_input])[0]
        diagnosis = "The person has diabetes" if diabetes_prediction == 1 else "The person does not have diabetes"
        st.success(diagnosis)
        show_model_explanation(diabetes_model, user_input)

# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    
    age = st.number_input("Age", min_value=0, step=1, help="Age in years.")
    sex = st.selectbox("Sex", options=["Male", "Female"], help="Select the biological sex.")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="Type of chest pain.")
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, help="Resting blood pressure (mm Hg).")
    chol = st.number_input("Cholesterol", min_value=0, help="Serum cholesterol in mg/dL.")
    fbs = st.selectbox("Fasting Blood Sugar", options=[0, 1], help="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).")
    restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], help="Resting electrocardiographic results.")
    thalach = st.number_input("Thalach", min_value=0, help="Maximum heart rate achieved.")
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], help="Exercise induced angina (1 = yes, 0 = no).")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, help="Depression induced by exercise relative to rest.")
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], help="Slope of the peak exercise ST segment.")
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3], help="Number of major vessels colored by fluoroscopy.")
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], help="Thalassemia.")

    # Prediction
    if st.button("Predict Heart Disease"):
        user_input = [age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = heart_model.predict([user_input])[0]
        diagnosis = "The person has heart disease" if heart_prediction == 1 else "The person does not have heart disease"
        st.success(diagnosis)
        show_model_explanation(heart_model, user_input)

# Parkinson's Disease Prediction Page
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")
    
    jitter = st.number_input("Jitter (%)", min_value=0.0, help="Variation in voice frequency.")
    shimmer = st.number_input("Shimmer", min_value=0.0, help="Variation in voice amplitude.")
    nhr = st.number_input("Noise-to-Harmonics Ratio", min_value=0.0, help="Measure of vocal noise.")
    hnr = st.number_input("Harmonics-to-Noise Ratio", min_value=0.0, help="Measure of voice clarity.")
    rpde = st.number_input("RPDE", min_value=0.0, help="Recurrence plot-based feature for non-linear dynamics.")
    dfa = st.number_input("DFA", min_value=0.0, help="Detrended Fluctuation Analysis of the signal.")
    spread1 = st.number_input("Spread1", min_value=0.0, help="Spread of frequency components.")
    spread2 = st.number_input("Spread2", min_value=0.0, help="Spread of frequency components.")
    d2 = st.number_input("D2", min_value=0.0, help="Correlation dimension.")
    ppe = st.number_input("PPE", min_value=0.0, help="Pitch Period Entropy.")

    # Prediction
    if st.button("Predict Parkinson's"):
        user_input = [jitter, shimmer, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        parkinsons_prediction = parkinsons_model.predict([user_input])[0]
        diagnosis = "The person has Parkinson's disease" if parkinsons_prediction == 1 else "The person does not have Parkinson's disease"
        st.success(diagnosis)
        show_model_explanation(parkinsons_model, user_input)

# Downloadable Results
st.sidebar.subheader("Save Results")
if st.sidebar.button("Download Results as CSV"):
    data = {
        "Diabetes Prediction": [diabetes_prediction],
        "Heart Prediction": [heart_prediction],
        "Parkinson's Prediction": [parkinsons_prediction]
    }
    results_df = pd.DataFrame(data)
    csv = results_df.to_csv(index=False)
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)
    st.sidebar.download_button("Download CSV", data=b, file_name="predictions.csv")



