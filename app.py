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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from io import BytesIO
import plotly.express as px
from sklearn.metrics import accuracy_score
import joblib

# Load models and vectorizers
diabetes_model = joblib.load('diabetes_model.pkl')
heart_model = joblib.load('heart_model.pkl')
parkinson_model = joblib.load('parkinson_model.pkl')

# Load scaler
scaler = joblib.load('scaler.pkl')

# Set title for the app
st.title("Multiple Disease Prediction System")

# Page selection
page = st.sidebar.radio("Select a Disease", ("Diabetes", "Heart Disease", "Parkinson's Disease"))

if page == "Diabetes":
    st.header("Diabetes Prediction")

    # Input fields for Diabetes
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000, step=1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, step=0.1)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.1)
    age = st.number_input('Age', min_value=0, max_value=120, step=1)

    # Button for prediction
    if st.button('Predict Diabetes'):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        input_data = scaler.transform(input_data)
        prediction = diabetes_model.predict(input_data)

        st.write(f"Prediction: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")

if page == "Heart Disease":
    st.header("Heart Disease Prediction")

    # Input fields for Heart Disease
    age = st.number_input('Age', min_value=1, max_value=120, step=1)
    sex = st.selectbox('Sex', ('Male', 'Female'))
    cp = st.selectbox('Chest Pain Type', (0, 1, 2, 3))  # 0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, step=1)
    chol = st.number_input('Cholesterol', min_value=0, max_value=600, step=1)
    fbs = st.selectbox('Fasting Blood Sugar', (0, 1))  # 0: false, 1: true
    restecg = st.selectbox('Resting Electrocardiographic Results', (0, 1, 2))  # 0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=220, step=1)
    exang = st.selectbox('Exercise Induced Angina', (0, 1))  # 0: no, 1: yes
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', (0, 1, 2))  # 0: upsloping, 1: flat, 2: downsloping
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', (0, 1, 2, 3))
    thal = st.selectbox('Thalassemia', (1, 2, 3))  # 1: normal, 2: fixed defect, 3: reversable defect

    # Button for prediction
    if st.button('Predict Heart Disease'):
        input_data = np.array([[age, sex == 'Male', cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_data = scaler.transform(input_data)
        prediction = heart_model.predict(input_data)

        st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")

if page == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")

    # Input fields for Parkinson's Disease
    name = st.text_input('Name', 'Patient Name')  # not used in prediction, just for display
    MDVP_Fo_Hz = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=1000.0, step=0.1)
    MDVP_Fhi_Hz = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=1000.0, step=0.1)
    MDVP_Flo_Hz = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=1000.0, step=0.1)
    MDVP_Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=100.0, step=0.1)
    MDVP_Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, step=0.01)
    MDVP_RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=1.0, step=0.01)
    MDVP_PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=1.0, step=0.01)
    Jitter_DDP = st.number_input('Jitter:DDP', min_value=0.0, max_value=1.0, step=0.01)
    MDVP_Shim = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=100.0, step=0.1)

    # Button for prediction
    if st.button('Predict Parkinson’s Disease'):
        input_data = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shim]])
        input_data = scaler.transform(input_data)
        prediction = parkinson_model.predict(input_data)

        prediction_text = "Parkinson's Disease" if prediction[0] == 1 else "No Parkinson's Disease"
        st.write(f"Prediction: {prediction_text}")


