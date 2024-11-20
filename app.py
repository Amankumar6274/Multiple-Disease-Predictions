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
import plotly.express as px
import shap
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have trained models saved as pickle files or in memory
# For example:
# model_diabetes = load_model('diabetes_model.pkl')
# model_heart = load_model('heart_attack_model.pkl')
# model_parkinsons = load_model('parkinsons_model.pkl')

# Placeholder model functions (replace these with actual model loading)
def load_model(model_name):
    return RandomForestClassifier()  # Replace with actual model loading logic

# Load models
model_diabetes = load_model('diabetes_model.pkl')
model_heart = load_model('heart_attack_model.pkl')
model_parkinsons = load_model('parkinsons_model.pkl')

# Helper function for tooltips
def feature_tooltip(feature):
    feature_info = {
        'Pregnancies': 'Number of pregnancies the patient has had.',
        'Glucose': 'Plasma glucose concentration after 2 hours in an oral glucose tolerance test.',
        'BloodPressure': 'Blood pressure measured in mm Hg.',
        'SkinThickness': 'Skin thickness (triceps skinfold thickness) in mm.',
        'Insulin': 'Insulin level in the blood.',
        'BMI': 'Body Mass Index (BMI) calculated from weight and height.',
        'DiabetesPedigreeFunction': 'A function that scores the likelihood of diabetes based on family history.',
        'Age': 'Age of the individual.',
        'Outcome': 'Diabetes outcome (1: diabetic, 0: non-diabetic).'
    }
    return feature_info.get(feature, 'No information available')

# Diabetes Prediction Form
def diabetes_prediction():
    st.title("Diabetes Prediction")
    
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, help=feature_tooltip('Pregnancies'))
    Glucose = st.number_input('Glucose Level', help=feature_tooltip('Glucose'))
    BloodPressure = st.number_input('Blood Pressure', help=feature_tooltip('BloodPressure'))
    SkinThickness = st.number_input('Skin Thickness', help=feature_tooltip('SkinThickness'))
    Insulin = st.number_input('Insulin Level', help=feature_tooltip('Insulin'))
    BMI = st.number_input('BMI', help=feature_tooltip('BMI'))
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', help=feature_tooltip('DiabetesPedigreeFunction'))
    Age = st.number_input('Age', help=feature_tooltip('Age'))
    
    if st.button("Predict"):
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = model_diabetes.predict(features)
        st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")

# Heart Attack Prediction Form
def heart_attack_prediction():
    st.title("Heart Attack Prediction")
    
    age = st.number_input('Age', help=feature_tooltip('Age'))
    sex = st.selectbox('Sex', [0, 1], help=feature_tooltip('Sex'))  # 0 for female, 1 for male
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], help=feature_tooltip('cp'))  # Various types of chest pain
    trestbps = st.number_input('Resting Blood Pressure', help=feature_tooltip('trestbps'))
    chol = st.number_input('Cholesterol Level', help=feature_tooltip('chol'))
    fbs = st.selectbox('Fasting Blood Sugar', [0, 1], help=feature_tooltip('fbs'))  # 0 or 1
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], help=feature_tooltip('restecg'))
    thalach = st.number_input('Maximum Heart Rate', help=feature_tooltip('thalach'))
    exang = st.selectbox('Exercise Induced Angina', [0, 1], help=feature_tooltip('exang'))
    oldpeak = st.number_input('Depression Induced by Exercise', help=feature_tooltip('oldpeak'))
    slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2], help=feature_tooltip('slope'))
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3], help=feature_tooltip('ca'))
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3], help=feature_tooltip('thal'))
    
    if st.button("Predict"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model_heart.predict(features)
        st.write(f"Prediction: {'Heart Attack Risk' if prediction[0] == 1 else 'No Heart Attack Risk'}")

# Parkinson's Disease Prediction Form
def parkinsons_prediction():
    st.title("Parkinson's Disease Prediction")
    
    MDVP_Fo_Hz = st.number_input('MDVP:Fo(Hz)', help=feature_tooltip('MDVP:Fo(Hz)'))
    MDVP_Fhi_Hz = st.number_input('MDVP:Fhi(Hz)', help=feature_tooltip('MDVP:Fhi(Hz)'))
    MDVP_Flo_Hz = st.number_input('MDVP:Flo(Hz)', help=feature_tooltip('MDVP:Flo(Hz)'))
    MDVP_Jitter_percent = st.number_input('MDVP:Jitter(%)', help=feature_tooltip('MDVP:Jitter(%)'))
    MDVP_Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', help=feature_tooltip('MDVP:Jitter(Abs)'))
    MDVP_RAP = st.number_input('MDVP:RAP', help=feature_tooltip('MDVP:RAP'))
    MDVP_PPQ = st.number_input('MDVP:PPQ', help=feature_tooltip('MDVP:PPQ'))
    Jitter_DDP = st.number_input('Jitter:DDP', help=feature_tooltip('Jitter:DDP'))
    MDVP_Shim = st.number_input('MDVP:Shimmer', help=feature_tooltip('MDVP:Shimmer'))
    # Add all features for Parkinsons

    if st.button("Predict"):
        features = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs,
                              MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shim]])  # Include all the required features
        prediction = model_parkinsons.predict(features)
        st.write(f"Prediction: {'Parkinson\'s Disease' if prediction[0] == 1 else 'No Parkinson\'s Disease'}")


# Main app layout
def main():
    st.sidebar.title("Disease Prediction System")
    option = st.sidebar.selectbox("Select Disease", ["Diabetes", "Heart Attack", "Parkinson's Disease"])
    
    if option == "Diabetes":
        diabetes_prediction()
    elif option == "Heart Attack":
        heart_attack_prediction()
    else:
        parkinsons_prediction()

if __name__ == "__main__":
    main()
