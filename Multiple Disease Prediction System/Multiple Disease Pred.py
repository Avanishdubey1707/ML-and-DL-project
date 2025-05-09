import pickle 
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models 
heart_disease_model = pickle.load(open("saved_models/heart_disease_model.sav", "rb"))
diabetes_model = pickle.load(open("saved_models/diabetes_model.sav", "rb"))
parkinson_model = pickle.load(open("saved_models/parkinsons_model.sav", "rb"))
breastCancer_model = pickle.load(open("saved_models/breast_cancer_model.sav", "rb"))
# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Heart disease prediction", "Diabetes prediction", "Parkinsons prediction", "Breast Cancer prediction"],
        icons=["heart", "activity", "person", "clipboard-heart"],
        default_index=0
    )

# Heart disease prediction page 
if selected == "Heart disease prediction":
    st.title("Heart Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain Types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('Major Vessels Colored by Fluoroscopy')
    with col1:
        thal = st.text_input('Thal: 0=Normal; 1=Fixed Defect; 2=Reversible Defect')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if heart_prediction[0] == 1:
            heart_diagnosis = "The person is having heart disease"
        else:
            heart_diagnosis = "The person does not have heart disease"

    st.success(heart_diagnosis)

# Diabetes prediction page 
if selected == "Diabetes prediction":
    st.title("Diabetes Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is not Diabetic'

    st.success(diab_diagnosis)

# Parkinson's prediction page 
if selected == "Parkinsons prediction":
    st.title("Parkinson's Prediction using ML")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_DB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('Spread1')
    with col5:
        spread2 = st.text_input('Spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinson_model.predict([[fo, fhi, flo, jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_DB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# Breast Cancer prediction page 
if selected == "Breast Cancer prediction":
    def breast_cancer_page():
        st.title("Breast Cancer Prediction using ML")
        cols = st.columns(3)
        inputs = []

        feature_names = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
            "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

        for idx, feature in enumerate(feature_names):
            col = cols[idx % 3]
            value = col.number_input(f"{feature}", key=feature)
            inputs.append(value)

        if st.button("Breast Cancer Test Result"):
            try:
                prediction = breastCancer_model.predict([inputs])
                if prediction[0] == 1:
                    st.success("The person **has breast cancer**.")
                else:
                    st.success("The person **does not have breast cancer**.")
            except Exception as e:
                st.error(f"Error: {e}")

    breast_cancer_page()

