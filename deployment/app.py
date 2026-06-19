
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Visit With Us - Wellness Package Prediction")

# Download model from HF Model Hub

model_path = hf_hub_download(
    repo_id="krisna-Labs/visit-with-us-mlops",
    filename="tourism_model.pkl"
)

model = joblib.load(model_path)

st.header("Customer Information")

Age = st.number_input("Age", 18, 100, 30)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    1,
    1000,
    30
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Person Visiting",
    1,
    10,
    2
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    0,
    10,
    2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

NumberOfTrips = st.number_input(
    "Number Of Trips",
    0,
    20,
    2
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Children Visiting",
    0,
    5,
    0
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    1000,
    500000,
    50000
)

TypeofContact = st.selectbox(
    "Type Of Contact",
    ["Self Enquiry", "Company Invited"]
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Free Lancer",
        "Small Business",
        "Large Business"
    ]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

if st.button("Predict"):

    data = pd.DataFrame({
        "Age":[Age],
        "CityTier":[CityTier],
        "DurationOfPitch":[DurationOfPitch],
        "NumberOfPersonVisiting":[NumberOfPersonVisiting],
        "NumberOfFollowups":[NumberOfFollowups],
        "PreferredPropertyStar":[PreferredPropertyStar],
        "NumberOfTrips":[NumberOfTrips],
        "Passport":[Passport],
        "PitchSatisfactionScore":[PitchSatisfactionScore],
        "OwnCar":[OwnCar],
        "NumberOfChildrenVisiting":[NumberOfChildrenVisiting],
        "MonthlyIncome":[MonthlyIncome],
        "TypeofContact":[TypeofContact],
        "Occupation":[Occupation],
        "Gender":[Gender],
        "ProductPitched":[ProductPitched],
        "MaritalStatus":[MaritalStatus],
        "Designation":[Designation]
    })

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success(
            "Customer is likely to purchase the Wellness Package"
        )
    else:
        st.error(
            "Customer is unlikely to purchase the Wellness Package"
        )
