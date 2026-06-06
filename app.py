import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scale.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction System")
st.header("Predict house prices using machine learning")

st.sidebar.title("About")
st.sidebar.write("""
This system predicts house prices using:

- Linear Regression model
- Scikit-learn
- Streamlit
""")
st.sidebar.write("Model R2 Score(accuracy) : 70%")

st.subheader("Enter the House details below:")

area = st.number_input("Area", min_value=100, step=100)
bedrooms = st.number_input("Bedrooms",min_value=1, max_value=6, step=1)
bathrooms = st.number_input("Bathrooms",min_value=1, max_value=3, step=1)
floors = st.number_input("Floors",max_value=4, step=1)
parking = st.number_input("Parking",max_value=3, step=1)

mainroad = st.selectbox("Mainroad",["Yes","No"])
guestroom = st.selectbox("Guest room",["Yes","No"])
basement = st.selectbox("Basement",["Yes","No"])
hotwaterheating = st.selectbox("Hot water heating",["Yes","No"])
airconditioning = st.selectbox("Air conditioning",["Yes","No"])
prefarea = st.selectbox("Preferred Area",["Yes","No"])
furnishingstatus = st.selectbox("Furnishing status",["furnished","semi-furnished","unfurnished"])

mainroad=1 if mainroad=="Yes" else 0
guestroom=1 if guestroom=="Yes" else 0
basement=1 if basement=="Yes" else 0
hotwaterheating=1 if hotwaterheating=="Yes" else 0
airconditioning=1 if airconditioning=="Yes" else 0
prefarea=1 if prefarea=="Yes" else 0

if st.button("🏠 Predict Price"):
    features = pd.DataFrame({
        "area":[np.log(area)],
        "mainroad_yes":[mainroad],
        "guestroom_yes":[guestroom],
        "basement_yes":[basement],
        "hotwaterheating_yes":[hotwaterheating],
        "airconditioning_yes":[airconditioning],
        "prefarea_yes":[prefarea],
        "furnishingstatus_semi-furnished":[1 if furnishingstatus=="semi-furnished" else 0],
        "furnishingstatus_unfurnished":[1 if furnishingstatus=="unfurnished" else 0],
        "bedrooms_2":[1 if bedrooms==2 else 0],
        "bedrooms_3":[1 if bedrooms==3 else 0],
        "bedrooms_4":[1 if bedrooms==4 else 0],
        "bedrooms_5":[1 if bedrooms==5 else 0],
        "bedrooms_6":[1 if bedrooms==6 else 0],
        "bathrooms_2":[1 if bathrooms==2 else 0],
        "bathrooms_3":[1 if bathrooms==3 else 0],
        "stories_2":[1 if floors==2 else 0],
        "stories_3":[1 if floors==3 else 0],
        "stories_4":[1 if floors==4 else 0],
        "parking_1":[1 if parking==1 else 0],
        "parking_2":[1 if parking==2 else 0],
        "parking_3":[1 if parking==3 else 0]
    })

    features = features.reindex(columns=columns, fill_value=0)

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    output = np.exp(prediction[0]) 

    st.success(f"Predicted House price : ₹{output:,.2f}")

    st.balloons()

st.markdown("---")
st.write("Built using Streamlit for UI and Machine Learning for Model.")
st.markdown("---")
st.write("""
#### Author: Sreejith T
📧 Email: sreejith.py3@gmail.com     
🌐 GitHub: https://github.com/Sreejith-005
""")