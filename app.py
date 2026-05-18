import streamlit as st
import joblib
import numpy as np

model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")

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
st.sidebar.write("Model R2 Score(accuracy) : 64%")

st.subheader("Enter the House details below:")

area = st.number_input("Area", min_value=100, step=100)
bedrooms = st.number_input("Bedrooms", step=1)
bathrooms = st.number_input("Bathrooms",step=1)
floors = st.number_input("Floors",step=1)
parking = st.number_input("Parking",step=1)

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

furnishing_map = {
    "furnished": 0,
    "semi-furnished": 1,
    "unfurnished": 2
}

furnishingstatus = furnishing_map[furnishingstatus]

if st.button("🏠 Predict Price"):
    features = np.array([[area, 
                          bedrooms, 
                          bathrooms, 
                          floors, 
                          mainroad, 
                          guestroom, 
                          basement,
                          hotwaterheating,
                          airconditioning,
                          parking,
                          prefarea,
                          furnishingstatus
                         ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    st.success(f"Predicted House price : ₹{prediction[0]:,.2f}")

    st.balloons()

st.markdown("---")
st.write("Built using Streamlit for UI and Machine Learning for Model.")
st.markdown("---")
st.write("""
#### Author: Sreejith T
📧 Email: sreejith.py3@gmail.com     
🌐 GitHub: https://github.com/Sreejith-005
""")
