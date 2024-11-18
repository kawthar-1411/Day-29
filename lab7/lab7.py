
import streamlit as st

st.title("player Price Prediction")



appearance = st.number_input("appearance", min_value=0, max_value=2024, value=2020, step=1)
minutes_played = st.number_input("minutes_played", min_value=0.0, value=5000)
award = st.number_input("award", min_value=0.0, value=50000)
highest_value = st.number_input("highest_value", min_value=0.0, value=5000)



if st.button("Predict"):
    input_data = InputFeatures(
        Appearance=appearance,
        Minutes_played=minutes_played,
        Award=award,
        Highest_value=highest_value
        
    )

    processed_data = preprocessing(input_data)
    prediction = model.predict(processed_data)

    st.success(f"Predicted Cluster: {prediction[0]:,.2f}")