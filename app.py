import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Fare & Mileage Prediction App 🚕")

# ✅ Service Type Selection
service_type = st.selectbox(
    "Select Service Type",
    ["🚕 Cab Service (Personal ride)", "🚙 Car Pooling (Shared ride)"]
)

trip_distance_km = st.number_input("Trip Distance (km)", min_value=0.1, step=0.1, value=5.0)
fuel_price_per_litre = st.number_input("Fuel Price (Rs/litre)", min_value=50.0, step=1.0, value=96.0)
vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
car_type = st.selectbox("Car Type", ["Hatch", "Sedan", "SUV"])

if service_type == "🚕 Cab Service (Personal ride)":
    ride_type = st.selectbox("Ride Type", ["Exclusive", "Shared"])
else:
    ride_type = "Shared"
    st.markdown("**Car Pooling always uses Shared ride.**")

# ✅ Fixed AC always ON internally
ac_factor = 1.2
ride_type_factor = 1.0 if ride_type == "Shared" else 1.2
car_type_factor = {"Hatch": 1.0, "Sedan": 1.1, "SUV": 1.2}[car_type]
time_of_day_factor = 1.0  # Time of day fixed internally

# ✅ Simulate Training Data with Mileage
np.random.seed(42)
n = 500
train_df = pd.DataFrame({
    'vehicle_age': np.random.randint(0, 20, n),
    'ac_factor': np.full(n, 1.2),
    'ride_type_factor': np.random.choice([1.0, 1.2], n),
    'car_type_factor': np.random.choice([1.0, 1.1, 1.2], n),
    'time_of_day_factor': np.full(n, 1.0)
})

train_df['mileage'] = (
    21
    - 0.5 * train_df['vehicle_age']
    - 2.0 * (train_df['ac_factor'] - 1.0)
    - 1.5 * (train_df['car_type_factor'] - 1.0)
    - 1.0 * (train_df['ride_type_factor'] - 1.0)
    - 0.5 * (train_df['time_of_day_factor'] - 1.0)
    + np.random.normal(0, 0.5, n)
).clip(lower=8, upper=25)

model = LinearRegression()
model.fit(train_df[['vehicle_age', 'ac_factor', 'ride_type_factor', 'car_type_factor', 'time_of_day_factor']], train_df['mileage'])

user_X = pd.DataFrame([{
    'vehicle_age': vehicle_age,
    'ac_factor': ac_factor,
    'ride_type_factor': ride_type_factor,
    'car_type_factor': car_type_factor,
    'time_of_day_factor': time_of_day_factor
}])

if st.button("Predict Fare & Mileage"):
    predicted_mileage = model.predict(user_X)[0]
    predicted_mileage = np.clip(predicted_mileage, 8.0, 22.0)

    fuel_used = trip_distance_km / predicted_mileage
    base_fare = fuel_used * fuel_price_per_litre
    final_fare = max(40, base_fare)

    st.success(f"Predicted Mileage: {predicted_mileage:.2f} km/l")
    st.success(f"Estimated Fare: Rs {final_fare:.2f}")

    receipt = f"""
Fare & Mileage Receipt - Tiaro

Service Type       : {service_type}
Trip Distance      : {trip_distance_km} km
Fuel Price         : Rs {fuel_price_per_litre}/litre
Vehicle Age        : {vehicle_age} years
AC                 : Always ON
Ride Type          : {ride_type}
Car Type           : {car_type}

Predicted Mileage  : {predicted_mileage:.2f} km/l
Raw Fare           : Rs {base_fare:.2f}
Final Fare         : Rs {final_fare:.2f}

Thank you for riding with Tiaro!
"""
    st.download_button("🧾 Download Fare Receipt", receipt, file_name="fare_receipt.txt")

