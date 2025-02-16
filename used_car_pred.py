import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Title
st.title('Used Car Price Predictor')
st.text('This web application predicts the price of a used car based on its features.')

# Sidebar Input Fields
st.sidebar.header("Please input car details")

def create_user_input():
    # Categorical Inputs (Extracted from Screenshot)
    make = st.sidebar.selectbox('Make', [
        'Hyundai', 'Toyota', 'INFINITI', 'Chevrolet', 'Ford', 'Honda', 'Fiat', 'Nissan', 'Lexus',
        'Mercedes', 'GMC', 'HAVAL', 'Kia', 'Land Rover', 'Mitsubishi', 'Mazda', 'Changan', 'BMW',
        'Other', 'Volkswagen', 'Chrysler', 'Jeep', 'Audi', 'Renault', 'MINI', 'Dodge', 'Peugeot',
        'Geely', 'Hummer', 'Isuzu', 'Genesis', 'Maserati', 'Cadillac', 'Bentley', 'GAC', 'Jaguar',
        'Porsche', 'Škoda', 'Suzuki', 'MG', 'Chery', 'Daihatsu', 'Lincoln', 'Mercury', 'Great Wall',
        'ASEW', 'BYD', 'Tata', 'Aston Martin', 'Exeed', 'Foton', 'Zhengzhou', 'Classic', 'SsangYong',
        'Victory Auto', 'Lifan'
    ])
    
    type_car = st.sidebar.selectbox('Type', [
        'Accent', 'Camry', 'QX', 'Malibu', 'Suburban', 'Innova', 'Marquis', 'Accord', '500', 'Explorer',
        'Datsun', 'Sentra', 'Maxima', 'Hilux', 'Civic', 'FJ', 'Patrol', 'LX', 'SL', 'Sierra', 'Impala',
        'Expedition', 'H6', 'Avalon', 'Seltos', 'POS24', 'Land Cruiser', 'Range Rover', 'Pajero', 'Tahoe',
        'Sunny', 'Yaris', 'Sportage', 'Coupe', 'Rio', 'Corolla', 'Elantra', 'Caprice', 'Spark', 'Yukon',
        'Carnival', 'H1', 'CS95', 'Pathfinder', 'Taurus', 'CX9', 'Prado', 'The 3', 'Flex', 'Cressida',
        'Senta fe', 'Seven', 'F150', 'Touareg', 'Vego', 'Pilot', 'C300', 'Cores', '7', 'Grand Cherokee',
        'Beetle', 'Victoria', 'Optima', 'A3', 'ES', 'Symbol', 'Hiace', 'Attrage', 'Azera', 'Copper', 'RX',
        'Fluence', 'CX7', 'X', 'Charger', 'The 5', 'Land Cruiser Pickup', '301', 'Durango', 'Tucson',
        'Navara', 'Kona', 'EC7', 'GLE', 'Armada', 'Mustang', 'Edge', 'Sonata', 'H3', 'Cherokee', 'Passat',
        'D-MAX', 'Cerato', 'Echo', '2', 'Camaro', 'Altima', 'Odyssey', 'i40', 'Van', 'LS', 'Cadenza',
        'Colorado', 'Duster', 'H2', 'Eado', 'CXS', 'Countryman', 'Challenger', 'Aurion', 'Platinum',
        'Fusion', 'GS', 'ML', '300', 'Crown', 'S300', 'Carens', 'Cruze', 'Nitro', 'Dokker', 'C',
        'Quattroporte', 'G', 'Megane', 'Rush', 'A', 'Lancer', 'GL', 'Echo Sport', 'CT-S', 'Q5', 'Escalade',
        'Compass', 'Kaptiva', 'Silverado', 'Blazer', 'Avanza', 'Flying Spur', 'Bus Urvan', 'Prius', 'Creta',
        'G80', 'Dyna', 'VTC', 'CS35', 'Traverse', 'CLA', 'Rav4', 'CX3', 'Coupe S', 'GS3', 'Opirus', '3',
        'Wrangler', 'IS', 'Envoy', 'CLS', 'FX', 'Bronco', 'SLK', 'Ciocca', 'SEL', 'KICKS', 'Sorento', 'GX',
        'XJ', 'The 4', 'X-Trail', 'Royal', 'Cayenne', 'Superb', 'UX', 'F-Pace', 'Jimny', 'Safrane',
        'Grand Vitara', 'Abeka', 'Q', 'RX5', 'Tiggo', 'Logan', 'Z', 'Gran Max', 'H100', 'Previa', 'Veloster',
        'MKZ', 'BT-50', '360', 'C575', 'MKX', 'Koleos', 'ZS', 'APV', 'A8', 'Grand Marquis', 'Power', 'Acadia',
        'Cayenne S', 'Bora', 'Safari', 'A6', 'Montero', 'Macan', 'L200', 'GLC', 'Panamera', 'ATS', 'Delta',
        'RC', 'Emgrand', 'Genesis', '5008', 'Discovery', 'Optra', 'The M', 'Ranger', 'Capture', 'S8', 'CT4',
        'Prestige', 'Coaster', 'Mohave', 'Defender', 'Focus', 'A7', 'XT5', 'Bus County', 'X40', 'Z350', 'Q7',
        'New Yorker', 'City', 'Boxer', 'Cayenne Turbo S', 'Tuscani', 'MKS', 'The 6', 'Terrain', 'HRV', 'Picanto',
        'Aveo', 'Ram', 'Juke', 'Lumina', 'H9', 'F3', '3008', 'Azkarr', 'TC', 'Town Car', 'X7', 'GC7', 'XF', 'CS85',
        'Corolla Cross', 'A4', '5', 'Doblo', 'Soul', 'Carenz', 'Cayenne Turbo', 'Sylvian Bus', 'V7', 'Wingle',
        'Dmax', 'RX8', 'Nexon', 'Navigator', 'CRV', 'DB9', 'Daily', 'Montero2', 'Mini Van', 'Nativa', '4Runner',
        'Ertiga', 'ASX', 'NX', 'Milan', 'A5', 'DTS', 'Liberty', 'Pick up', 'Stinger', 'Prestige Plus',
        'ACTIS V80', 'SRT', 'CS35 Plus', 'Fleetwood', 'Golf', 'CT5', 'Viano', 'Avalanche', 'EC8', 'S5', 'SRX',
        'Sedona', 'CC', 'Suvana', 'B50', 'L300', 'Tiguan', 'Dzire', 'Jetta', 'C200', 'Cayman', 'K5', 'HS',
        'Centennial', 'Thunderbird', 'Avante', 'M', 'Murano', 'Z370', 'Cadillac', 'G70', 'Koranado', 'Pegas',
        'Vitara', 'Van R', 'LF X60', 'Dakota', 'X-Terra', 'Savana', 'F Type', 'CL', 'Coolray', 'Teros'
    ])
    
    options = st.sidebar.radio('Options', ['Standard', 'Semi Full', 'Full'])
    region = st.sidebar.selectbox('Region', [
        'Najran', 'Riyadh', 'Jeddah', 'Abha', 'Al-Ahsa', 'Al-Medina', 'Dammam', 'Hail', 'Tabouk',
        'Qassim', 'Khobar', 'Aseer', 'Makkah', 'Taef', 'Al-Jouf', 'Hafar Al-Batin', 'Jazan', 
        'Al-Baha', 'Jubail', 'Arar', 'Yanbu', 'Al-Namas', 'Wadi Dawasir', 'Sakaka', 'Qurayyat', 
        'Besha', 'Sabya'
    ])
    
    gear_type = st.sidebar.radio('Gear Type', ['Automatic', 'Manual'])
    origin = st.sidebar.radio('Origin', ['Saudi', 'Gulf Arabic', 'Other'])

    # Numerical Inputs
    year = st.sidebar.slider('Year', min_value=2000, max_value=2024, value=2015)
    mileage = st.sidebar.number_input('Mileage (KM)', min_value=0, max_value=500000, value=100000)
    engine_size = st.sidebar.slider('Engine Size', min_value=1.0, max_value=9.0, value=2.0, step=0.1)

    # Feature Engineering
    metro_cities = ['Riyadh', 'Jeddah', 'Dammam']
    luxury_suvs = ['Range Rover', 'Land Cruiser', 'RX5']
    luxury_brands = ['Mercedes', 'Lexus', 'Porsche', 'Land Rover', 'Bentley']
    reference_date = 2023

    user_data = {
        'Make': make,
        'Type': type_car,
        'Options': options,
        'Region': region,
        'Gear_Type': gear_type,
        'Origin': origin,
        'Year': year,
        'Mileage': mileage,
        'Engine_Size': engine_size,
        'Metro_Region': 1 if region in metro_cities else 0,
        'Age': reference_date - year,
        'Luxury_SUV': 1 if type_car in luxury_suvs else 0,
        'Luxury_Brand': 1 if make in luxury_brands else 0
    }
    
    return pd.DataFrame([user_data])

# Get user input
data_car = create_user_input()

# Display the input data
st.subheader("Selected Car Features")
st.write(data_car)

# Load model
with open('Model Final.sav', 'rb') as f:
    model_loaded = pickle.load(f)

# Predict car price
predicted_price = model_loaded.predict(data_car)

# Show Prediction
st.subheader('Predicted Car Price')
st.write(f"Estimated Price: **{predicted_price[0]:,.2f} SAR**")