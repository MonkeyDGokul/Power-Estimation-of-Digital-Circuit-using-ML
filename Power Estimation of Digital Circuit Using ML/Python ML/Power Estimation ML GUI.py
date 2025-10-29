import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title of the app
st.title("Power Consumption Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Data")
    st.write(df.head())
    
    if "Power_total(nW)" not in df.columns:
        st.error("Dataset must contain 'Power_total(nW)' column as target variable")
    else:
        # Remove the first column (categorical identifier)
        df = df.iloc[:, 1:]
        
        # Prepare data
        X = df.drop(columns=["Power_total(nW)"])
        y = df["Power_total(nW)"]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the MLP Regressor
        if 'model' not in st.session_state:
            st.session_state.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

        if st.button("Train Model"):
            st.session_state.model.fit(X_train_scaled, y_train)
            y_pred = st.session_state.model.predict(X_test_scaled)
            
            # Performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.write("### Model Performance")
            st.write(f"*MAE:* {mae:.2f}")
            st.write(f"*MSE:* {mse:.2f}")
            st.write(f"*R² Score:* {r2:.4f}")
            
            # Plot results
            fig, ax = plt.subplots()
            ax.plot(y_test.values, label='Actual', marker='o', linestyle='-', color='blue')
            ax.plot(y_pred, label='Predicted', marker='x', linestyle='--', color='red')
            ax.set_xlabel("Test Sample Index")
            ax.set_ylabel("Power_total(nW)")
            ax.set_title("Actual vs Predicted Values")
            ax.legend()
            st.pyplot(fig)

        # Prediction section
        st.write("### Make a Prediction")
        user_input = []
        for col in X.columns:
            user_input.append(st.number_input(f"Enter {col}", value=float(df[col].mean())))

        if st.button("Predict Power Consumption"):
            if 'model' in st.session_state:
                user_input = np.array(user_input).reshape(1, -1)
                user_input_scaled = scaler.transform(user_input)
                prediction = st.session_state.model.predict(user_input_scaled)
                st.success(f"Predicted Power Consumption: {prediction[0]:.2f} nW")
            else:
                st.error("Model is not trained yet. Please train the model first.")
