 import streamlit as st
 import pandas as pd
 import numpy as np
 import matplotlib . pyplot as plt
 from sklearn.model selection import train  test split
 from sklearn.linear model import LinearRegression , Lasso , Ridge
 from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
 from sklearn.svm import SVR
 from sklearn.neural network import MLPRegressor
 from xgboost import XGBRegressor
 from lightgbm import LGBMRegressor
 from catboost import CatBoostRegressor
 from sklearn.metrics import mean _absolute_error, mean_squared_error, r2_score
 import random

 # Title of the app
 st.title ("Power Consumption Prediction App")
 # File uploader
 uploaded_file = st . file uploader ("Upload your dataset (CSV file )" , type=["csv"])
 # if uploaded_file is not None:
 df = pd.read
 csv ( uploaded file )
 st.write ("Sample Data")
 st.write (df.head())

# Preprocessing : Remove ’circuit ’ and existing ’ID’ columns , then add new random IDs
 df . drop(columns=[ 'circuit' , 'ID' ] , inplace=True , errors= 'ignore')
 df . insert (0 , 'ID' , [random. randint(1000, 9999) for in range(len(df)) ])

 # Check required columns
 required_cols = ["Power total(nW)", "Power dynamic(Switching ) (nW)"]
 if not all(col in df.columns for col in required_cols):
 st.error ("Dataset must contain 'Power total(nW) ' and ' Power dynamic(Switching)(nW) ' columns as target 
variables")

 else :
 # Prepare data (exclude ID column from features)
 X = df.drop(columns=required_cols + ['ID'])
 y_total = df["Power total(nW)"]
 y_dynamic = df["Power dynamic(Switching)(nW)"]
 # Single train−test split for both targets
 X_train , X_test , y_train, y_test_total = train_test_split ( X, y_total , test size =0.2, random state=42 )
 # Use same split indices for dynamic target
 y_train_dynamic = y_dynamic.iloc [ X_train.index ]
 test dynamic = y dynamic.iloc[ X_test.index ]
 
# Model selection

model_name = st.selectbox(
 'Choose a model for training',
 options=[
  'Linear Regression', 'RandomForest', 'Gradient Boosting',
 'MLPRegressor', 'Support Vector Regressor' , 'XGBoost' ,
 'Lasso Regression' , 'Ridge Regression' , 'LightGBM' , 'CatBoost'
 ]
 )

 if st.button("Train Model") :
 # Initialize models for both targets
 models = {
 'Linear Regression' : (LinearRegression() ,LinearRegression()),
 'RandomForest' : (RandomForestRegressor( n_estimators=100, random_state=42),RandomForestRegressor( n_estimators=100, random_state=42)),
 'Gradient Boosting' : (GradientBoostingRegressor( n_estimators=100, random_state=42) , GradientBoostingRegressor( n_estimators=100, random_state=42)),
 'MLPRegressor' : (MLPRegressor( hidden_layer_sizes=(64, 32), max_iter = 1000,  random_state=42) , MLPRegressor( hidden_layer_sizes=(64, 32), max_iter =1000,random_state=42)),
 'Support Vector Regressor' : (SVR() , SVR()) ,    
 'XGBoost': (XGBRegressor(n_estimators=100, random_state=42) , XGBRegressor( n_estimators=100, random_state=42)) ,
 'Lasso Regression' : (Lasso() , Lasso()) ,
 'Ridge Regression' : (Ridge() , Ridge()) ,   
 'LightGBM' : (LGBMRegressor(n_estimators=100, random_state=42) , LGBMRegressor(n_estimators=100, random_state=42)) ,
 'CatBoost' : (CatBoostRegressor(verbose=0, random_state=42),CatBoostRegressor(verbose=0, random_state=42))
 }

model_total ,model_dynamic = models[model_name]

# Train models
 model_total.fit (X_train , y_train_total)
 model dynamic.fit (X_train , y_train_dynamic)

# Store trained models
 st.session_state [ 'model_total' ] = model_total
 st.session_state [ 'model dynamic' ] = model dynamic

# Get predictions
 y_pred_total = model_total.predict( X_test )
 y_pred_dynamic = model_dynamic. predict( X_test )

# Performance metrics
def display_metrics (y true , y pred , label):
 mae = mean_absolute_error(y true , y pred)
 mse = mean_squared_error ( y true , y pred)
 r2 = r2_score (y true , y pred)
 st.write ( f"###{label}Model Performance")
 st.write ( f"∗∗Mean Absolute Error (MAE):∗∗ {mae : . 2f}")
 st.write ( f"∗∗Mean Squared Error (MSE):∗∗ {mse:.2f }")

display_metrics ( y_test_total , y_pred_total, "Power total" )
display_metrics ( y_test_dynamic , y_pred_dynamic , "Power dynamic" )

  
     
 
 
