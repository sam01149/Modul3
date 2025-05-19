
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Load the dataset
data = pd.read_csv('covid_19_indonesia_time_series_all.csv')

# Clean the 'Case Fatality Rate' column
data['Case Fatality Rate'] = data['Case Fatality Rate'].str.rstrip('%').astype('float') / 100.0

# Prepare the features and target variable
features = data[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']]
target = data['Total Cases']

# Train the linear regression model
model = LinearRegression()
model.fit(features, target)

# Predict the total cases
data['Predicted Total Cases'] = model.predict(features)

# Clip the predicted total cases to non-negative values
data['Predicted Total Cases'] = data['Predicted Total Cases'].clip(lower=0)

# Create a Streamlit app
st.title('COVID-19 Indonesia Dashboard')

# Interactive map showing clustering of regions
st.header('Interactive Map of COVID-19 Cases')
fig_map = px.scatter_geo(data, lat='Latitude', lon='Longitude', color='Predicted Total Cases',
                         hover_name='Location', size='Predicted Total Cases',
                         projection='natural earth')
st.plotly_chart(fig_map)

# Line chart of daily new cases
st.header('Daily New Cases Trend')
fig_line = px.line(data, x='Date', y='New Cases', title='Daily New Cases')
st.plotly_chart(fig_line)

# Summary table classifying regions by risk level
st.header('Risk Level Summary')
data['Risk Level'] = np.where(data['Predicted Total Cases'] > data['Predicted Total Cases'].median(), 'High Risk', 'Low Risk')
risk_summary = data[['Location', 'Predicted Total Cases', 'Risk Level']]
st.write(risk_summary)

# Model evaluation
st.header('Model Evaluation')
r2_score = model.score(features, target)
st.write(f'R² Score: {r2_score:.2f}')
