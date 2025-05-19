import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('covid_19_indonesia_time_series_all.csv')

# Select relevant columns and sample 10% of the data
data = data[['Date', 'Location', 'Total Deaths', 'Total Recovered', 'Population Density',
             'Case Fatality Rate', 'Total Cases', 'Latitude', 'Longitude', 'New Cases']]
data_sample = data.sample(frac=0.1, random_state=42)

# Clean the 'Case Fatality Rate' column
data_sample['Case Fatality Rate'] = data_sample['Case Fatality Rate'].str.rstrip('%').astype('float') / 100.0

# Prepare the features and target variable for regression
features = data_sample[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']]
target = data_sample['Total Cases']

# Train the linear regression model
model = LinearRegression()
model.fit(features, target)

# Predict the total cases
data_sample['Predicted Total Cases'] = model.predict(features)
data_sample['Predicted Total Cases'] = data_sample['Predicted Total Cases'].clip(lower=0)

# Streamlit App
st.title('COVID-19 Indonesia Dashboard')

# --- Interactive Map of COVID-19 Cases ---
st.header('Interactive Map of Predicted COVID-19 Cases')
fig_map = px.scatter_geo(
    data_sample,
    lat='Latitude',
    lon='Longitude',
    color='Predicted Total Cases',
    hover_name='Location',
    size='Predicted Total Cases',
    projection='natural earth',
    title='COVID-19 Predicted Cases in Indonesia',
    scope='asia',
    center={'lat': -2, 'lon': 118},
    height=500
)
st.plotly_chart(fig_map)

# --- Line chart of daily new cases ---
st.header('Daily New Cases Trend')
fig_line = px.line(data_sample.sort_values(by='Date'), x='Date', y='New Cases', title='Daily New Cases')
st.plotly_chart(fig_line)

# --- Risk Level Summary ---
st.header('Risk Level Summary')
median_predicted = data_sample['Predicted Total Cases'].median()
data_sample['Risk Level'] = np.where(data_sample['Predicted Total Cases'] > median_predicted, 'High Risk', 'Low Risk')
risk_summary = data_sample[['Location', 'Predicted Total Cases', 'Risk Level']]
st.write(risk_summary)

# --- Model Evaluation ---
st.header('Model Evaluation')
r2_score = model.score(features, target)
st.write(f'R² Score: {r2_score:.2f}')

# --- KMeans Clustering ---
st.header('Clustering COVID-19 Impacted Regions (KMeans)')

# Select features for clustering
clustering_features = data_sample[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)

# Fit KMeans (with 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
data_sample['Cluster'] = kmeans.fit_predict(scaled_features)

# Show cluster map
fig_cluster = px.scatter_geo(
    data_sample,
    lat='Latitude',
    lon='Longitude',
    color='Cluster',
    hover_name='Location',
    title='Cluster of Regions Based on COVID-19 Data',
    scope='asia',
    center={'lat': -2, 'lon': 118},
    projection='natural earth',
    height=500
)
st.plotly_chart(fig_cluster)

# Optional: Elbow & Silhouette Method Plot
st.subheader("Elbow Method & Silhouette Score (Optional)")

inertia = []
silhouette = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled_features)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(scaled_features, labels))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(k_range, inertia, 'bo-', label='Inertia')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Inertia', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Elbow Method and Silhouette Score')
fig.tight_layout()
st.pyplot(fig)
