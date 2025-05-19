import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load dataset
data = pd.read_csv('covid_19_indonesia_time_series_all.csv')

# Pilih kolom yang relevan dan sampling 10%
data = data[['Date', 'Location', 'Total Deaths', 'Total Recovered', 'Population Density',
             'Case Fatality Rate', 'Total Cases', 'Latitude', 'Longitude', 'New Cases']]
data_sample = data.sample(frac=0.1, random_state=42)

# Bersihkan kolom 'Case Fatality Rate'
data_sample['Case Fatality Rate'] = data_sample['Case Fatality Rate'].str.rstrip('%').astype('float') / 100.0

# ------------------------------
# SUPERVISED LEARNING: Prediksi Total Kasus
# ------------------------------
X = data_sample[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']]
y = data_sample['Total Cases']

model = LinearRegression()
model.fit(X, y)
data_sample['Predicted Total Cases'] = model.predict(X)
data_sample['Predicted Total Cases'] = data_sample['Predicted Total Cases'].clip(lower=0)

# ------------------------------
# UNSUPERVISED LEARNING: Clustering Lokasi
# ------------------------------
clustering_features = data_sample[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)

kmeans = KMeans(n_clusters=3, random_state=42)
data_sample['Cluster'] = kmeans.fit_predict(scaled_features)

# Map cluster ke level risiko (berdasarkan rata-rata total kasus)
cluster_mean = data_sample.groupby('Cluster')['Total Cases'].mean().sort_values()
risk_mapping = {
    cluster_mean.index[0]: 'Low Risk',
    cluster_mean.index[1]: 'Moderate Risk',
    cluster_mean.index[2]: 'High Risk'
}
data_sample['Risk Level'] = data_sample['Cluster'].map(risk_mapping)

# ------------------------------
# STREAMLIT DASHBOARD
# ------------------------------
st.set_page_config(layout='wide')
st.title('COVID-19 Indonesia Dashboard')

# Peta hasil clustering
st.header('Peta Interaktif Hasil Clustering Wilayah')
fig_cluster_map = px.scatter_geo(
    data_sample,
    lat='Latitude',
    lon='Longitude',
    color='Cluster',
    hover_name='Location',
    title='Cluster Wilayah Berdasarkan COVID-19',
    scope='asia',
    center={'lat': -2, 'lon': 118},
    projection='natural earth',
    height=500
)
st.plotly_chart(fig_cluster_map)

# Grafik tren kasus harian
st.header('Grafik Tren Kasus Harian')
fig_trend = px.line(data_sample.sort_values(by='Date'), x='Date', y='New Cases', title='Tren Kasus Harian (New Cases)')
st.plotly_chart(fig_trend)

# Ringkasan risiko wilayah
st.header('Ringkasan Tingkat Risiko Wilayah')
risk_summary = data_sample[['Location', 'Predicted Total Cases', 'Risk Level']].sort_values(by='Predicted Total Cases', ascending=False)
st.dataframe(risk_summary)

# Evaluasi model regresi
st.header('Evaluasi Model Prediksi')
r2_score = model.score(X, y)
st.write(f'R² Score (Akurasi Model Prediksi Total Kasus): **{r2_score:.2f}**')

# Elbow & Silhouette Score
st.subheader("Evaluasi Clustering (Elbow Method & Silhouette Score)")
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
ax1.set_xlabel('Jumlah Cluster')
ax1.set_ylabel('Inertia', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Elbow Method dan Silhouette Score')
fig.tight_layout()
st.pyplot(fig)
