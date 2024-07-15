import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('synthetic_myntra_dataset.csv')
df = df.drop('asin',axis=1)

# ARIMA Sales Forecasting
sales_data = df['sales']
model = ARIMA(sales_data, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

trend_keywords = [
    "stylish", "fashion", "trendy", "modern", "elegant", "chic", "popular", 
    "comfortable", "durable", "affordable", "quality", "classic", "unique", 
    "beautiful", "cool", "awesome", "great", "nice", "good", "excellent"
]

# Function to extract trends from reviews
def extract_trends(reviews):
    trends = []
    if pd.isnull(reviews) or not reviews:
        return ["general"]  # Assign a default keyword if no review or keywords found
    reviews_lower = reviews.lower()
    for keyword in trend_keywords:
        if keyword in reviews_lower:
            trends.append(keyword)
    if not trends:
        trends.append("general")  # Assign a default keyword if no keywords are found
    return trends

df['trends'] = df['reviews'].apply(extract_trends)

st.title('Myntra Dashboard')
# Display dataset
st.dataframe(df)

# Display popularity score distribution
st.subheader('Popularity Score Distribution')
st.bar_chart(df['popularity_score'])

# Display top products
st.subheader('Top Products')
top_products = df.sort_values(by='popularity_score', ascending=False).head(10)
st.table(top_products[['name', 'popularity_score']])


# Display sales forecast
st.subheader('Sales Forecast')
fig, ax = plt.subplots()
ax.plot(df['sales'], label='Historical Sales')
ax.plot(range(len(df['sales']), len(df['sales']) + len(forecast)), forecast, label='Forecast')
ax.legend()
ax.set_title('Sales Forecast')
ax.set_xlabel('Time')
ax.set_ylabel('Sales')
st.pyplot(fig)

# Display trend predictions
st.subheader('Trend Predictions')
st.write(df['trends'])
