import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv('synthetic_myntra_dataset.csv')

# Function to search product by ID
def search_product_by_id(product_id, df):
    result = df[df['id'].astype(str) == str(product_id)]
    return result

# Streamlit Page for Product Popularity Score Prediction
st.title('Product Popularity Score Prediction')

# Input box for product ID
product_id_query = st.text_input('Enter Product ID')

# Search and display results
if product_id_query:
    result = search_product_by_id(product_id_query, df)
    if not result.empty:
        st.write('Product found:')
        st.table(result[['id', 'name', 'popularity_score']])
    else:
        st.write('No product found with the given ID.')
