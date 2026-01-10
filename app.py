import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from category_encoders import BinaryEncoder

# Page Config
st.set_page_config(page_title="Retail Sales Analytics", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Loading exactly what you dumped in your notebook
    model = joblib.load('decision_tree_model.pkl')
    scaler = joblib.load('scaler.pkl') 
    b_enc = joblib.load('binary_encoder.pkl') 
    o_enc = joblib.load('ordinal_encoder.pkl')
    ohe_enc = joblib.load('onehot_encoder.pkl')
    return model, scaler, b_enc, o_enc, ohe_enc

@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_sales.csv")
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    return df

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Retail Project")
page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction Engine"])

# --- DASHBOARD ---
if page == "Dashboard":
    st.title("📊 Sales Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${df['Total Spent'].sum():,.2f}")
    c2.metric("Avg Order", f"${df['Total Spent'].mean():.2f}")
    c3.metric("Total Items", f"{len(df):,}")
    
    fig = px.pie(df, names='Category', values='Total Spent', hole=0.4, title="Revenue by Category")
    st.plotly_chart(fig, use_container_width=True)

# --- PREDICTION (ERROR HANDLING LOGIC) ---
else:
    st.title("🤖 Predict Transaction Value")
    
    with st.form("input_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            cat = st.selectbox("Category", df['Category'].unique())
            item = st.selectbox("Item", df['Item'].unique())
            loc = st.selectbox("Location", df['Location'].unique())
            pay = st.selectbox("Payment Method", df['Payment Method'].unique())
        with col_b:
            price = st.number_input("Price Per Unit", value=float(df['Price Per Unit'].median()))
            qty = st.slider("Quantity", 1, 15, 1)
            disc = st.selectbox("Discount Applied", df['Discount Applied'].unique())
            cust_id = st.selectbox("Customer ID (Sample)", df['Customer ID'].unique()[:50])
        
        predict_btn = st.form_submit_button("Calculate Total Spent")

    if predict_btn:
        try:
            model, scaler, b_enc, o_enc, ohe_enc = load_assets()

            # --- STEP 1: RECONSTRUCT RAW INPUT ---
            # Recreating the exact dataframe structure from your notebook's X_train
            input_df = pd.DataFrame({
                'Transaction ID': [0], # Dummy
                'Customer ID': [cust_id],
                'Category': [cat],
                'Item': [item],
                'Price Per Unit': [price],
                'Quantity': [qty],
                'Payment Method': [pay],
                'Location': [loc],
                'Transaction Date': [pd.Timestamp.now()],
                'Discount Applied': [disc]
            })

            # Replicate your engineered feature
            input_df['Customer_Transaction_Count'] = 1 

            # --- STEP 2: SEQUENTIAL TRANSFORMATIONS ---
            # This order must match your notebook exactly to avoid shape errors
            
            # A. Ordinal Encoding
            input_df[['Category']] = o_enc.transform(input_df[['Category']])

            # B. One-Hot Encoding (for Low Cardinality columns)
            # Adjust the list below to match what you put in 'low_cardinality' in the notebook
            low_card_cols = ["Payment Method", "Location"] 
            ohe_data = ohe_enc.transform(input_df[low_card_cols])
            ohe_cols = ohe_enc.get_feature_names_out(low_card_cols)
            input_df = input_df.drop(columns=low_card_cols).join(pd.DataFrame(ohe_data, columns=ohe_cols, index=input_df.index))

            # C. Binary Encoding (for High Cardinality)
            final_df = b_enc.transform(input_df)

            # D. Final Column Drops (as done in notebook)
            # You dropped 'Transaction Date' and potentially IDs before scaling
            cols_to_drop = ['Transaction Date', 'Transaction ID']
            final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns])

            # --- STEP 3: SCALE AND PREDICT ---
            final_scaled = scaler.transform(final_df)
            prediction = model.predict(final_scaled)

            st.success(f"### Predicted Total Spent: ${prediction[0]:.2f}")
            st.balloons()

        except Exception as e:
            st.error(f"Transformation Error: {e}")
            st.warning("Hint: Check if the columns in your notebook's X_train match this sequence.")
