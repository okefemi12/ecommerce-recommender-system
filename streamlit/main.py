import streamlit as st
import pandas as pd
import requests
import os
from io import BytesIO
from dotenv import load_dotenv

# -------------------
# Load environment variables
# -------------------
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "https://ecommerce-recommender-system-iikz.onrender.com")

# -------------------
# Streamlit Config
# -------------------
st.set_page_config(page_title="🛒 E-commerce Recommender System", layout="centered")
st.title("🧠 Multi-Model E-commerce Recommender System")

st.markdown("""
This app provides recommendations using **Transformer, BERT, and VAE-based** models.
You can upload your dataset and test different recommendation strategies.
""")

# -------------------
# API Health Check
# -------------------
st.sidebar.title("🔌 Backend Status")
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=10)
    if health.status_code == 200:
        st.sidebar.success("✅ Backend is Online")
    else:
        st.sidebar.warning("⚠️ Backend reachable but not healthy.")
except Exception as e:
    st.sidebar.error("❌ Backend not reachable")
    st.sidebar.caption(str(e))

# -------------------
# File Upload Section
# -------------------
st.subheader("Step 1: Upload Your Dataset (products.csv)")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        csv_buffer = df.to_csv(index=False).encode('utf-8')
        files = {"file": ("products.csv", BytesIO(csv_buffer), "text/csv")}

        upload_url = f"{API_BASE_URL}/upload_dataset"

        with st.spinner("📤 Uploading dataset to backend..."):
            upload_response = requests.post(upload_url, files=files, timeout=60)

        if upload_response.status_code == 200:
            st.success("✅ Dataset uploaded and backend updated!")
        else:
            st.error("❌ Failed to upload dataset to backend.")
            st.error(upload_response.json().get("error", "Unknown error"))

    except Exception as e:
        st.error(f"⚠️ Upload failed: {str(e)}")
else:
    st.info("Please upload a valid CSV or Excel file. Columns must include: `user_id`, `product_id`, `brand`, `price`, `category_code`, `event_type` (view/purchase).")

# -------------------
# Model Selection
# -------------------
st.subheader("Step 2: Choose Recommendation Model")
model_choice = st.selectbox(
    "Select a model to generate recommendations:",
    [
        "Transformer (Sequential)",
        "BERT-based: User Profile Recommender",
        "BERT-based: Similar Brand/Price",
        "Variation Autoencoder: Collaborative Filter"
    ]
)

# -------------------
# Helper: Display recommendations
# -------------------
def display_recommendations(recs, title="Recommendations"):
    if not recs:
        st.warning("No recommendations found.")
        return
    st.success(f"Top {len(recs)} {title}:")
    for i, rec in enumerate(recs, start=1):
        st.markdown(f"""
        **{i}. {rec.get('product_name', 'Unknown Product')}**
        - 🏷️ Brand: {rec.get('brand', 'N/A')}
        - 💰 Price: ${rec.get('price', 'N/A')}
        - 📦 Category: {rec.get('category_code', 'N/A')}
        """)

# -------------------
# Transformer (Sequential)
# -------------------
if model_choice == "Transformer (Sequential)":
    st.subheader("Step 3: Transformer Recommendations")
    user_id = st.text_input("Enter User ID (e.g., 513103710)", key="transformer_user")

    if st.button("Get Transformer Recommendations"):
        if not user_id.strip():
            st.warning("Please enter a valid user ID.")
        else:
            url = f"{API_BASE_URL}/recommend"
            payload = {"user_id": user_id}
            with st.spinner("🤖 Generating Transformer recommendations..."):
                response = requests.post(url, json=payload)
            if response.status_code == 200:
                recs = response.json().get("recommendations", [])
                display_recommendations(recs, "Transformer Recommendations")
            else:
                st.error(f"Error: {response.status_code}")
                st.error(response.json().get("error", "Unknown error"))

# -------------------
# BERT: User Profile
# -------------------
elif model_choice == "BERT-based: User Profile Recommender":
    st.subheader("Step 3: BERT User Profile Recommendations")
    user_id = st.text_input("Enter User ID (e.g., 513103710)", key="bert_user")
    top_k = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

    if st.button("Get BERT User Recommendations"):
        if not user_id.strip():
            st.warning("Please enter a valid user ID.")
        else:
            url = f"{API_BASE_URL}/recommend_user_content_bert"
            payload = {"user_id": user_id, "top_k": top_k}
            with st.spinner("🧠 Generating BERT User Profile recommendations..."):
                response = requests.post(url, json=payload)
            if response.status_code == 200:
                recs = response.json()
                display_recommendations(recs, f"Top {top_k} BERT Recommendations")
            else:
                st.error(f"Error: {response.status_code}")
                st.error(response.json().get("error", "Unknown error"))

# -------------------
# BERT: Similar Brand/Price
# -------------------
elif model_choice == "BERT-based: Similar Brand/Price":
    st.subheader("Step 3: Similar Brand/Price Recommendations")
    search_by = st.radio("Search by:", ["Brand", "Product ID"])

    brand_name = None
    product_id_input = None
    if search_by == "Brand":
        brand_name = st.text_input("Enter Brand Name (e.g., wincars)")
    else:
        product_id_input = st.text_input("Enter Product ID (e.g., 513103710)")

    top_k = st.number_input("Top K Similar Products", min_value=1, max_value=20, value=5)
    min_price = st.number_input("Min Price", value=0.0)
    max_price = st.number_input("Max Price", value=10000.0)
    same_category = st.checkbox("Only show same category products", value=True)

    if st.button("Get Similar Products"):
        url = f"{API_BASE_URL}/recommend_brand_similarity_bert"
        payload = {
            "brand": brand_name,
            "product_id": product_id_input,
            "top_k": top_k,
            "min_price": min_price,
            "max_price": max_price,
            "same_category": same_category
        }
        with st.spinner("🔍 Searching for similar products..."):
            response = requests.post(url, json=payload)
        if response.status_code == 200:
            recs = response.json()
            display_recommendations(recs, f"Top {top_k} Similar Products")
        else:
            st.error(f"Error: {response.status_code}")
            st.error(response.json().get("error", "Unknown error"))

# -------------------
# VAE: Collaborative Filter
# -------------------
elif model_choice == "Variation Autoencoder: Collaborative Filter":
    st.subheader("Step 3: Collaborative Recommendations")
    user_id = st.text_input("Enter User ID (e.g., 513103710)", key="vae_user")
    top_n = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

    if st.button("Get Collaborative Recommendations"):
        url = f"{API_BASE_URL}/vae_collaborative_recommender"
        payload = {"user_id": user_id, "top_n": top_n}
        with st.spinner("🤝 Generating Collaborative recommendations..."):
            response = requests.post(url, json=payload)
        if response.status_code == 200:
            recs = response.json().get("recommendations", [])
            display_recommendations(recs, f"Top {top_n} Collaborative Recommendations")
        else:
            st.error(f"Error: {response.status_code}")
            st.error(response.json().get("error", "Unknown error"))
