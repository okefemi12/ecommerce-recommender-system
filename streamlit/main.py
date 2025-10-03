import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="E-commerce Recommendation System", layout="centered")
st.title("🛒 Multi-Model Product Recommender System")

# -------------------
# Upload Dataset Step
# -------------------
st.subheader("Step 1: Upload Your Dataset (products.csv)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())
    # Save to Flask-accessible location
    uploaded_path = os.path.join("..", "app", "products.csv")
    df.to_csv(uploaded_path, index=False)
else:
    st.info("Please upload a CSV with columns: user_id, product_id, brand, price, category_code,event_type(view,purchase)")

# -------------------
# Choose Recommender
# -------------------
st.subheader("Step 2: Choose Recommendation Model")

model_choice = st.selectbox(
    "Select a recommendation model:",
    [
        "Transformer (Sequential)",
        "BERT-based: User Profile Recommender",
        "BERT-based: Similar Brand/Price",
        "Varation Autoencoder: Collaborative Filter Recommender"
    ]
)

# -------------------
# Transformer 
# -------------------
if model_choice == "Transformer (Sequential)":
    st.subheader("Step 3: Enter User ID for Transformer Recommendation")
    user_id = st.text_input("Enter a valid user ID (e.g., 513103710)", key="transformer_user")

    if st.button("Get Transformer Recommendations"):
        try:
            url = "http://localhost:5000/recommend"
            payload = {"user_id": user_id}
            response = requests.post(url, json=payload, )

            if response.status_code == 200:
                recs = response.json()["recommendations"]
                st.success("Top 5 Recommendations:")
                st.dataframe(pd.DataFrame(recs))
            else:
                st.error(f"Server error: {response.status_code}")
                st.error(response.json()["error"])
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
 
# -------------------
# BERT: User Profile
# -------------------
elif model_choice == "BERT-based: User Profile Recommender":
    st.subheader("Step 3: Enter User ID for BERT User Profile Recommendation")
    user_id_bert = st.text_input("Enter a valid user ID (e.g., 513103710)", key="bert_user")

    top_k_bert = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

    if st.button("Get BERT User Recommendations"):
        try:
            url = "http://localhost:5000/recommend_user_content_bert"
            payload = {"user_id": user_id_bert, "top_k": top_k_bert}
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                recs = response.json()
                st.success(f"Top {top_k_bert} Recommendations:")
                st.dataframe(pd.DataFrame(recs))
            else:
                st.error(f"Server error: {response.status_code}")
                st.error(response.json()["error"])
        except Exception as e:
            st.error(f"Request failed: {str(e)}")

# -------------------
# BERT: Similar Brand & Price
# -------------------
elif model_choice == "BERT-based: Similar Brand/Price":
    st.subheader("Step 3: Enter Product/Brand Information for Similarity Search")

    search_by = st.radio("Search by:", ["Brand", "Product ID"])

    if search_by == "Brand":
        brand_name = st.text_input("Enter Brand Name (e.g., wincars)", key="brand_search")
        product_id_input = None
    else:
        product_id_input = st.text_input("Enter Product ID (e.g., 513103710)", key="product_search")
        brand_name = None

    top_k_similar = st.number_input("Top K Similar Products", min_value=1, max_value=20, value=5)
    min_price = st.number_input("Min Price (optional)", value=0.0)
    max_price = st.number_input("Max Price (optional)", value=10000.0)
    same_category = st.checkbox("Only show products from the same category", value=True)

    if st.button("Get Similar Products"):
        try:
            url = "http://localhost:5000/recommend_brand_similarity_bert"
            payload = {
                "brand": brand_name,
                "product_id": product_id_input,
                "top_k": top_k_similar,
                "min_price": min_price,
                "max_price": max_price,
                "same_category": same_category
            }
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                recs = response.json()
                if len(recs) == 0:
                    st.warning("No similar products found.")
                else:
                    st.success(f"Top {top_k_similar} Similar Products:")
                    st.dataframe(pd.DataFrame(recs))
            else:
                st.error(f"Server error: {response.status_code}")
                st.error(response.json()["error"])
        except Exception as e:
            st.error(f"Request failed: {str(e)}")

elif model_choice == "Varation Autoencoder: Collaborative Filter Recommender":
     st.subheader("Step 3: Enter User ID for Collab Recommendations")
     user_id_vae = st.text_input("Enter a valid user ID (e.g., 513103710)", key="vae_user")

     top_n_vae= st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)
     if st.button("Get Collaborative Recommendations"):
         try:
             url= "http://localhost:5000/vae_collaborative_recommender"
             payload = {"user_id": user_id_vae, "top_n": top_n_vae}
             response = requests.post(url, json=payload)
             if response.status_code == 200:
                recs = response.json()
                recs = response.json().get("recommendations", [])
                if recs:
                   st.success(f"Top {top_n_vae} Recommendations:")
                   st.dataframe(pd.DataFrame(recs))
                else:
                    st.warning("No recommendations found for this user.")
             else:
                st.error(f"Server error: {response.status_code}")
                st.error(response.json()["error"])
         except Exception as e:
          st.error(f"Request failed: {str(e)}")

         
         
         

