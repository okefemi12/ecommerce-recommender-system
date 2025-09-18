# E-Commerce Recommendation System  

This project implements both **content-based** and **sequential recommendation systems** for an e-commerce platform.  

- **Content-Based Filtering**: Uses brand, category, and product metadata to recommend similar products.  
- **Sequential Model**: Uses Transformer-based architecture to predict the next likely purchase based on user interaction history.  

## Dataset  
- Collected from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).  
- ~7M user events across multiple categories (views, carts, purchases).  

## Project Structure  
- `notebooks/`: EDA and model training notebooks.  
- `flask_app/`: Flask backend to serve trained models via API.  
- `report/`: Markdown project report with objectives, business value, conclusions, and saved plots.  

## Running the Flask API  
```bash
cd flask_app
pip install -r requirements.txt
python app.py
