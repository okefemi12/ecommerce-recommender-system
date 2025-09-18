# E-Commerce Recommendation System  

This project implements **Collaborative Filtering (CoNAR)**, **content-based** and **sequential recommendation systems** for an e-commerce platform.  

- **Content-Based Filtering**: Uses brand, category, and product metadata to recommend similar products.  
- **Sequential Model**: Uses Transformer-based architecture to predict the next likely purchase based on user interaction history.
- **Collaborative Filtering (CoNAR)**: Leverages the user–item interaction matrix and applies neural attention mechanisms to capture hidden patterns in user behavior, enabling recommendations even when explicit product metadata is limited.

### Business Value  

- **Personalization at Scale**: Users receive recommendations tailored to their preferences, browsing history, and hidden interaction patterns.  
- **Higher Conversion Rates**: By predicting the next likely purchase, the system nudges users toward completing transactions.  
- **Enhanced Engagement**: Personalized feeds increase session length, repeat visits, and customer retention.  
- **Cold-Start Handling**: Content-based methods mitigate new product/user scenarios, while collaborative and sequential models improve relevance for active users.  
- **Business Growth**: Overall, the system translates **user behavior into actionable insights**, driving sales, customer satisfaction, and long-term loyalty.  


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


