# 🛍️ Shopper Spectrum - Product Recommendation App

An AI-powered product recommendation system using collaborative filtering and content-based approaches.

## Features
- **Product Recommendations**: Get 5 similar products based on input
- **Content-Based Filtering**: Uses TF-IDF and cosine similarity
- **Interactive UI**: Clean Streamlit interface
- **Real-time Processing**: Fast recommendation engine

## Setup & Installation

### Local Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your dataset as `ecommerce_data.csv`
4. Run: `streamlit run RetailClassificationApp.py`

### Required Dataset Format
Your `ecommerce_data.csv` should contain:
- `CustomerID`: Customer identifier
- `StockCode`: Product code
- `Description`: Product description
- `Quantity`: Purchase quantity
- `UnitPrice`: Product price

## Usage
1. Enter a product name in the text input
2. Click "Get Recommendations"
3. View 5 similar products with similarity scores

## Technology Stack
- **Frontend**: Streamlit
- **ML**: Scikit-learn (TF-IDF, Cosine Similarity)
- **Data**: Pandas, NumPy

## License
MIT License
