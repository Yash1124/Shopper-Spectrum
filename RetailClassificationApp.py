import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🛍️ Shopper Spectrum - Product Recommendations",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .product-name {
        font-weight: bold;
        color: #2c3e50;
        font-size: 1.1rem;
    }
    .product-score {
        color: #27ae60;
        font-size: 0.9rem;
    }
    .recommendation-header {
        color: #e74c3c;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        # Load your dataset - replace with your actual file path
        df = pd.read_csv('ecommerce_data.csv')
        
        # Basic data cleaning
        df = df.dropna(subset=['Description', 'StockCode'])
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Remove duplicates and clean descriptions
        df['Description'] = df['Description'].str.strip().str.upper()
        df = df.drop_duplicates(subset=['StockCode', 'Description'])
        
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please upload 'ecommerce_data.csv' to the app directory.")
        return None

@st.cache_data
def prepare_recommendation_data(df):
    """Prepare data for collaborative filtering"""
    # Create user-item matrix
    user_item_matrix = df.pivot_table(
        index='CustomerID', 
        columns='StockCode', 
        values='Quantity', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Create product similarity matrix using TF-IDF on descriptions
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Get unique products with descriptions
    products_df = df.groupby('StockCode')['Description'].first().reset_index()
    
    try:
        tfidf_matrix = tfidf.fit_transform(products_df['Description'])
        content_similarity = cosine_similarity(tfidf_matrix)
        
        # Create product lookup dictionary
        product_lookup = dict(zip(products_df['StockCode'], products_df['Description']))
        
        return user_item_matrix, content_similarity, products_df, product_lookup
    except:
        st.error("Error processing product descriptions for recommendations.")
        return None, None, None, None

def get_product_recommendations(product_name, products_df, content_similarity, product_lookup, top_n=5):
    """Get product recommendations based on content similarity"""
    
    # Find matching products
    matching_products = products_df[
        products_df['Description'].str.contains(product_name.upper(), na=False)
    ]
    
    if matching_products.empty:
        return None, "No matching products found. Please try a different product name."
    
    # Get the first matching product
    target_product = matching_products.iloc[0]
    target_idx = products_df[products_df['StockCode'] == target_product['StockCode']].index[0]
    
    # Get similarity scores
    similarity_scores = list(enumerate(content_similarity[target_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar products (excluding the target product itself)
    recommendations = []
    for idx, score in similarity_scores[1:top_n+1]:
        product_code = products_df.iloc[idx]['StockCode']
        product_desc = products_df.iloc[idx]['Description']
        recommendations.append({
            'StockCode': product_code,
            'Description': product_desc,
            'Similarity': score
        })
    
    return recommendations, None

def display_recommendations(recommendations):
    """Display recommendations in a styled format"""
    st.markdown('<p class="recommendation-header">🎯 Recommended Products:</p>', unsafe_allow_html=True)
    
    for i, product in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="product-card">
            <div class="product-name">#{i}. {product['Description']}</div>
            <div class="product-score">Stock Code: {product['StockCode']} | Similarity: {product['Similarity']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛍️ Shopper Spectrum</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #7f8c8d;">AI-Powered Product Recommendation System</h3>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 App Information")
        st.info("""
        **How it works:**
        1. Enter a product name
        2. Click 'Get Recommendations'
        3. View 5 similar products
        
        **Technology:**
        - Content-based filtering
        - TF-IDF vectorization
        - Cosine similarity
        """)
        
        st.header("📈 Model Info")
        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
            st.json(metadata)
        except:
            st.write("Model metadata not available")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Prepare recommendation data
    with st.spinner("🔄 Preparing recommendation engine..."):
        user_item_matrix, content_similarity, products_df, product_lookup = prepare_recommendation_data(df)
    
    if content_similarity is None:
        st.stop()
    
    # Main interface
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("🔍 Find Similar Products")
        
        # Product input
        product_name = st.text_input(
            "Enter Product Name:",
            placeholder="e.g., WHITE HANGING HEART T-LIGHT HOLDER",
            help="Enter any product name to find similar items"
        )
        
        # Get recommendations button
        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            if product_name.strip():
                with st.spinner("🔍 Finding similar products..."):
                    recommendations, error = get_product_recommendations(
                        product_name, products_df, content_similarity, product_lookup
                    )
                
                if error:
                    st.error(error)
                    
                    # Show available products hint
                    st.info("💡 **Suggestion:** Try searching for these popular products:")
                    sample_products = products_df['Description'].head(10).tolist()
                    for product in sample_products:
                        st.write(f"• {product}")
                        
                else:
                    display_recommendations(recommendations)
                    
                    # Show statistics
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Products", len(products_df))
                    with col2:
                        st.metric("Recommendations", len(recommendations))
                    with col3:
                        st.metric("Best Match Score", f"{recommendations[0]['Similarity']:.3f}")
            else:
                st.warning("⚠️ Please enter a product name to get recommendations.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #7f8c8d;'>Built with ❤️ using Streamlit | Shopper Spectrum © 2024</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
