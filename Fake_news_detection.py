import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    html, body, [class*="css"]  {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.8), rgba(48, 43, 99, 0.8));
        border-bottom: 3px solid #00d4ff;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-top: 0.5rem;
    }
    
    .prediction-container {
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid;
        background: rgba(30, 30, 40, 0.8);
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .prediction-real {
        border-left-color: #00d4ff;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 100, 150, 0.1));
    }
    
    .prediction-fake {
        border-left-color: #ff006e;
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.1), rgba(150, 0, 50, 0.1));
    }
    
    .metric-box {
        background: rgba(50, 50, 70, 0.8);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #00d4ff;
        text-align: center;
    }
    
    .metric-box h3 {
        color: #00d4ff;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-box .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00ff88;
    }
    
    .stTextInput, .stTextArea {
        background-color: rgba(50, 50, 70, 0.8) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        color: #000;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff006e, #ffbe0b, #00d4ff);
        height: 10px;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stats-header {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 0, 110, 0.1);
        border-left: 4px solid #ff006e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: rgba(255, 0, 110, 0.15);
        border-left: 4px solid #ff006e;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(0, 255, 136, 0.15);
        border-left: 4px solid #00ff88;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(50, 50, 70, 0.8);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        color: #000;
    }
</style>
""", unsafe_allow_html=True)


def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    return text

def predict_news(text, model):
    """Predict if news is fake or real"""
    cleaned_text = clean_text(text)
    
    if not cleaned_text or len(cleaned_text.split()) < 3:
        return None, None
    
    prediction = model.predict([cleaned_text])[0]
    probability = model.predict_proba([cleaned_text])[0]
    
    return prediction, probability

def clear_cache():
    """Clear model and metrics cache"""
    model_path = "fake_news_model.pkl"
    metrics_path = "fake_news_metrics.pkl"
    
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            st.success(f"✅ Deleted {model_path}")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
            st.success(f"✅ Deleted {metrics_path}")
    except Exception as e:
        st.error(f"❌ Error deleting cache: {e}")

@st.cache_resource
def load_or_train_model():
    """Load model or train if not exists - WITH GUARANTEED METRICS"""
    model_path = "fake_news_model.pkl"
    metrics_path = "fake_news_metrics.pkl"

    if os.path.exists(model_path) and os.path.exists(metrics_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
        
            return model, metrics
        except Exception as e:
            st.sidebar.warning(f"⚠️ Cache load error: {e}. Retraining...")
    
    # Load data
    with st.spinner("🔄 Loading and preprocessing data..."):
        try:
            possible_paths = [
                "FakeNewsNet.csv"]
            
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.sidebar.info(f"📁 Loaded from: {path}")
                    break

            df = df[['title', 'real']].copy()
            df = df.dropna()
            
            if len(df) == 0:
                st.error("❌ No valid data in CSV!")
                return None, None
           
            df['title'] = df['title'].apply(clean_text)
            df = df[df['title'].str.len() > 0] 
            
            st.sidebar.info(f"📊 Dataset: {len(df)} articles loaded")
            st.sidebar.info(f"   Fake: {(df['real']==0).sum()} | Real: {(df['real']==1).sum()}")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    with st.spinner("🤖 Training machine learning model... (This may take 1-2 minutes)"):
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['title'], df['real'], 
                test_size=0.2, 
                random_state=42,
                stratify=df['real']
            )
            
            st.sidebar.info(f"📈 Training set: {len(X_train)} articles")
            st.sidebar.info(f"📈 Test set: {len(X_test)} articles")
            
            # Create pipeline
            model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    min_df=5,
                    max_df=0.8,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('clf', MultinomialNB())
            ])
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
    
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            # Save to disk
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(metrics_path, 'wb') as f:
                    pickle.dump(metrics, f)
                st.sidebar.success("✅ Model & Metrics saved to disk!")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not save to disk: {e}")
            
            return model, metrics
            
        except Exception as e:
            st.error(f"❌ Error training model: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None, None


# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Fake News Detector</h1>
    <p>Fake News Detection System Using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Load or train model
model, metrics = load_or_train_model()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ App Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence required to make a prediction"
    )
    
    st.divider()
    st.markdown("### 📚 About")
    st.info(
        "This app uses Naive Bayes with TF-IDF vectorization to detect fake news. "
        "It's trained on real-world news datasets."
    )
    
# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Detect News", "📊 Model Performance", "📈 Batch Analysis", "❓ How It Works"])


with tab1:
    st.markdown("### 📝 Enter Article Content")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        news_text = st.text_area(
            "Paste your news article here:",
            placeholder="Enter the full text of the article you want to analyze...",
            height=250,
            key="news_input"
        )
    
    with col2:
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🔍 Analyze Article", use_container_width=True):
            if news_text.strip():
                prediction, probability = predict_news(news_text, model)
                
                if prediction is not None:
                    st.session_state.last_prediction = prediction
                    st.session_state.last_probability = probability
                    st.session_state.last_text = news_text
                else:
                    st.warning("⚠️ Please enter a longer article for analysis")
    
    
    if 'last_prediction' in st.session_state:
        prediction = st.session_state.last_prediction
        probability = st.session_state.last_probability
        
        # Result container
        if prediction == 0:  
            st.markdown(f"""
            <div class="prediction-container prediction-fake">
                <h2 style="color: #ff006e; margin-top: 0;">⚠️ LIKELY FAKE NEWS</h2>
                <p style="font-size: 1.1rem; color: #e0e0e0;">
                    This article has characteristics commonly found in fake or misinformation content.
                </p>
                <div style="margin-top: 1rem;">
                    <p><strong>Confidence:</strong> {probability[0]*100:.2f}%</p>
                </div>
                <div class="confidence-bar" style="background: linear-gradient(90deg, #ff006e 0%, #ff006e {probability[0]*100}%, rgba(255,0,110,0.2) {probability[0]*100}%);"></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Caution:</strong> 
                <ul>
                    <li>Verify facts from multiple credible sources</li>
                    <li>Check the author and publication credentials</li>
                    <li>Look for sensational language or emotional appeals</li>
                    <li>Cross-reference key claims with fact-checking websites</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:  
            st.markdown(f"""
            <div class="prediction-container prediction-real">
                <h2 style="color: #00d4ff; margin-top: 0;">✅ LIKELY AUTHENTIC NEWS</h2>
                <p style="font-size: 1.1rem; color: #e0e0e0;">
                    This article appears to have characteristics of legitimate news content.
                </p>
                <div style="margin-top: 1rem;">
                    <p><strong>Confidence:</strong> {probability[1]*100:.2f}%</p>
                </div>
                <div class="confidence-bar" style="background: linear-gradient(90deg, #00d4ff 0%, #00d4ff {probability[1]*100}%, rgba(0,212,255,0.2) {probability[1]*100}%);"></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>ℹ️ Note:</strong> While this article appears authentic, always practice media literacy:
                <ul>
                    <li>Still verify major claims independently</li>
                    <li>Check for multiple perspectives on the issue</li>
                    <li>Consider the source's track record for accuracy</li>
                    <li>Be aware of potential bias in reporting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Model Metrics")
    
    if metrics and isinstance(metrics, dict) and 'accuracy' in metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Accuracy</h3>
                <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Precision</h3>
                <div class="metric-value">{metrics['precision']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Recall</h3>
                <div class="metric-value">{metrics['recall']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <h3>F1-Score</h3>
                <div class="metric-value">{metrics['f1']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <strong>✅ All metrics loaded successfully!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### 📊 Detailed Metrics Explanation")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [f"{metrics['accuracy']*100:.2f}%", f"{metrics['precision']*100:.2f}%", 
                     f"{metrics['recall']*100:.2f}%", f"{metrics['f1']*100:.2f}%"],
            'Description': [
                'Overall correctness of the model',
                'Of predicted fake, how many are actually fake',
                'Of actual fake articles, how many we caught',
                'Harmonic mean of Precision and Recall'
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        
    else:
        st.markdown("""
        <div class="error-box">
            <strong>❌ Metrics not available</strong>
            <p>This can happen if:</p>
            <ul>
                <li>Dataset file not found</li>
                <li>CSV doesn't have 'title' and 'real' columns</li>
                <li>Training failed silently</li>
            </ul>
            <p><strong>Solutions:</strong></p>
            <ol>
                <li>Check sidebar for error messages</li>
                <li>Enable "Show Debug Info" in sidebar</li>
                <li>Click "Clear Cache & Retrain" button</li>
                <li>Verify your CSV file path and columns</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🔧 Model Architecture")
    
    architecture = {
        'Component': ['Feature Extraction', 'Classification', 'Training Data', 'Validation Method'],
        'Details': [
            'TF-IDF Vectorization (5000 features, bi-grams)',
            'Naive Bayes Classifier',
            'Real-world news dataset (FakeNewsNet)',
            '80-20 Train-Test Split'
        ]
    }
    
    st.dataframe(pd.DataFrame(architecture), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📈 Analyze Multiple Articles")
    
    batch_text = st.text_area(
        "Enter multiple articles (separate each with '---')",
        placeholder="Article 1\n---\nArticle 2\n---\nArticle 3",
        height=300
    )
    
    if st.button("🔄 Analyze Batch", use_container_width=True):
        if batch_text.strip():
            articles = [a.strip() for a in batch_text.split('---') if a.strip()]
            
            if not articles:
                st.warning("⚠️ No articles found. Please enter at least one article.")
            else:
                results = []
                for i, article in enumerate(articles, 1):
                    prediction, probability = predict_news(article, model)
                    
                    if prediction is not None:
                        results.append({
                            'Article #': i,
                            'Label': '🔴 FAKE' if prediction == 0 else '🟢 REAL',
                            'Confidence': f"{max(probability)*100:.2f}%",
                            'Fake Score': f"{probability[0]*100:.2f}%",
                            'Real Score': f"{probability[1]*100:.2f}%"
                        })
                
                if results:
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
         
                    fake_count = sum(1 for r in results if '🔴' in r['Label'])
                    real_count = sum(1 for r in results if '🟢' in r['Label'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fake Articles Detected", fake_count)
                    with col2:
                        st.metric("Real Articles Detected", real_count)
                else:
                    st.warning("⚠️ No valid articles found in batch.")

with tab4:
    st.markdown("### 🔬 How This Detector Works")
    
    st.markdown("""
    #### 1️⃣ **Data Collection**
    - Dataset: FakeNewsNet - Real and fake news from multiple sources
    - Label: 1 = Real News, 0 = Fake News
    - CSV Format: Must have 'title' and 'real' columns
    
    #### 2️⃣ **Text Preprocessing**
    - Remove URLs, emails, special characters
    - Convert to lowercase
    - Remove extra whitespace
    
    #### 3️⃣ **Feature Extraction (TF-IDF)**
    - Converts text into numerical features
    - Uses up to 5000 most important features
    - Captures bi-grams (two-word combinations)
    - Removes common stop words
    
    #### 4️⃣ **Machine Learning**
    - **Algorithm**: Multinomial Naive Bayes
    - **Why Naive Bayes?**
      - Works well with text data
      - Fast and efficient
      - Provides probability scores
    
    #### 5️⃣ **Prediction**
    - Model outputs probability for each class
    - Results shown as percentage
    
    ### 📚 Key Indicators of Fake News
    The model learns to identify sensational language, emotional triggers, and false claims.
    
    ### ⚠️ Limitations
    - Not 100% accurate
    - Always verify with multiple sources
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #b0b0b0; padding: 2rem; margin-top: 3rem;'>
    <p><strong>🔍 Fake News Detector v2.0</strong></p>
</div>
""", unsafe_allow_html=True)
