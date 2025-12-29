import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        color: #e74c3c;
        font-weight: bold;
        font-size: 24px;
    }
    .medium-risk {
        color: #f39c12;
        font-weight: bold;
        font-size: 24px;
    }
    .low-risk {
        color: #27ae60;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoder_geo.pkl', 'rb') as f:
        label_encoder_geo = pickle.load(f)
    with open('models/label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('data/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, label_encoder_geo, label_encoder_gender, feature_names

# Load everything
model, scaler, label_encoder_geo, label_encoder_gender, feature_names = load_model_and_preprocessors()

# Title
st.markdown('<p class="main-header">🏦 Bank Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predict Churn", "📊 Model Performance", "ℹ️ About"])

# TAB 1: PREDICTION
with tab1:
    st.markdown('<p class="sub-header">Customer Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        credit_score = st.slider("Credit Score", 300, 850, 650, help="Customer's credit score (300-850)")
        age = st.slider("Age", 18, 100, 35, help="Customer's age in years")
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"], help="Customer's country")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
    
    with col2:
        st.markdown("#### Account Details")
        tenure = st.slider("Tenure (years)", 0, 10, 5, help="How long they've been a customer")
        balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0, 
                                   help="Current account balance")
        num_products = st.slider("Number of Products", 1, 4, 2, help="How many bank products they use")
        has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    
    with col3:
        st.markdown("#### Activity")
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"], 
                                 help="Has the customer been active recently?")
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0, step=1000.0,
                                          help="Customer's estimated annual salary")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        
        # Prepare input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [label_encoder_geo.transform([geography])[0]],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [1 if has_credit_card == "Yes" else 0],
            'IsActiveMember': [1 if is_active == "Yes" else 0],
            'EstimatedSalary': [estimated_salary]
        })
        
        # Scale the data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        churn_probability = prediction_proba[1] * 100
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)
        
        # Create three columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Churn Probability", f"{churn_probability:.1f}%")
        
        with res_col2:
            if churn_probability >= 70:
                risk_level = "HIGH RISK"
                risk_color = "high-risk"
            elif churn_probability >= 40:
                risk_level = "MEDIUM RISK"
                risk_color = "medium-risk"
            else:
                risk_level = "LOW RISK"
                risk_color = "low-risk"
            
            st.markdown(f'<p class="{risk_color}">Risk Level: {risk_level}</p>', unsafe_allow_html=True)
        
        with res_col3:
            recommendation = "✅ RETAIN" if prediction == 0 else "⚠️ AT RISK"
            st.metric("Recommendation", recommendation)
        
        # Risk explanation
        st.markdown("---")
        st.markdown("#### 🎯 Key Risk Factors")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        else:
            feature_importance = np.abs(model.coef_[0])
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Map feature names to readable names
        feature_map = {
            'CreditScore': 'Credit Score',
            'Geography': 'Geography',
            'Gender': 'Gender',
            'Age': 'Age',
            'Tenure': 'Tenure',
            'Balance': 'Account Balance',
            'NumOfProducts': 'Number of Products',
            'HasCrCard': 'Has Credit Card',
            'IsActiveMember': 'Active Member Status',
            'EstimatedSalary': 'Estimated Salary'
        }
        
        top_factors = importance_df.head(5)
        
        for idx, row in top_factors.iterrows():
            readable_name = feature_map.get(row['Feature'], row['Feature'])
            st.write(f"• **{readable_name}** - Important factor in churn prediction")
        
        # Action recommendations
        st.markdown("---")
        st.markdown("#### 💡 Recommended Actions")
        
        if churn_probability >= 70:
            st.error("""
            **HIGH RISK - Immediate Action Required:**
            - Schedule a personal call with relationship manager
            - Offer retention incentives (fee waivers, better rates)
            - Review and address any service issues
            - Consider loyalty rewards program enrollment
            """)
        elif churn_probability >= 40:
            st.warning("""
            **MEDIUM RISK - Proactive Engagement:**
            - Send personalized email about new products/services
            - Offer financial planning consultation
            - Check if customer needs are being met
            - Monitor account activity closely
            """)
        else:
            st.success("""
            **LOW RISK - Standard Engagement:**
            - Continue regular service excellence
            - Periodic check-ins to ensure satisfaction
            - Cross-sell opportunities for additional products
            - Maintain positive customer experience
            """)

# TAB 2: MODEL PERFORMANCE
with tab2:
    st.markdown('<p class="sub-header">Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Load and display model results
    try:
        results_df = pd.read_csv('data/model_results.csv')
        
        st.markdown("#### 📈 Model Comparison")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']), 
                     use_container_width=True)
        
        # Display visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Confusion Matrix")
            try:
                img = Image.open('data/confusion_matrix.png')
                st.image(img, use_container_width=True)
            except:
                st.info("Confusion matrix visualization not available")
        
        with col2:
            st.markdown("#### ROC Curve")
            try:
                img = Image.open('data/roc_curve.png')
                st.image(img, use_container_width=True)
            except:
                st.info("ROC curve visualization not available")
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Feature Importance")
            try:
                img = Image.open('data/feature_importance.png')
                st.image(img, use_container_width=True)
            except:
                st.info("Feature importance visualization not available")
        
        with col4:
            st.markdown("#### Model Comparison Chart")
            try:
                img = Image.open('data/model_comparison.png')
                st.image(img, use_container_width=True)
            except:
                st.info("Model comparison chart not available")
        
    except FileNotFoundError:
        st.error("Model results not found. Please run train_model_simple.py first.")

# TAB 3: ABOUT
with tab3:
    st.markdown('<p class="sub-header">About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This application predicts the likelihood of bank customers churning (leaving the bank) 
    using machine learning algorithms. It helps financial institutions identify at-risk 
    customers and take proactive retention measures.
    
    ### 🔍 How It Works
    1. **Data Collection**: The model was trained on historical customer data including 
       demographics, account details, and transaction patterns
    2. **Machine Learning**: We compared multiple algorithms (Logistic Regression, Random Forest)
       to find the best predictor
    3. **Prediction**: Input customer information to get real-time churn probability
    4. **Action**: Receive tailored recommendations based on risk level
    
    ### 📊 Model Performance
    - **Algorithm**: Random Forest / Logistic Regression (best performer selected)
    - **Training Data**: 10,000 customer records
    - **Accuracy**: ~85% (see Model Performance tab for details)
    - **Key Predictors**: Age, account balance, number of products, geography
    
    ### 💼 Business Impact
    **Assumptions:**
    - Average Customer Lifetime Value (LTV): $5,000
    - Retention Campaign Success Rate: 30%
    
    **Potential Value:**
    - By identifying high-risk customers early, banks can implement targeted retention 
      strategies and save significant revenue
    - Proactive engagement improves customer satisfaction and loyalty
    
    ### 🛠️ Technical Stack
    - **Python**: Data processing and modeling
    - **Scikit-learn**: Machine learning algorithms
    - **Streamlit**: Web application framework
    - **Pandas/NumPy**: Data manipulation
    - **Matplotlib/Seaborn**: Visualizations
    
    ### 👤 Created By
    Senior Engagement Manager | McKinsey & Company
    
    **Project Goal**: Demonstrate AI/ML product development capabilities for 
    transition into AI Product Management roles in financial services.
    
    ---
    
    ### 📝 Next Steps (V2.0 Ideas)
    - Integration with CRM systems
    - Real-time model monitoring and retraining
    - A/B testing framework for retention campaigns
    - Customer segmentation and personalized strategies
    - Explainable AI (SHAP values) for individual predictions
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the 'Predict Churn' tab to try different customer profiles and see how predictions change!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>🏦 Bank Customer Churn Predictor | Built with Streamlit & Scikit-learn</p>
    </div>
""", unsafe_allow_html=True)