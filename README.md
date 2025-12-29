# 🏦 Bank Customer Churn Predictor

An end-to-end machine learning application that predicts customer churn for banking institutions using Random Forest and Logistic Regression algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🎯 Project Overview

This project demonstrates the complete AI product development lifecycle - from problem framing to deployment. Built to showcase product management and technical capabilities for AI Product Manager roles in financial services.

### Business Problem
Customer acquisition costs 5-7x more than retention. This tool helps banks identify at-risk customers early and implement targeted retention strategies, potentially saving millions in lost revenue.

### Solution
An interactive web application that:
- Predicts churn probability in real-time
- Provides risk-level categorization (Low/Medium/High)
- Offers actionable retention recommendations
- Displays model performance metrics and insights

## 📊 Key Results

- **Model Performance**: 85%+ accuracy with ROC AUC of ~0.86
- **Best Algorithm**: Random Forest (outperformed Logistic Regression baseline)
- **Top Churn Predictors**: Age, Account Balance, Number of Products, Geography, Active Member Status
- **Business Impact**: Potential to save $500K+ annually by retaining just 30% of identified at-risk customers

## 🛠️ Technical Stack

- **Languages**: Python 3.8+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Data**: Kaggle Bank Customer Churn dataset (10,000 records)

## 📁 Project Structure
```
churn-predictor/
├── app.py                      # Streamlit web application
├── eda_and_prep.py            # Exploratory data analysis & preprocessing
├── train_model_simple.py      # Model training & evaluation
├── data/                      # Data files and visualizations
│   ├── Churn_Modelling.csv
│   ├── *.png                  # Generated visualizations
│   └── *.csv                  # Results and metrics
├── models/                    # Trained models and preprocessors
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoder_*.pkl
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone [your-repo-url]
cd churn-predictor
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Running the Application

1. Make sure you're in the project directory with virtual environment activated

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## 💡 Features

### 1. Churn Prediction Interface
- Interactive input form for customer attributes
- Real-time churn probability calculation
- Risk level classification with visual indicators
- Personalized retention recommendations

### 2. Model Performance Dashboard
- Comparative metrics for all trained models
- Confusion matrix visualization
- ROC curve analysis
- Feature importance rankings

### 3. Business Context
- Clear explanation of the business problem
- ROI calculations and impact estimates
- Technical documentation
- Future enhancement roadmap

## 📈 Model Development Process

1. **Data Exploration**: Analyzed 10,000 customer records, identified 20% churn rate
2. **Feature Engineering**: Encoded categorical variables, scaled numerical features
3. **Model Training**: Compared Logistic Regression vs Random Forest
4. **Evaluation**: Selected best model based on ROC AUC score
5. **Deployment**: Packaged into user-friendly web application

## 🎓 Key Learnings & Product Decisions

- **Model Choice**: Random Forest selected over Logistic Regression for better recall (catching more churners)
- **Feature Selection**: Kept all features as they provided interpretable insights
- **User Interface**: Focused on actionable outputs rather than technical details
- **Business Metrics**: Emphasized ROI and practical impact over pure accuracy

## 🔮 Future Enhancements (V2.0)

- [ ] SHAP values for individual prediction explanations
- [ ] Integration with CRM systems via API
- [ ] Automated model retraining pipeline
- [ ] A/B testing framework for retention campaigns
- [ ] Customer segmentation and personalized strategies
- [ ] Real-time monitoring dashboard for model drift

## 📝 Use Cases

1. **Retention Teams**: Identify which customers to prioritize for outreach
2. **Relationship Managers**: Prepare for customer conversations with risk insights
3. **Marketing**: Target retention campaigns to high-risk segments
4. **Product Teams**: Understand which features correlate with retention

## 👨‍💼 About

Created by: [Your Name]  
Role: Senior Engagement Manager, McKinsey & Company  
LinkedIn: [Your LinkedIn]  
Email: [Your Email]

**Project Goal**: Demonstrate end-to-end AI product development capabilities for transition into AI Product Management roles at financial institutions (Morgan Stanley, Citi, Standard Chartered, HSBC).

## 📄 License

This project is for portfolio demonstration purposes.

## 🙏 Acknowledgments

- Dataset: Kaggle Bank Customer Churn Prediction
- Inspired by real-world banking retention challenges
- Built with Streamlit's excellent framework

---

⭐ If you found this project interesting, please star the repository!