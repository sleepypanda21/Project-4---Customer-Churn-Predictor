import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
import streamlit

print("✓ All libraries imported successfully!")
print(f"✓ Pandas version: {pd.__version__}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Scikit-learn version: {sklearn.__version__}")

# Try to load data
try:
    df = pd.read_csv('data/Churn_Modelling.csv')
    print(f"✓ Data loaded successfully! Shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns[:5])}...")  # Show first 5 columns
    print(f"✓ Churn rate: {df['Exited'].value_counts(normalize=True)[1]:.2%}")
except FileNotFoundError:
    print("✗ Could not find data/Churn_Modelling.csv")
    print("  Make sure the CSV file is in the 'data' folder")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    
