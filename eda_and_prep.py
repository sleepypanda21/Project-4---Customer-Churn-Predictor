import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
print("Loading data...")
df = pd.read_csv('data/Churn_Modelling.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Basic info
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(f"Total customers: {len(df)}")
print(f"Churned customers: {df['Exited'].sum()} ({df['Exited'].mean()*100:.2f}%)")
print(f"Retained customers: {(1-df['Exited']).sum()} ({(1-df['Exited'].mean())*100:.2f}%)")

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
print(df.isnull().sum())

# Create visualizations
print("\nCreating visualizations...")

# 1. Churn distribution
plt.figure(figsize=(8, 5))
df['Exited'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Exited (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('data/churn_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: churn_distribution.png")

# 2. Age distribution by churn
plt.figure(figsize=(10, 5))
df[df['Exited']==0]['Age'].hist(bins=30, alpha=0.5, label='Retained', color='green')
df[df['Exited']==1]['Age'].hist(bins=30, alpha=0.5, label='Churned', color='red')
plt.title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('data/age_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_distribution.png")

# 3. Geography breakdown
plt.figure(figsize=(10, 5))
geo_churn = pd.crosstab(df['Geography'], df['Exited'], normalize='index') * 100
geo_churn.plot(kind='bar', stacked=False, color=['green', 'red'])
plt.title('Churn Rate by Geography', fontsize=14, fontweight='bold')
plt.xlabel('Geography')
plt.ylabel('Percentage')
plt.legend(['Retained', 'Churned'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/geography_churn.png', dpi=300, bbox_inches='tight')
print("✓ Saved: geography_churn.png")

# 4. Balance vs Churn
plt.figure(figsize=(10, 5))
df.boxplot(column='Balance', by='Exited', grid=False)
plt.title('Account Balance by Churn Status', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
plt.xlabel('Exited (0=No, 1=Yes)')
plt.ylabel('Balance')
plt.tight_layout()
plt.savefig('data/balance_churn.png', dpi=300, bbox_inches='tight')
print("✓ Saved: balance_churn.png")

print("\n" + "="*50)
print("KEY INSIGHTS FROM EDA")
print("="*50)

# Calculate some key statistics
avg_age_churned = df[df['Exited']==1]['Age'].mean()
avg_age_retained = df[df['Exited']==0]['Age'].mean()
print(f"• Average age of churned customers: {avg_age_churned:.1f} years")
print(f"• Average age of retained customers: {avg_age_retained:.1f} years")

churn_by_geo = df.groupby('Geography')['Exited'].mean() * 100
print(f"\n• Churn rate by geography:")
for geo, rate in churn_by_geo.items():
    print(f"  - {geo}: {rate:.2f}%")

churn_by_gender = df.groupby('Gender')['Exited'].mean() * 100
print(f"\n• Churn rate by gender:")
for gender, rate in churn_by_gender.items():
    print(f"  - {gender}: {rate:.2f}%")

# DATA PREPROCESSING
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Drop unnecessary columns
print("Dropping unnecessary columns...")
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
print("Encoding categorical variables...")
label_encoder_geo = LabelEncoder()
label_encoder_gender = LabelEncoder()

df_clean['Geography'] = label_encoder_geo.fit_transform(df_clean['Geography'])
df_clean['Gender'] = label_encoder_gender.fit_transform(df_clean['Gender'])

# Save encoders for later use
with open('models/label_encoder_geo.pkl', 'wb') as f:
    pickle.dump(label_encoder_geo, f)
with open('models/label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)

print("✓ Encoders saved")

# Separate features and target
X = df_clean.drop('Exited', axis=1)
y = df_clean['Exited']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved")

# Save processed data
print("\nSaving processed data...")
np.save('data/X_train.npy', X_train_scaled)
np.save('data/X_test.npy', X_test_scaled)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

# Also save column names for later
with open('data/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✓ Processed data saved")

print("\n" + "="*50)
print("HOUR 1 COMPLETE! ✓")
print("="*50)
print("\nFiles created:")
print("• 4 visualization PNG files in data/")
print("• Encoders and scaler saved in models/")
print("• Processed train/test data saved in data/")
print("\nReady for Hour 2: Model Development!")