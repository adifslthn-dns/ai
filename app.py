# Final Assignment – Regression Project (IBM Coursera)

# 1. Import Libraries
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 2. Dataset Description

# Load dataset
california = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

# Show top rows
df.head()

# Informasi data
df.info()
df.describe()

# 3. Analytic Objective

# 4. Exploratory Data Analysis (EDA)

# Korelasi antar fitur
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap='Reds', annot=False)
plt.title('Correlation Heatmap of California Housing Features')
plt.show()

# Distribusi target
sns.histplot(df['MedHouseVal'], kde=True, bins=30, color='red')
plt.title('Distribution of Median House Value')
plt.show()

# 5. Data Splitting

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# 6. Model Training & Evaluation


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_mse, lin_r2 = evaluate_model(lin_reg, X_test, y_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_mse, ridge_r2 = evaluate_model(ridge, X_test, y_test)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_mse, lasso_r2 = evaluate_model(lasso, X_test, y_test)

# Hasil evaluasi
results = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Lasso'],
    'MSE': [lin_mse, ridge_mse, lasso_mse],
    'R2 Score': [lin_r2, ridge_r2]
})

results

# 7. Findings

plt.figure(figsize=(8, 4))
sns.barplot(x='Model', y='R2 Score', data=results, palette='Reds')
plt.title('Model Performance Comparison')
plt.show()



# Simpan hasil ke file CSV (optional)
results.to_csv('regression_results.csv', index=False)
