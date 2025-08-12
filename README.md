# Multiple_Linear_Regression_Advertising_Dataset
This project applies Multiple Linear Regression on the Advertising dataset to predict product sales based on budgets for TV, Radio, and Newspaper ads. Analysis reveals that TV and Radio significantly boost sales, while Newspaper has minimal impact, achieving an R² of 0.9025912899684558 for accurate predictions.
# 📊 Multiple Linear Regression on Advertising Dataset

## 📌 Overview
This project demonstrates the application of **Multiple Linear Regression** using the `advertising.csv` dataset.  
The dataset contains sales data along with advertising budgets across three media channels — **TV**, **Radio**, and **Newspaper**.

We aim to:
- Understand how advertising budgets affect sales.
- Build a predictive model for sales.
- Evaluate the model’s performance.

---

## 📂 Dataset Details

**File:** `advertising.csv`

| Column Name  | Description |
|--------------|-------------|
| `TV`         | Advertising budget spent on TV (in thousands of dollars) |
| `Radio`      | Advertising budget spent on Radio (in thousands of dollars) |
| `Newspaper`  | Advertising budget spent on Newspaper (in thousands of dollars) |
| `Sales`      | Units sold (in thousands) |

---

## 🛠 Requirements

Install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
🚀 Steps to Run
1️⃣ Import Libraries & Load Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("advertising.csv")
print(df.head())
2️⃣ Exploratory Data Analysis (EDA)

# Pairplot for relationships
sns.pairplot(df)
plt.show()

# Correlation matrix
print(df.corr())
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
3️⃣ Define Features & Target

X = df[['TV', 'Radio', 'Newspaper']]  # Independent variables
y = df['Sales']  # Dependent variable
4️⃣ Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
5️⃣ Train the Model

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
6️⃣ Model Evaluation

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
7️⃣ Statistical Analysis with statsmodels

X_sm = sm.add_constant(X)  # Add intercept term
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())
📈 Example Output
Intercept: 2.94

Coefficients:

TV: 0.0458

Radio: 0.1885

Newspaper: -0.0010

R² Score: 0.9025912899684558
🧠 Interpretation
TV and Radio advertising significantly increase sales.

Newspaper advertising shows negligible or negative impact.






