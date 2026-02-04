import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Missing imports (IMPORTANT)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv(r"C:\Users\THC\Downloads\archive (6)\StudentsPerformance.csv")


# ----------------------------
# Create Final Score
# ----------------------------
df['final_score'] = (
    df['math score'] +
    df['reading score'] +
    df['writing score']
) / 3


# ----------------------------
# Encode Categorical Data
# ----------------------------
le = LabelEncoder()

df['parental level of education'] = le.fit_transform(
    df['parental level of education']
)

df['test preparation course'] = le.fit_transform(
    df['test preparation course']
)


# ----------------------------
# Add Synthetic Features
# ----------------------------
np.random.seed(0)

df['study_hours'] = np.random.randint(1, 6, size=len(df))
df['attendance'] = np.random.randint(70, 100, size=len(df))
df['sleep_hours'] = np.random.randint(5, 9, size=len(df))


# ----------------------------
# Features & Target
# ----------------------------
X = df[
    [
        'study_hours',
        'attendance',
        'parental level of education',
        'test preparation course',
        'sleep_hours'
    ]
]

y = df['final_score']


# ----------------------------
# Handle Missing Values
# ----------------------------
X = X.fillna(X.mean())


# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)


# ----------------------------
# Linear Regression
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ----------------------------
# Evaluation
# ----------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ----------------------------
# Coefficients
# ----------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nCoefficients:")
print(coefficients)


# ----------------------------
# Ridge & Lasso
# ----------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)


# ----------------------------
# Plots
# ----------------------------

# Predicted vs Actual
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Predicted vs Actual Scores")
plt.show()


# Coefficient Plot
plt.figure()
sns.barplot(x='Feature', y='Coefficient', data=coefficients)
plt.xticks(rotation=45)
plt.title("Coefficient Magnitudes")
plt.show()


# Residuals
residuals = y_test - y_pred

plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
