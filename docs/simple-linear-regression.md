# Simple Linear Regression

Based on `Code/Day2_Simple_Linear_Regression.md`. Predict a continuous target from a single feature.

## Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv('datasets/studentscores.csv')
X = dataset.iloc[:, :1].values  # study hours
y = dataset.iloc[:, 1].values   # scores

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)
print('Predictions:', y_pred)

# Plot
plt.scatter(X_train, y_train, color='red', label='Train')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Fit')
plt.scatter(X_test, y_test, color='green', label='Test')
plt.legend()
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title('Simple Linear Regression')
plt.show()
```

## Inputs/Outputs
- Input: `X` shape (n_samples, 1)
- Target: `y` shape (n_samples,)
- Output: predictions `y_pred` shape (n_samples,)