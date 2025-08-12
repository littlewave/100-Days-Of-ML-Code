# Multiple Linear Regression

Based on `Code/Day3_Multiple_Linear_Regression.md`. Predict a continuous target from multiple features including a categorical column.

## Example (modern ColumnTransformer)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Load
dataset = pd.read_csv('datasets/50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# One-hot encode the 'State' column (assumed to be named 'State')
categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
], remainder='passthrough')

model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('R^2 on test:', model.score(X_test, y_test))
```

## Notes
- `drop='first'` avoids the dummy variable trap.
- If your column names differ, adjust `categorical_columns` accordingly.