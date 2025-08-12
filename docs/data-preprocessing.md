# Data Preprocessing

Based on `Code/Day 1_Data PreProcessing.md`.

Typical steps:
- Library imports
- Load CSV into a `pandas.DataFrame`
- Split features/target
- Handle missing data
- Encode categorical variables
- Train/test split
- Feature scaling

## Example (modern scikit-learn)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
dataset = pd.read_csv('datasets/Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Identify columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(X_train_processed.shape, X_test_processed.shape)
```

## Tips
- Fit scalers/encoders on the training data only; transform the test data.
- Keep preprocessing in a `Pipeline` to avoid leakage.