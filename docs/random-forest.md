# Random Forest Classification

Based on `Code/Day 34 Random_Forest.md`. Ensemble of decision trees for classification.

## Example
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load
dataset = pd.read_csv('datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Optional: scaling not required for tree ensembles
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('\nReport:\n', classification_report(y_test, y_pred))
```

## Tuning
- `n_estimators`: number of trees (e.g., 100+)
- `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`: control bias-variance
- Use cross-validation to select hyperparameters