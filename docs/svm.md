# Support Vector Machine (SVM)

Based on `Code/Day 13 SVM.md`. Linear SVM classification on `Social_Network_Ads.csv`.

## Example
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load
dataset = pd.read_csv('datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train linear SVM
clf = SVC(kernel='linear', random_state=0)
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('\nReport:\n', classification_report(y_test, y_pred))
```

## Notes
- Feature scaling is important for SVMs.
- Swap `kernel` to `rbf`, `poly`, or `sigmoid` to try non-linear boundaries.