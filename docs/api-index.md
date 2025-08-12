# API Index (Components used in examples)

This repository demonstrates classic ML workflows using scikit-learn. Below are the primary public classes and functions used across examples, with typical parameters and references.

- **Data split**: `train_test_split(X, y, test_size=..., random_state=...)`  
  Source: `sklearn.model_selection.train_test_split`  
  See: [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

- **Imputation**: `SimpleImputer(strategy='mean'|'most_frequent'|...)`  
  Source: `sklearn.impute.SimpleImputer`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

- **Encoding**: `OneHotEncoder(drop=None|'first', handle_unknown='ignore')`  
  Source: `sklearn.preprocessing.OneHotEncoder`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

- **Scaling**: `StandardScaler()`  
  Source: `sklearn.preprocessing.StandardScaler`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

- **Composition**: `ColumnTransformer(transformers=[...], remainder='passthrough')`  
  Source: `sklearn.compose.ColumnTransformer`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)

- **Pipelines**: `Pipeline(steps=[('step', transformer_or_estimator), ...])`  
  Source: `sklearn.pipeline.Pipeline`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

- **Linear Regression**: `LinearRegression()` with `.fit(X, y)` and `.predict(X)`  
  Source: `sklearn.linear_model.LinearRegression`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

- **Logistic Regression**: `LogisticRegression(solver='liblinear', random_state=...)`  
  Source: `sklearn.linear_model.LogisticRegression`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

- **K-NN Classifier**: `KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)`  
  Source: `sklearn.neighbors.KNeighborsClassifier`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

- **SVM Classifier**: `SVC(kernel='linear'|'rbf'|...)`  
  Source: `sklearn.svm.SVC`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

- **Decision Tree Classifier**: `DecisionTreeClassifier(criterion='gini'|'entropy', random_state=...)`  
  Source: `sklearn.tree.DecisionTreeClassifier`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

- **Random Forest Classifier**: `RandomForestClassifier(n_estimators=100, random_state=...)`  
  Source: `sklearn.ensemble.RandomForestClassifier`  
  See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

- **Metrics**:
  - `confusion_matrix(y_true, y_pred)` → `sklearn.metrics.confusion_matrix`  
    See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
  - `classification_report(y_true, y_pred)` → `sklearn.metrics.classification_report`  
    See: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)