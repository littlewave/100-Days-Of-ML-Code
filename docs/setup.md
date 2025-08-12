# Setup

## Requirements
- Python 3.8+
- pip

Install the core scientific stack and scikit-learn:

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn
```

Optional for plots in notebooks:
```bash
pip install jupyter seaborn
```

## Datasets
The examples reference CSV files in `datasets/`:
- `Data.csv`
- `studentscores.csv`
- `50_Startups.csv`
- `Social_Network_Ads.csv`

Place your working directory at the repository root so relative paths resolve correctly.

## Running examples
- Copy any example code block from the docs into a new Python file, e.g., `examples/logistic_regression.py`.
- Ensure the referenced dataset is present under `datasets/`.
- Run it:
```bash
python examples/logistic_regression.py
```

## Notes on scikit-learn versions
The original lesson snippets use older APIs (e.g., `sklearn.cross_validation`, `Imputer`, `OneHotEncoder(categorical_features=...)`). Examples here use current equivalents:
- `from sklearn.model_selection import train_test_split`
- `from sklearn.impute import SimpleImputer`
- `from sklearn.preprocessing import OneHotEncoder`
- `from sklearn.compose import ColumnTransformer`