# Housing Price Prediction

This project focuses on building a machine learning model to predict housing prices based on various property features. The dataset includes detailed attributes related to housing structure, location, amenities, and sale information. The final goal was to achieve the lowest possible RMSE on the test set to ensure high prediction accuracy.

---

## Project Workflow

### Step 1: Exploratory Data Analysis (EDA)
- Used `pandas` for data manipulation and summary statistics.
- Visualized data using `matplotlib` and `seaborn`.
- Investigated:
  - Distribution of sale prices
  - Missing values
  - Correlations between features
  - Pairplots for top numeric relationships
- Identified and removed significant outliers.

---

### Step 2: Data Preprocessing
- **Missing value treatment**:
  - KNN Imputer for numerical columns
  - Mode imputation for categorical (`object`) columns
  - Filled some missing categorical fields with `"NA"` string
- **Encoding**:
  - `get_dummies` for one-hot encoding
  - `LabelEncoder` for label encoding
- **Feature engineering**:
  - Created new features to enhance model learning
  - Removed highly correlated features to reduce redundancy
- **Standardization**:
  - Applied `StandardScaler` to standardize numeric features
- **Train-Test Split**:
  - Used `train_test_split` to separate data for training and evaluation

---

### Step 3: Model Selection & Evaluation
The main objective was to minimize **Root Mean Squared Error (RMSE)** on the test set.

| Model | RMSE |
|-------|------|
| Linear Regression (baseline) | 22,345.95 |
| Lasso Regression (for feature selection) | - |
| Random Forest (basic) | 11,135.07 |
| Random Forest (with Grid Search) | 10,769.56 |
| XGBoost (Randomized Search) | **3,444.32** ✅ |

#### Best XGBoost Hyperparameters:
```json
{
  "colsample_bytree": 0.856,
  "gamma": 2.372,
  "learning_rate": 0.108,
  "max_depth": 4,
  "n_estimators": 603,
  "reg_alpha": 6.496,
  "reg_lambda": 8.492,
  "subsample": 0.829
}
```

## Dataset Overview

The dataset includes numerous features, such as:
- Property attributes (e.g. OverallQual, GrLivArea, YearBuilt)
- Lot and neighborhood characteristics (LotArea, Neighborhood, MSZoning)
- Interior/exterior details (KitchenQual, RoofStyle, Exterior1st)
- Amenities (GarageCars, Fireplaces, PoolQC)
- Sale metadata (SaleType, SaleCondition)

A full description of all features is provided in data_description.txt.

## Tools & Libraries Used
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Conclusion

The final model, trained using XGBoost and tuned with randomized search, significantly improved prediction accuracy, reducing the RMSE to 3,444, a strong result for a housing price prediction problem.

This model can be further improved by:
- Incorporating ensemble methods
- Tuning feature engineering
- Using cross-validation more extensively
