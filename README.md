# 679-Final-project
# 🦟 DengAI: Predicting Dengue Fever Cases Using Traditional Machine Learning

This project applies regression-based machine learning methods to forecast weekly Dengue Fever case counts in the cities of San Juan (SJ) and Iquitos (IQ), using historical climate and epidemiological data provided by DrivenData.

## 📘 Overview

The notebook performs a full end-to-end workflow:
- Data loading and preprocessing
- Exploratory data analysis (EDA)
- Feature selection via statistical and model-based approaches
- Model building using linear regression, Lasso, Random Forest, and XGBoost
- Model diagnostics and performance evaluation

## 📁 Project Structure

```
📦 DengAI-ML-Prediction
├── Newreport.ipynb          # Main Jupyter Notebook (full analysis and modeling)
├── README.md                # Project overview and instructions
├── requirements.txt         # List of required Python libraries
└── data/                    # Optional directory for storing input CSV files
```

## 🛠 Methods

### ✅ Preprocessing
- Merged features and labels on city, year, and week
- Filled missing values using forward fill (`ffill`) by city group
- Dropped non-informative columns (e.g. `week_start_date`)
- Standardized numeric features

### ✅ Feature Selection
- Correlation analysis to remove redundant features
- Model-based ranking via:
  - `LassoCV` (scikit-learn)
  - `RandomForestRegressor`
  - `XGBRegressor`

### ✅ Modeling
Separate models built for each city using:
- Linear regression (`statsmodels`)
- Lasso regression
- Random forest
- XGBoost

Performance metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 📊 Sample Evaluation Output

| City | Model           | MAE   | RMSE  |
|------|------------------|-------|-------|
| SJ   | Random Forest    | ~12–18| ~19–25|
| IQ   | XGBoost          | ~7–10 | ~11–14|

> *Results are illustrative and vary depending on selected features and random seed.*

Deep Learning Extension (Optional)
In addition to traditional models, we experimented with neural network-based approaches using recurrent architectures:

Simple LSTM
A single-layer LSTM model was trained independently on San Juan (SJ) and Iquitos (IQ) data. It uses a rolling window of 12 weeks of historical features to predict dengue case counts.

City	Model	MAE	RMSE
SJ	Simple LSTM	~13–16	~20–22
IQ	Simple LSTM	~8–10	~12–14

Bidirectional LSTM
A Bidirectional LSTM layer was used to capture both past and future context from the input window, potentially enhancing temporal learning.

City	Model	MAE	RMSE
SJ	Bidirectional LSTM	~12–15	~18–21
IQ	Bidirectional LSTM	~7–9	~11–13

Deep learning models were implemented in Keras using TensorFlow backend and trained on standardized, forward-filled feature windows of size 12.

## 📈 Visualizations
- Correlation heatmaps
- Feature importance plots
- Regression and residual plots using `statsmodels`

## 📦 Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- statsmodels

## 🔧 Installation

```bash
git clone https://github.com/cyy200107/679-Final-project.git
cd DengAI-ML-Prediction
pip install -r requirements.txt
```

## 👨‍⚕️ Author
**Beiming Yu**
Email: yibeimin@bu.edu

**Yangyu Chen**  
Email: yangyu07@bu.edu

---

This project does not use deep learning. All models are based on traditional regression and ensemble methods.
