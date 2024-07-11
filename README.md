# House Prices Prediction Capstone Project

This capstone project aims to develop a machine learning model to predict house prices using a dataset containing various house-related features. The project involves data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Project Phases

### Phase 1: Data Collection and Preparation
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data) or another source.
2. Load the dataset into a Pandas DataFrame.
3. Inspect the dataset for missing values and handle them appropriately.
4. Perform data cleaning to ensure the dataset is ready for analysis.

### Phase 2: Exploratory Data Analysis (EDA)
```python

# Visualize the distribution of the target variable (MEDV)
sns.histplot(df['MEDV'], kde=True, bins=30)
plt.title('Distribution of House Prices (MEDV)')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.show()
```
### Phase 3: Feature Engineering
```python
# Feature engineering
from sklearn.preprocessing import StandardScaler

# Encode categorical variables
df['CHAS'] = df['CHAS'].astype('category')

# Normalize numerical features
numerical_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

### Phase 4: Model Training and Evaluation

###  Phase 5: Model Interpretation and Reporting


### Deliverables
**Code:** Submit the complete code used for data preparation, EDA, feature engineering, model training, and evaluation in a Jupyter Notebook format.

**Report:** Submit a detailed report documenting your approach, findings, and conclusions, including visualizations and a clear explanation of your steps.

**Presentation:** Prepare a brief presentation summarizing your project and key findings.


### Acknowledgments

Dataset from Kaggle

Inspired by the Data Science - Tinyuka Session Capstone Project






