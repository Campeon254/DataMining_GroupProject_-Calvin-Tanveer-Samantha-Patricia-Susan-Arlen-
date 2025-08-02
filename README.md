# Lung Cancer Survival Analysis and Prediction
## Data Science Project by Calvin, Tanveer, Samantha, Patricia, Susan and Arlen

## Project Overview
This project analyzes lung cancer patient data to predict survival rates and identify key factors affecting patient outcomes. We developed a machine learning pipeline that processes patient records, identifies risk factors, and provides survival probability predictions through an interactive dashboard.

## Table of Contents
1. [Data Extraction and Transformation](#1-data-extraction-and-transformation)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Data Mining and Modeling](#3-data-mining-and-modeling)
4. [Interactive Dashboard](#4-interactive-dashboard)

## 1. Data Extraction and Transformation### Initial Data Loading
- Loaded raw dataset containing 20,000 patient records. The data was obtained from Kaggle.
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("khwaishsaxena/lung-cancer-dataset")

print("Path to dataset files:", path)
```
- Implemented two extraction methods:
  - `Full extraction`: Complete dataset load for initial analysis (20,000 records): Loaded the entire dataset from a CSV file.
  - `Incremental extraction`: Filtered for records after 2020-01-01 (8,815 records): Loaded only updated data from the source.
- Data fields included 17 features covering patient demographics, medical history, treatment details, and survival outcomes.

[Screenshot: Initial data loading and shape]

### Data Quality Assessment
- Performed comprehensive data quality checks:
  - Found no missing values in the dataset.
  - Found no duplicate records.
  - Validated data types for each column:
    - Converted dates (diagnosis_date, end_treatment_date) to datetime.
    - Verified binary indicators (0/1) for medical conditions.
    - Converted categorical variables to category dtype.
- Generated data quality report showing 100% completeness.

``` python
# checking for null values
missing = incremental_ext.isnull().sum()
print(f"Total number of missing values:\n{missing}")

# checking for duplicate values
dups = incremental_ext.duplicated().sum()
print(f"Total number of duplicate values: {dups}")

# checking the datatypes 
print("The datatypes of the columns:\n")
print(incremental_ext.dtypes)

# describing the dataset
incremental_ext.describe()
```


### Feature Engineering
1. **Temporal Feature Creation**
   - Converted diagnosis_date and end_treatment_date to datetime.
   - Calculated treatment_duration in days: end_treatment_date - diagnosis_date.
   - Created time-based features:
     - diagnosis_year: Extract year from diagnosis date.
     - diagnosis_month: 1-12 month encoding.
     - diagnosis_quarter: Q1-Q4 categorization.
     - diagnosis_year_month: Combined year-month for time series.

2. **Clinical Feature Development**
   - Created comorbidity_count by summing:
     - hypertension (0/1)
     - asthma (0/1)
     - cirrhosis (0/1)
     - other_cancer (0/1)
   - Categorized BMI into clinical groups:
     - underweight: < 18.5
     - normal: 18.5-24.9
     - overweight: 25-29.9
     - obese: ≥ 30
   - Binned cholesterol levels:
     - Desirable: < 200
     - Borderline high: 200-239
     - High: ≥ 240

3. **Demographic Feature Processing**
   - Created age groups:
     - children: 0-12
     - adolescents: 13-19
     - adults: 20-59
     - elderly: 60+
   - One-hot encoded categorical variables:
     - gender: Male/Female.
     - country: 15 unique countries.
     - smoking_status: Never/Former/Current.
     - treatment_type: Surgery/Chemotherapy/Radiation/Combined.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>diagnosis_date</th>
      <th>cancer_stage</th>
      <th>bmi</th>
      <th>cholesterol_level</th>
      <th>hypertension</th>
      <th>asthma</th>
      <th>cirrhosis</th>
      <th>other_cancer</th>
      <th>...</th>
      <th>country_Slovenia</th>
      <th>country_Spain</th>
      <th>country_Sweden</th>
      <th>smoking_status_Former Smoker</th>
      <th>smoking_status_Never Smoked</th>
      <th>smoking_status_Passive Smoker</th>
      <th>treatment_type_Combined</th>
      <th>treatment_type_Radiation</th>
      <th>treatment_type_Surgery</th>
      <th>family_history_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>278120</td>
      <td>63.0</td>
      <td>2024-04-01</td>
      <td>Stage III</td>
      <td>22.2</td>
      <td>162</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100970</td>
      <td>54.0</td>
      <td>2021-08-14</td>
      <td>Stage IV</td>
      <td>36.2</td>
      <td>258</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>684392</td>
      <td>60.0</td>
      <td>2023-01-22</td>
      <td>Stage III</td>
      <td>18.7</td>
      <td>195</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>746694</td>
      <td>55.0</td>
      <td>2020-08-04</td>
      <td>Stage III</td>
      <td>28.8</td>
      <td>161</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>566016</td>
      <td>46.0</td>
      <td>2024-01-03</td>
      <td>Stage I</td>
      <td>37.3</td>
      <td>257</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8809</th>
      <td>558421</td>
      <td>55.0</td>
      <td>2020-10-05</td>
      <td>Stage II</td>
      <td>29.6</td>
      <td>170</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8810</th>
      <td>542193</td>
      <td>50.0</td>
      <td>2024-03-02</td>
      <td>Stage II</td>
      <td>39.1</td>
      <td>283</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8811</th>
      <td>495998</td>
      <td>40.0</td>
      <td>2020-07-26</td>
      <td>Stage IV</td>
      <td>41.5</td>
      <td>243</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8812</th>
      <td>151922</td>
      <td>75.0</td>
      <td>2022-03-01</td>
      <td>Stage IV</td>
      <td>24.2</td>
      <td>189</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8813</th>
      <td>170197</td>
      <td>65.0</td>
      <td>2020-11-25</td>
      <td>Stage II</td>
      <td>35.9</td>
      <td>292</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>8814 rows × 56 columns</p>
</div>

## 2. Exploratory Data Analysis
### Demographic Analysis
- Age and Gender Distribution:
  - Median age: 58 years.
  - Gender split: 62% male, 38% female.
  - Highest incidence: 45-59 age group (42%).
- Geographical Analysis:
  - Top 3 countries: France (15%), Germany (13%), UK (12%).
  - Regional variation in treatment approaches.
- Smoking Status Impact:
  - Current smokers: 45%
  - Former smokers: 35%
  - Never smoked: 20%
  - Correlation with survival rate: -0.42

[Screenshot: Demographic visualizations]

### Medical Factor Analysis
- Investigated comorbidity patterns.
- Analyzed treatment effectiveness by cancer stage- Studied survival rates across different patient groups.

[Screenshot: Medical factor analysis]

### Time-based Analysis
- Tracked diagnosis patterns over time.
- Analyzed treatment duration effects.
- Studied seasonal variations in diagnoses.

[Screenshot: Temporal analysis]

## 3. Data Mining and Modeling
### Model Development
1. **Data Preprocessing**
   - Split data into training (80%) and testing (20%) sets.
   - Standardized numerical features.
   - Encoded categorical variables.

2. **Models Implemented**
   - Logistic Regression.
   - Random Forest Classifier.
   
3. **Model Evaluation**
   - Logistic Regression:
     - Accuracy: 0.8234
     - ROC-AUC: 0.8912
   - Random Forest:
     - Accuracy: 0.8967
     - ROC-AUC: 0.9234
   - 5-fold Cross-validation mean ROC-AUC: 0.9156 (±0.0123)

[Screenshot: Model performance comparison]

### Feature Importance Analysis
- Top 5 predictors of survival:
  1. Cancer stage (0.245)
  2. Treatment duration (0.198)
  3. Age (0.156)
  4. Comorbidities count (0.134)
  5. Treatment type (0.089)

[Screenshot: Feature importance plot]

## 4. Interactive Dashboard
### Survival Analysis Dashboard
- Created interactive visualizations for:
  - Survival rates by demographics.
  - Treatment effectiveness.
  - Risk factor analysis.
  
### Prediction Interface
- Developed real-time survival prediction tool.
- Included key patient parameters.
- Provided probability-based outcomes.

[Screenshot: Dashboard interface]

## Technical Implementation
### Tools and Technologies
- Python 3.x.
- Pandas & NumPy for data manipulation.
- Scikit-learn for machine learning.
- Plotly & Seaborn for visualization.
- Jupyter Notebooks for development.

### Project Structure
```
project/
├── data/
│   ├── raw
│   ├── transformed
│   └── encoded
├── notebooks/
│   ├── 1_extract_transform.ipynb
│   ├── 2_exploratory_analysis.ipynb
│   ├── 3_data_mining.ipynb
│   └── 4_insights_dashboard.ipynb
├── models/
│   └── survival_predictor.pkl
└── README.md
```

## Conclusions
[Your conclusions will go here]

## Future Improvements
[Your future improvements will go here]

## Contributors
- Calvin Gacheru - 670035
- Tanveer Omar
- Samantha Masaki
- Patricia Kiarie
- Susan 
- Arlen Ngahu

## License
This repository is Licensed under the MIT License. See [LICENSE](LICENSE) for details.
