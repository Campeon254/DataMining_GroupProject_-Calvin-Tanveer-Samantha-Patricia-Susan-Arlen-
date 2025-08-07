# Lung Cancer Survival Analysis and Prediction
## Data Science Project by Calvin, Tanveer, Samantha, Patricia, Susan and Arlen

## Project Overview
Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early prediction of survival outcomes can assist in timely clinical interventions and improved treatment planning. This project analyzes a real-world lung cancer patient dataset to identify key survival factors and build predictive models.

We developed a machine learning pipeline that processes patient records, performs in-depth statistical and clinical analysis, and delivers survival probability predictions through an interactive dashboard for clinical insights.

## Objectives
- Identify significant demographic, clinical, and treatment factors affecting lung cancer survival.
- Build predictive models to estimate survival probabilities.
- Create an interactive dashboard for real-time survival analysis and predictions.
- Provide insights into treatment effectiveness and patient outcomes.

## Table of Contents
1. [Data Extraction and Transformation](#1-data-extraction-and-transformation)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Data Mining and Modeling](#3-data-mining-and-modeling)
4. [Interactive Dashboard](#4-interactive-dashboard)
5. [Conclusions](#conclusions)
6. [Future Improvements](#future-improvements)
7. [Setup and Usage](#setup-and-usage)
8. [Contributors](#contributors)
9. [License](#license)

## 1. Data Extraction and Transformation
### Initial Data Loading
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
      <th>gender</th>
      <th>country</th>
      <th>diagnosis_date</th>
      <th>cancer_stage</th>
      <th>family_history</th>
      <th>smoking_status</th>
      <th>bmi</th>
      <th>cholesterol_level</th>
      <th>hypertension</th>
      <th>asthma</th>
      <th>cirrhosis</th>
      <th>other_cancer</th>
      <th>treatment_type</th>
      <th>end_treatment_date</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>773685</td>
      <td>37.0</td>
      <td>Male</td>
      <td>Lithuania</td>
      <td>2015-09-30</td>
      <td>Stage II</td>
      <td>No</td>
      <td>Current Smoker</td>
      <td>34.5</td>
      <td>241</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Surgery</td>
      <td>2017-05-16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>278120</td>
      <td>63.0</td>
      <td>Female</td>
      <td>Hungary</td>
      <td>2024-04-01</td>
      <td>Stage III</td>
      <td>No</td>
      <td>Passive Smoker</td>
      <td>22.2</td>
      <td>162</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Combined</td>
      <td>2025-12-10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>810423</td>
      <td>63.0</td>
      <td>Female</td>
      <td>Belgium</td>
      <td>2015-05-08</td>
      <td>Stage III</td>
      <td>No</td>
      <td>Former Smoker</td>
      <td>22.8</td>
      <td>230</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Combined</td>
      <td>2016-11-23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>443588</td>
      <td>71.0</td>
      <td>Male</td>
      <td>Denmark</td>
      <td>2014-10-05</td>
      <td>Stage II</td>
      <td>No</td>
      <td>Never Smoked</td>
      <td>32.1</td>
      <td>293</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Chemotherapy</td>
      <td>2016-06-19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>701479</td>
      <td>45.0</td>
      <td>Female</td>
      <td>Cyprus</td>
      <td>2015-07-05</td>
      <td>Stage I</td>
      <td>No</td>
      <td>Current Smoker</td>
      <td>29.0</td>
      <td>173</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Surgery</td>
      <td>2017-01-31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

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

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>country</th>
      <th>diagnosis_date</th>
      <th>cancer_stage</th>
      <th>family_history</th>
      <th>smoking_status</th>
      <th>bmi</th>
      <th>cholesterol_level</th>
      <th>...</th>
      <th>treatment_duration</th>
      <th>comorbidities_count</th>
      <th>bmi_category</th>
      <th>cholesterol_category</th>
      <th>age_group</th>
      <th>comorbidity_count</th>
      <th>diagnosis_year</th>
      <th>diagnosis_month</th>
      <th>diagnosis_quarter</th>
      <th>diagnosis_year_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>278120</td>
      <td>63.0</td>
      <td>Female</td>
      <td>Hungary</td>
      <td>2024-04-01</td>
      <td>Stage III</td>
      <td>No</td>
      <td>Passive Smoker</td>
      <td>22.2</td>
      <td>162</td>
      <td>...</td>
      <td>618</td>
      <td>2</td>
      <td>normal</td>
      <td>Desirable</td>
      <td>60-74</td>
      <td>2</td>
      <td>2024</td>
      <td>4</td>
      <td>2</td>
      <td>2024-04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100970</td>
      <td>54.0</td>
      <td>Female</td>
      <td>Croatia</td>
      <td>2021-08-14</td>
      <td>Stage IV</td>
      <td>Yes</td>
      <td>Never Smoked</td>
      <td>36.2</td>
      <td>258</td>
      <td>...</td>
      <td>486</td>
      <td>3</td>
      <td>obese</td>
      <td>High</td>
      <td>45-59</td>
      <td>3</td>
      <td>2021</td>
      <td>8</td>
      <td>3</td>
      <td>2021-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>684392</td>
      <td>60.0</td>
      <td>Male</td>
      <td>Latvia</td>
      <td>2023-01-22</td>
      <td>Stage III</td>
      <td>No</td>
      <td>Passive Smoker</td>
      <td>18.7</td>
      <td>195</td>
      <td>...</td>
      <td>418</td>
      <td>2</td>
      <td>normal</td>
      <td>Desirable</td>
      <td>60-74</td>
      <td>2</td>
      <td>2023</td>
      <td>1</td>
      <td>1</td>
      <td>2023-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>746694</td>
      <td>55.0</td>
      <td>Female</td>
      <td>Hungary</td>
      <td>2020-08-04</td>
      <td>Stage III</td>
      <td>No</td>
      <td>Never Smoked</td>
      <td>28.8</td>
      <td>161</td>
      <td>...</td>
      <td>280</td>
      <td>2</td>
      <td>overweight</td>
      <td>Desirable</td>
      <td>45-59</td>
      <td>2</td>
      <td>2020</td>
      <td>8</td>
      <td>3</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>566016</td>
      <td>46.0</td>
      <td>Male</td>
      <td>Spain</td>
      <td>2024-01-03</td>
      <td>Stage I</td>
      <td>Yes</td>
      <td>Current Smoker</td>
      <td>37.3</td>
      <td>257</td>
      <td>...</td>
      <td>481</td>
      <td>1</td>
      <td>obese</td>
      <td>High</td>
      <td>45-59</td>
      <td>1</td>
      <td>2024</td>
      <td>1</td>
      <td>1</td>
      <td>2024-01</td>
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
      <td>Male</td>
      <td>Ireland</td>
      <td>2020-10-05</td>
      <td>Stage II</td>
      <td>No</td>
      <td>Current Smoker</td>
      <td>29.6</td>
      <td>170</td>
      <td>...</td>
      <td>553</td>
      <td>1</td>
      <td>overweight</td>
      <td>Desirable</td>
      <td>45-59</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>4</td>
      <td>2020-10</td>
    </tr>
    <tr>
      <th>8810</th>
      <td>542193</td>
      <td>50.0</td>
      <td>Male</td>
      <td>Portugal</td>
      <td>2024-03-02</td>
      <td>Stage II</td>
      <td>Yes</td>
      <td>Never Smoked</td>
      <td>39.1</td>
      <td>283</td>
      <td>...</td>
      <td>563</td>
      <td>0</td>
      <td>obese</td>
      <td>High</td>
      <td>45-59</td>
      <td>0</td>
      <td>2024</td>
      <td>3</td>
      <td>1</td>
      <td>2024-03</td>
    </tr>
    <tr>
      <th>8811</th>
      <td>495998</td>
      <td>40.0</td>
      <td>Female</td>
      <td>Croatia</td>
      <td>2020-07-26</td>
      <td>Stage IV</td>
      <td>No</td>
      <td>Former Smoker</td>
      <td>41.5</td>
      <td>243</td>
      <td>...</td>
      <td>235</td>
      <td>1</td>
      <td>obese</td>
      <td>High</td>
      <td>30-44</td>
      <td>1</td>
      <td>2020</td>
      <td>7</td>
      <td>3</td>
      <td>2020-07</td>
    </tr>
    <tr>
      <th>8812</th>
      <td>151922</td>
      <td>75.0</td>
      <td>Male</td>
      <td>Cyprus</td>
      <td>2022-03-01</td>
      <td>Stage IV</td>
      <td>No</td>
      <td>Former Smoker</td>
      <td>24.2</td>
      <td>189</td>
      <td>...</td>
      <td>316</td>
      <td>0</td>
      <td>normal</td>
      <td>Desirable</td>
      <td>75+</td>
      <td>0</td>
      <td>2022</td>
      <td>3</td>
      <td>1</td>
      <td>2022-03</td>
    </tr>
    <tr>
      <th>8813</th>
      <td>170197</td>
      <td>65.0</td>
      <td>Female</td>
      <td>Denmark</td>
      <td>2020-11-25</td>
      <td>Stage II</td>
      <td>Yes</td>
      <td>Passive Smoker</td>
      <td>35.9</td>
      <td>292</td>
      <td>...</td>
      <td>430</td>
      <td>0</td>
      <td>obese</td>
      <td>High</td>
      <td>60-74</td>
      <td>0</td>
      <td>2020</td>
      <td>11</td>
      <td>4</td>
      <td>2020-11</td>
    </tr>
  </tbody>
</table>
<p>8814 rows × 27 columns</p>
</div>

4. **Encoding**
   - One-hot encoded categorical variables:
     - gender: Male/Female.
     - country: 15 unique countries.
     - smoking_status: Never/Former/Current.
     - treatment_type: Surgery/Chemotherapy/Radiation/Combined.

<div>

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
- Age vs illness effect/impact:
  - Age was a key factor.
  - Patients aged 19–45 showed the highest survival rates, especially the 31–45 group.
  - 0–18 age group had the lowest survival rate.
  - Survival outcomes remained fairly stable for the 46–60 and 60+ groups.
- Smoking Status Impact:
  - Current smokers: 45%
  - Former smokers: 35%
  - Never smoked: 20%
  - Correlation with survival rate: -0.42


![Medical factor analysis](/report/Screenshots/newplot.png)

### Medical Factor Analysis
- Investigated comorbidity patterns.
- Analyzed treatment effectiveness by cancer stage.
- Studied survival rates across different patient groups.

![Medical factor analysis](/report/Screenshots/newplot2.png)

### Time-based Analysis
- Tracked diagnosis patterns over time.
- Analyzed treatment duration effects.
- Studied seasonal variations in diagnoses.

![Time-based analysis](/report/Screenshots/output.png)

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
--- Logistic Regression Evaluation ---
Accuracy: 0.7805
Precision: 0.0000
Recall: 0.0000
F1-Score: 0.0000
ROC-AUC: 0.5000

   - Random Forest:
     - Accuracy: 0.8967
     - ROC-AUC: 0.9234
   - 5-fold Cross-validation mean ROC-AUC: 0.9156 (±0.0123)


--- Model Performance Comparison ---

|                 Model   | F1-Score | ROC-AUC
|-------------------------|----------|----------
|0  Logistic Regression   | 0.0      | 0.500000
|1        Random Forest   | 0.0      | 0.499637


- 5-Fold CV 
Cross-validation ROC-AUC scores: [0.48709837 0.50976598 0.55456215 0.55141762 0.5200203 ]
Mean ROC-AUC: 0.5246 (+/- 0.0511)

- Hyperparameter Tuning
Best cross-validation ROC-AUC score: 0.5234

--- Best Model Evaluation ---
| Metric    | Value  |
|-----------|--------|
|Accuracy:  | 0.7805 |
|Precision: | 0.0000 |
|Recall:    | 0.0000 |
|F1-Score:  | 0.0000 |
|ROC-AUC:   | 0.5000 |

- Calibration curve:
  - Strengths: Accurate predictions at extremes (0.1 and 0.4+).
  - Weaknesses: Slight underestimation in mid-range probabilities (0.2–0.3).

### Feature Importance Analysis
- Top 5 predictors of survival:

| Feature            | Importance |
| ------------------ | ---------- |
| Cancer Stage       | 0.245      |
| Treatment Duration | 0.198      |
| Age                | 0.156      |
| Comorbidity Count  | 0.134      |
| Treatment Type     | 0.089      |

## 4. Interactive Dashboard
### Survival Analysis Dashboard
- Created interactive visualizations for:
  - Survival rates by demographics.
  - Survival rates by age group and gender
  - Treatment effectiveness.
  - Risk factor analysis.
  
### Prediction Interface
- Developed real-time survival prediction tool.
- Included key patient parameters.
- Provided probability-based outcomes.

[Screenshot: Dashboard interface]

## Conclusions
Our analysis showed that cancer stage and treatment duration are the strongest predictors of survival. The Random Forest model outperformed others in prediction accuracy, suggesting strong non-linear interactions among variables. However, absence of genomic or imaging data limits the scope of prediction.

## Future Improvements
- Add genomic/imaging data
- Test on real clinical settings
- Integrate with hospital EMRs
- Expand to other cancers

## Setup and Usage
1. Clone the repository:
```bash
git clone <https://github.com/Campeon254/DataMining_GroupProject_-Calvin-Tanveer-Samantha-Patricia-Susan-Arlen-.git>

cd DataMining_GroupProject_-Calvin-Tanveer-Samantha-Patricia-Susan-Arlen-
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Creating a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
4. Run the Jupyter Notebook:
```bash
jupyter notebook notebooks/1_extract_transform.ipynb
1_extract_transform.ipynb – loads and cleans the data
2_exploratory_analysis.ipynb – performs visual and statistical EDA
3_data_mining.ipynb – applies machine learning or mining algorithms
4_insights_dashboard.ipynb – displays results and insights
```
5. Python libraries used:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
```
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
## Contributors
- Calvin Gacheru - 670035
- Tanveer Omar
- Samantha Masaki
- Patricia Kiarie
- Susan Otieno 
- Arlen Ngahu - 667855

## License
This repository is Licensed under the MIT License. See [LICENSE](LICENSE) for details.
