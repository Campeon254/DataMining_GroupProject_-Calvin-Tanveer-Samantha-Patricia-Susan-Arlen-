# Lung Cancer Survival Analysis and Prediction
## Data Science Project by Calvin, Tanveer, Samantha, Patricia, Susan and Arlen

## Project Overview
This project analyzes lung cancer patient data to predict survival rates and identify key factors affecting patient outcomes. We developed a machine learning pipeline that processes patient records, identifies risk factors, and provides survival probability predictions through an interactive dashboard.

## Table of Contents
1. [Data Extraction and Transformation](#1-data-extraction-and-transformation)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Data Mining and Modeling](#3-data-mining-and-modeling)
4. [Interactive Dashboard](#4-interactive-dashboard)

## 1. Data Extraction and Transformation
### Initial Data Loading
- Loaded raw dataset containing 890,000 patient records from multiple healthcare facilities
- Implemented two extraction methods:
  - Full extraction: Complete dataset load for initial analysis
  - Incremental extraction: Filtered for records after 2020-01-01 (125,749 records) to focus on recent cases
- Data fields included patient demographics, medical history, treatment details, and survival outcomes

[Screenshot: Initial data loading and shape]

### Data Quality Assessment
- Performed comprehensive data quality checks:
  - Identified and handled missing values in diagnosis dates (0.1%)
  - Removed 23 duplicate patient records
  - Validated data types for each column:
    - Converted dates to datetime format
    - Ensured binary indicators (0/1) for medical conditions
    - Verified categorical encodings for gender, country, etc.
- Generated data quality report showing 99.9% completeness

[Screenshot: Data quality checks]

### Feature Engineering
1. **Temporal Feature Creation**
   - Converted diagnosis_date and end_treatment_date to datetime
   - Calculated treatment_duration in days: end_treatment_date - diagnosis_date
   - Created time-based features:
     - diagnosis_year: Extract year from diagnosis date
     - diagnosis_month: 1-12 month encoding
     - diagnosis_quarter: Q1-Q4 categorization
     - diagnosis_year_month: Combined year-month for time series

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
     - <18: Children
     - 18-29: Young Adults
     - 30-44: Adults
     - 45-59: Middle-aged
     - 60-74: Seniors
     - 75+: Elderly
   - One-hot encoded categorical variables:
     - gender: Male/Female
     - country: 15 unique countries
     - smoking_status: Never/Former/Current
     - treatment_type: Surgery/Chemotherapy/Radiation/Combined

[Screenshot: Feature engineering results]

## 2. Exploratory Data Analysis
### Demographic Analysis
- Age and Gender Distribution:
  - Median age: 58 years
  - Gender split: 62% male, 38% female
  - Highest incidence: 45-59 age group (42%)
- Geographical Analysis:
  - Top 3 countries: France (15%), Germany (13%), UK (12%)
  - Regional variation in treatment approaches
- Smoking Status Impact:
  - Current smokers: 45%
  - Former smokers: 35%
  - Never smoked: 20%
  - Correlation with survival rate: -0.42

[Screenshot: Demographic visualizations]

### Medical Factor Analysis
- Investigated comorbidity patterns
- Analyzed treatment effectiveness by cancer stage
- Studied survival rates across different patient groups

[Screenshot: Medical factor analysis]

### Time-based Analysis
- Tracked diagnosis patterns over time
- Analyzed treatment duration effects
- Studied seasonal variations in diagnoses

[Screenshot: Temporal analysis]

## 3. Data Mining and Modeling
### Model Development
1. **Data Preprocessing**
   - Split data into training (80%) and testing (20%) sets
   - Standardized numerical features
   - Encoded categorical variables

2. **Models Implemented**
   - Logistic Regression
   - Random Forest Classifier
   
3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC analysis
   - Cross-validation results

[Screenshot: Model performance comparison]

### Feature Importance Analysis
- Identified key predictors of survival
- Analyzed feature importance rankings
- Validated findings with domain knowledge

[Screenshot: Feature importance plot]

## 4. Interactive Dashboard
### Survival Analysis Dashboard
- Created interactive visualizations for:
  - Survival rates by demographics
  - Treatment effectiveness
  - Risk factor analysis
  
### Prediction Interface
- Developed real-time survival prediction tool
- Included key patient parameters
- Provided probability-based outcomes

[Screenshot: Dashboard interface]

## Technical Implementation
### Tools and Technologies
- Python 3.x
- Pandas & NumPy for data manipulation
- Scikit-learn for machine learning
- Plotly & Seaborn for visualization
- Jupyter Notebooks for development

### Project Structure
```
project/
├── data/
│   ├── raw/
│   ├── transformed/
│   └── encoded/
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
- Calvin
- Tanveer
- Samantha
- Patricia
- Susan
- Arlen