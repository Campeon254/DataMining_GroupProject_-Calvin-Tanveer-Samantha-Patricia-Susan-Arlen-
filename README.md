# Lung Cancer Survival Analysis and Prediction
## Data Science Project by Calvin, Tanveer, Samantha, Patricia, Susan and Arlen

## Project Overview
This project analyzes lung cancer patient data to predict survival rates and identify key factors affecting patient outcomes. We implemented a complete data science pipeline from data extraction to interactive visualization.

## Table of Contents
1. [Data Extraction and Transformation](#1-data-extraction-and-transformation)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Data Mining and Modeling](#3-data-mining-and-modeling)
4. [Interactive Dashboard](#4-interactive-dashboard)

## 1. Data Extraction and Transformation
### Initial Data Loading
- Loaded raw dataset containing 890,000 patient records
- Implemented both full and incremental extraction methods
- Filtered data to include only records from 2020-01-01 onwards (125,749 records)

[Screenshot: Initial data loading and shape]

### Data Cleaning
- Checked and handled missing values
- Removed duplicate records
- Converted data types appropriately
- Verified data quality and consistency

[Screenshot: Data quality checks]

### Feature Engineering
1. **Date-based Features**
   - Converted diagnosis and treatment dates to datetime format
   - Created treatment duration in days
   - Added temporal features (year, month, quarter)

2. **Medical Features**
   - Created comorbidity count from multiple conditions
   - Categorized BMI into clinical groups
   - Binned cholesterol levels into risk categories

3. **Demographic Features**
   - Created age groups for better analysis
   - Encoded categorical variables using one-hot encoding

[Screenshot: Feature engineering results]

## 2. Exploratory Data Analysis
### Demographic Analysis
- Analyzed age and gender distribution
- Explored geographical patterns
- Examined smoking status impact

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