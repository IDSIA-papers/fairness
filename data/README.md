# Datasets Used in the Fairness Project

This document provides information about the datasets used in this fairness research project and where to find them.

## Dataset Sources

### 1. German Credit Data
- **Source**: [UCI Machine Learning Repository - Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Description**: Credit risk assessment dataset for classification tasks
- **Preprocessed file**: `german_credit_complete.csv`

### 2. Law School Dataset
- **Source**: [Fairness Dataset Repository - Law School](https://github.com/tailequy/fairness_dataset/tree/main/Law_school)
- **Description**: Law school admission and bar passage prediction data
- **Preprocessed file**: `law_bar_pass_prediction.csv`

### 3. OULAD (Open University Learning Analytics Dataset)
- **Source**: [Kaggle - Student Demographics Online Education Data](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad?select=studentInfo.csv)
- **Description**: Online education student demographics and performance data
- **Preprocessed file**: `studentInfo_OULAD.csv`

### 4. Student Performance Datasets
- **Source**: [Kaggle - Student Performance in Mathematics and Portuguese](https://www.kaggle.com/datasets/mrigaankjaswal/student-performance-in-mathematics-and-portuguese)
- **Description**: Student performance in secondary education
- **Preprocessed files**: 
  - `student_mat.csv` (Mathematics)
  - `student_por.csv` (Portuguese)

### 5. Credit Card Default Dataset
- **Source**: [UCI Machine Learning Repository - Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Description**: Credit card default prediction dataset
- **Preprocessed file**: `UCI_Credit_Card.csv`

### 6. Adult (Census Income) Dataset
- **Source**: [UCI Machine Learning Repository - Adult](https://archive.ics.uci.edu/dataset/2/adult)
- **Description**: Census income prediction dataset
- **Preprocessed file**: `adult.csv`

### 7. Bank Marketing Dataset
- **Source**: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Description**: Bank marketing campaign dataset
- **Preprocessed file**: `bank_marketing.csv`

### 8. USA Census Income Data
- **Source**: [Kaggle - USA Census Income Data](https://www.kaggle.com/datasets/manishkc06/usa-census-income-data)
- **Description**: US Census income prediction dataset
- **Preprocessed file**: `Census_income_kdd.csv`

### 9. COMPAS Recidivism Datasets
- **Source**: [ProPublica - COMPAS Recidivism Risk Score Data](https://projects.propublica.org/datastore/#compas-recidivism-risk-score-data-and-analysis)
- **Description**: Criminal recidivism risk assessment data
- **Preprocessed files**:
  - `compas-scores-two-years.csv` (General recidivism)
  - `compas-scores-two-years-violent.csv` (Violent recidivism)

### 10. Communities and Crime Dataset
- **Source**: [UCI Machine Learning Repository - Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime)
- **Description**: Crime prediction based on community attributes
- **Preprocessed file**: `crimedata.csv`

### 11. Diabetes 130-US Hospitals Dataset
- **Source**: [UCI Machine Learning Repository - Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Description**: Diabetes patient hospital readmission prediction
- **Preprocessed file**: `diabetes_130.csv`

### 12. Dutch Census Dataset
- **Source**: [Fairness Dataset Repository - Dutch Census](https://github.com/tailequy/fairness_dataset/tree/main/Dutch_census)
- **Description**: Dutch census data for fairness analysis
- **Preprocessed file**: `dutch_census_2001.csv`

## Usage Notes

1. All datasets have been preprocessed and are available in the `preprocessed_data/` folder
2. The original dataset sources are provided for reference and reproducibility
3. The preprocessed versions are also already standardized and ready to use with the fairness analysis pipeline in this project