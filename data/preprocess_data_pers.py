import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import KBinsDiscretizer

def read_csv_files(directory):
    f_csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    dataframes = {}
    for file in f_csv_files:
        file_path = os.path.join(directory, file)
        dataframes[file[:-4]] = pd.read_csv(file_path, encoding="latin-1")
    
    return list(dataframes.keys()), dataframes

# Save preprocessed datasets
def save_csv_files(directory, dfs):
    for key in dfs.keys():
        file_path = os.path.join(directory, key + ".csv")
        dfs[key].to_csv(file_path, index=False)

def print_class_imbalance(file_names, dfs):
    """
    Prints the class distribution of target variables for each dataset
    """
    print("\n----- CLASS IMBALANCE FOR TARGET VARIABLES -----")
    for name in file_names:
        df = dfs[name]
        # Find target column (starts with "T_")
        target_cols = [col for col in df.columns if col.startswith("T_")]
        if target_cols:
            target_col = target_cols[0]  # Take the first target column if multiple exist
            print(f"\nDataset: {name}")
            print(f"Target: {target_col}")
            
            # Calculate class distribution as counts and percentages
            value_counts = df[target_col].value_counts()
            total = len(df)
            
            print("Class distribution:")
            for val, count in value_counts.items():
                percentage = (count / total) * 100
                print(f"  {val}: {count} samples ({percentage:.2f}%)")
                
            # Calculate imbalance ratio (majority / minority)
            if len(value_counts) > 1:
                majority = value_counts.max()
                minority = value_counts.min()
                imbalance_ratio = majority / minority
                print(f"Imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")

def preprocess_datasets(file_names, dfs):

    # Preprocess each dataset
    # 1. Adult
    df = dfs[file_names[0]]
    df = df.drop(columns=["fnlwgt", "educational-num"])
    df = df.rename(columns={"income": "T_income", "age": "S1_age", "gender": "S2_gender", "race": "S3_race"})
    df = df.replace("?", "Other")
    dfs[file_names[0]] = df

    # 2. Bank marketing
    df = dfs[file_names[1]]
    df = df.rename(columns={"age": "S1_age", "deposit":"T_deposit", "marital": "S2_marital"})
    df = df.fillna("unknown")
    df = df.drop(columns=["day_of_week"])
    dfs[file_names[1]] = df

    # 3. Census income kdd
    df = dfs[file_names[2]]
    df = df.rename(columns={"sex": "S1_sex", "race": "S2_race", "income_level": "T_income_level"})
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    race2 = []
    for i in df["S2_race"]:
        if i == "White":
            race2.append("White")
        else:
            race2.append("Other")
    df["S2_race"] = race2

    df_col = ["age", "class_of_worker", "industry_code", "occupation_code", "education", "wage_per_hour", 
            "marital_status", "S2_race", "S1_sex", "full_parttime_employment_stat", "capital_gains", 
            "capital_losses", "dividend_from_Stocks", "tax_filer_status", "d_household_family_stat",
            "d_household_summary", "num_person_Worked_employer", "family_members_under_18", "citizenship", 
            "business_or_self_employed", "veterans_benefits", "weeks_worked_in_year", "year", "T_income_level"]
    df = df[df_col]

    cat_col = ["industry_code", "occupation_code"]
    for col in cat_col:
        df[col] = df[col].astype("category")
    dfs[file_names[2]] = df

    # 4. Compas scores two years violent
    df = dfs[file_names[3]]
    df_paper = df[["sex", "age_cat", "race", "juv_fel_count", "juv_misd_count", "juv_other_count", 
                "priors_count", "c_charge_degree", "score_text", "v_score_text", "two_year_recid"]]
    df_paper = df_paper.rename(columns={"two_year_recid": "T_two_year_recid", "race": "S1_race", "sex": "S2_sex"})
    df_paper = df_paper.dropna()
    dfs[file_names[3]] = df_paper

    # 5. Compas scores two years
    df = dfs[file_names[4]]
    df_paper = df[["sex", "age_cat", "race", "juv_fel_count", "juv_misd_count", "juv_other_count", 
                "priors_count", "c_charge_degree", "score_text", "v_score_text", "two_year_recid"]]
    df_paper = df_paper.rename(columns={"two_year_recid": "T_two_year_recid", "race": "S1_race", "sex": "S2_sex"})
    dfs[file_names[4]] = df_paper

    # 6. Crime data
    df = dfs[file_names[5]]
    df_columns = ["racepctblack", "pctWInvInc", "pctWPubAsst", "NumUnderPov", "PctPopUnderPov", "PctUnemployed", 
                "MalePctDivorce", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctKids2Par", "PctYoungKids2Par", 
                "PctTeen2Par", "NumKidsBornNeverMar", "PctKidsBornNeverMar", "PctPersOwnOccup", "HousVacant", 
                "PctHousOwnOcc", "PctVacantBoarded", "NumInShelters", "NumStreet", "ViolentCrimesPerPop"]
    df = df[df_columns]
    df = df.replace("?", np.nan)
    df = df.dropna()
    df["ViolentCrimesPerPop"] = df["ViolentCrimesPerPop"].astype("float64")
    df = df.rename(columns={"ViolentCrimesPerPop": "T_ViolentCrimesPerPop", "racepctblack": "S1_racepctblack"})

    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy="quantile")
    print(df["T_ViolentCrimesPerPop"].median())
    print(discretizer.get_params())
    df["T_ViolentCrimesPerPop"] = discretizer.fit_transform(df[["T_ViolentCrimesPerPop"]]).astype(float)
    print(df["T_ViolentCrimesPerPop"][:20])
    dfs[file_names[5]] = df

    # 7. Diabetes 130
    df = dfs[file_names[6]]
    df_columns = ["race", "gender", "age", "time_in_hospital", "num_procedures", "num_medications", 
                "number_outpatient", "number_emergency", "number_inpatient", "A1Cresult", "metformin", 
                "chlorpropamide", "glipizide", "rosiglitazone", "acarbose", "miglitol", "diabetesMed", "readmitted"]
    df = df[df_columns]
    df = df.rename(columns={"gender": "S1_gender", "readmitted": "T_readmitted"})
    df = df.fillna("Other")
    df = df.replace("None", "Nothing")
    df["T_readmitted"] = df["T_readmitted"].replace({"<30": "YES", ">30": "YES"})
    dfs[file_names[6]] = df

    # 8. Dutch census 2001
    df = dfs[file_names[7]]
    df = df.applymap(lambda x: str(x)[2:-1])
    df = df.rename(columns={"sex": "S1_sex", "occupation": "T_occupation"})
    for col in df.columns:
        df[col] = df[col].astype("category")
    dfs[file_names[7]] = df

    # 9. German credit complete
    df = dfs[file_names[8]]
    df = df.rename(columns={"Creditability": "T_Creditability", "Age (years)": "S1_Age_(years)", 
                            "Sex & Marital Status": "S2_Sex & Marital Status"})
    sex = []
    marital_status = []

    for s in df["S2_Sex & Marital Status"]:
        if s == 1:
            sex.append("Male")
            marital_status.append("Divorced/Separated")
        elif s == 2:
            sex.append("Female")
            marital_status.append("Divorced/Separated/Married")
        elif s == 3:
            sex.append("Male")
            marital_status.append("Single")
        else:
            sex.append("Male")
            marital_status.append("Married/Widowed")

    df["S2_Sex"] = sex
    df["Marital_status"] = marital_status
    df = df.drop(columns=["S2_Sex & Marital Status"])
    dfs[file_names[8]] = df

    # 10. Law bar pass prediction
    df = dfs[file_names[9]]
    df = df.rename(columns={"bar_passed": "T_bar_passed", "sex": "S1_sex", "race": "S2_race"})
    df = df[["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "fulltime", "fam_inc", "S1_sex", "S2_race", "T_bar_passed"]]
    race = []
    for r in df["S2_race"]:
        if r == 7:
            race.append("White")
        else:
            race.append("Other")
    df["S2_race"] = race
    df = df.dropna()
    dfs[file_names[9]] = df

    # 11. Student info OULAD
    df = dfs[file_names[10]]
    df = df.dropna()
    df = df.drop(columns=["id_student"])
    df = df.rename(columns={"final_result": "T_final_result", "gender": "S1_gender"})
    df["T_final_result"] = df["T_final_result"].replace({"Distinction": "Pass", 
                                                    "Withdrawn": "Fail/Withdrawn", 
                                                    "Fail": "Fail/Withdrawn"})
    dfs[file_names[10]] = df

    # 12. Student mat
    df = dfs[file_names[11]]
    target = []
    for i in df["G3"]:
        if i >= 10:
            target.append(1)
        else:
            target.append(0) 
    df["T_grade"] = target
    df = df.rename(columns={"age": "S1_age", "sex": "S2_sex"})
    dfs[file_names[11]] = df

    # 13. Student por
    df = dfs[file_names[12]]
    target = []
    for i in df["G3"]:
        if i >= 10:
            target.append(1)
        else:
            target.append(0) 
    df["T_grade"] = target
    df = df.rename(columns={"age": "S1_age", "sex": "S2_sex"})
    dfs[file_names[12]] = df

    # 14. UCI Credit Card
    df = dfs[file_names[13]]
    df = df.rename(columns={"default.payment.next.month": "T_default.payment.next.month", 
                            "SEX": "S1_SEX", "MARRIAGE": "S2_MARRIAGE", "EDUCATION": "S3_EDUCATION"})
    df = df.drop(columns=["ID"])
    dfs[file_names[13]] = df


if __name__ == "__main__":

    # datasets_names = ['adult', 'bank_marketing', 'Census_income_kdd', 'compas-scores-two-years-violent', 'compas-scores-two-years',
    # 'crimedata', 'diabetes_130', 'dutch_census_2001', 'german_credit_complete', 'law_bar_pass_prediction', 'studentInfo_OULAD', 'student_mat',
    # 'student_por', 'UCI_Credit_Card']

    file_names, dfs = read_csv_files('data')

    preprocess_datasets(file_names, dfs)

    print_class_imbalance(file_names, dfs)

    save_csv_files("preprocessed_data", dfs)
