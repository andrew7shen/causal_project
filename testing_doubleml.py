import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)


def apply_double_ml(df, treatment_value, outcome_value, confounders):
    print(df.shape)
    # Encode "Location" categorically when it is detected as a confounder
    df = pd.get_dummies(df, columns=[col for col in confounders if df[col].dtype == 'object'])

    # Create binary treatment indicator and a column for the specific outcome
    df['treatment'] = (df['Treatment'] == treatment_value).astype(int)
    df['outcome'] = df['Outcome_Value'].where(df['Outcome'] == outcome_value)

    # Drop NaNs in 'outcome' which may have been introduced when filtering
    df = df.dropna(subset=['outcome'])

    final_confounders = df.columns[df.columns.str.startswith(tuple(confounders))].tolist()
    print("Printing nulls")
    print(sum(df.isna().sum()))

    # Prepare the data for DoubleML
    data_double_ml = DoubleMLData(df, 'outcome', 'treatment', final_confounders)

    # Initialize the DoubleMLPLR model with random forests as ML methods
    learner_l = RandomForestRegressor(n_estimators=100)  # model for the outcome
    learner_m = RandomForestClassifier(n_estimators=100)  # model for the treatment
    ml_model = DoubleMLPLR(data_double_ml, ml_l=learner_l, ml_m=learner_m)

    # Fit the model
    ml_model.fit()
    print("Model fitted")
    # Print the estimated causal effects
    summary_df = ml_model.summary
    print("Estimated causal effect of", treatment_value, "on", outcome_value, ":", summary_df['coef'].values[0])


if __name__ == "__main__":
    df_path = "datasets/df_treatments_outcomes.csv"
    df = pd.read_csv(df_path)
    # print(df.head())
    # Defining the confounding variables
    confounders = ["Location", "Age", "Gender/Race"]
    print("Printing the treatment variables")
    print(df["Treatment"].unique())
    print("Printing the outcome variables ")
    print(df["Outcome"].unique())
    # Checking the effect of vegetables on depression
    # apply_double_ml(df, 'Vegetables', 'Depression', confounders)
    apply_double_ml(df, 'Vegetables', 'Distress', confounders)
    apply_double_ml(df, 'Vegetables', 'Depression', confounders)
    apply_double_ml(df, 'Sleep', 'Depression', confounders)
    apply_double_ml(df, 'Sleep', 'Distress', confounders)
    print("****Completed****")

