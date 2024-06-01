import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import resample
import numpy as np


def bootstrap_confidence_interval(df, treatment_value, outcome_value, confounders, n_bootstrap=500, ci=95):
    estimates = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        df_sample = resample(df)

        # One-hot encoding for categorical data
        df_sample = pd.get_dummies(df_sample, columns=[col for col in confounders if df_sample[col].dtype == 'object'])

        # Create binary treatment and outcome variables
        df_sample['treatment'] = (df_sample['Treatment'] == treatment_value).astype(int)
        df_sample['outcome'] = df_sample['Outcome_Value'].where(df_sample['Outcome'] == outcome_value)
        df_sample = df_sample.dropna(subset=['outcome'])

        # Prepare DoubleML data
        final_confounders = df_sample.columns[df_sample.columns.str.startswith(tuple(confounders))].tolist()
        data_double_ml = DoubleMLData(df_sample, 'outcome', 'treatment', final_confounders)

        # Define and fit the model
        learner_l = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        learner_m = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        ml_model = DoubleMLPLR(data_double_ml, ml_l=learner_l, ml_m=learner_m)
        ml_model.fit()

        # Store the estimate
        estimates.append(ml_model.summary['coef'].values[0])
        # print("Estimated value is : ", estimates)

    # Calculate the confidence intervals
    lower_bound = np.percentile(estimates, (100 - ci) / 2)
    upper_bound = np.percentile(estimates, 100 - (100 - ci) / 2)

    return np.mean(estimates), lower_bound, upper_bound


# Example usage
if __name__ == "__main__":
    df_path = "datasets/df_treatments_outcomes.csv"
    df = pd.read_csv(df_path)
    confounders = ["Location", "Age", "Gender/Race"]

    mean_effect, lower_ci, upper_ci = bootstrap_confidence_interval(df, 'Vegetables', 'Depression', confounders)
    print(f"Estimated Effect: {mean_effect}, 95% CI: ({lower_ci}, {upper_ci})")
