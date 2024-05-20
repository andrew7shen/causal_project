import numpy as np
import statsmodels.formula.api as smf
from statistics import mean
import pandas as pd

np.random.seed(42)


def backdoor(df, confounders=["Location", "Age", "Gender/Race"]):
    """
    confounders:
        Location
        Age
        Gender/Race
    """
    print("Running backdoor with synthetic data of size ", df.shape)
    # one-hot encoding for categorical values
    df = pd.get_dummies(df, columns=confounders, drop_first=True)
    # print(df.columns)
    results = []

    # Model with confounders
    ols = "Outcome_Value ~ Treatment_Value"
    for cf in df.columns:
        if cf.startswith(tuple(confounders)):  # This will let the model include all new-confounder columns
            ols += " + Q('%s')" % cf  # For '/' in Gender/Race column
    model = smf.ols(ols, data=df).fit()
    # print(model.summary()) #Maybe there's some interesting thing we can do with p, t values. R squared value looks good now.

    # Calculate backdoor estimate for each value of "Treatment_Value"
    # 259 unique values for "Treatment_Value"
    a_vals_binary = [0.0, 100.0]
    for a in sorted(a_vals_binary):
        # Calculate predicted values using the model after assigning value of "Treatment_Value"
        curr_df = df.copy()
        curr_df["Treatment_Value"] = a
        curr_estimate_list = model.predict(curr_df)

        # Calculate mean of estimate list
        curr_estimate = mean(curr_estimate_list)
        results.append(curr_estimate)

    return np.array(results)


def gather_data(df, treatment, outcome):
    # Read in dataset
    # df = pd.read_csv(df_path)

    # Check the column names
    # print("DataFrame columns:", df.columns)

    # Filter for just Ai and Yi
    return df[(df["Treatment"] == treatment) & (df["Outcome"] == outcome)]


if __name__ == "__main__":
    print("\nRunning 'synthetic data distribution '...\n")
    # min numbedr of samples to be povode
    df_path = "datasets/df_treatments_outcomes.csv"
    df = pd.read_csv(df_path)
    n_samples = 10000
    # Generating the synthetic dataset by sampling from the real world data
    synthetic_df = df.sample(n=n_samples, replace=True, random_state=42)

    # Display the first few rows of the synthetic dataset
    print(synthetic_df.head())

    # Calculate causal effects of treatments on outcomes
    treatments = ["Vegetables", "Sleep"]
    outcomes = ["Depression", "Distress"]

    for outcome in outcomes:
        for treatment in treatments:
            print("Treatment: %s\nOutcome: %s" % (treatment, outcome))

            # Gather data
            syn_df = gather_data(synthetic_df, treatment, outcome)

            # Calculate causal estimate with backdoor estimator
            results = backdoor(syn_df)
            print("Effect of 0 Percent Treatment on Outcome: %s" % results[0])
            print("Effect of 100 Percent Treatment on Outcome: %s\n" % results[1])
