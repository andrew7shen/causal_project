
# Import packages
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from statistics import mean
import pandas as pd

# Script to generate causal estimate for Alzheimer's Disease and Healthy Aging dataset using backdoor estimator.
# We start with generating an estimate without taking into account confounders.


def backdoor(df, confounders=["c", "d", "e", "f"]):
    """
    Confounders being:
        c = race/ethnicity
        d = gender
        e = age
        f = location
    """

    # Use no confounders first
    confounders = []
    
    results = [] # shape of (a_dim)

    # Train outcome model without confounders
    ols = "Outcome_Value ~ Treatment_Value"
    for cf in confounders:
        ols += " + %s" % cf
    model = smf.ols(ols, data=df).fit()

    # Calculate backdoor estimate for each value of "Treatment_Value"
    # 259 unique values for "Treatment_Value"
    a_vals = np.unique(df["Treatment_Value"])
    # Set values for treatment to 0% or 100%
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


def gather_data(df_path, treatment, outcome):

    # Read in dataset
    df = pd.read_csv(df_path)

    # Filter for just Ai and Yi
    df = df[(df["Treatment"]==treatment) & (df["Outcome"]==outcome)]

    return df


if __name__ == "__main__":
    print("\nRunning 'backdoor.py'...\n")

    # Calculate causal effects of treatments on outcomes
    df_path = "datasets/df_treatments_outcomes.csv"
    treatments = ["Vegetables", "Sleep"]
    outcomes = ["Depression", "Distress"]

    for outcome in outcomes: 
        for treatment in treatments:
            print("Treatment: %s\nOutcome: %s" % (treatment, outcome))

            # Gather data
            df = gather_data(df_path, treatment, outcome)
            # print(df)

            # Calculate causal estimate with backdoor estimator
            results = backdoor(df)
            print("Effect of 0 Percent Treatment on Outcome: %s" % results[0])
            print("Effect of 100 Percent Treatment on Outcome: %s\n" % results[1])

