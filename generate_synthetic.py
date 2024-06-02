
# Import packages
import numpy as np
import scipy.stats as stats
import random
import pandas as pd

# Script to generate synthetic dataset to check if causal estimate framework is accurate for known causal effect


def generate_sample():
    """
    Generate single synthetic sample based on background distributions
    """

    # Exploring samples for Treatment="Veggies" and Outcome="Depression" and no location
    variables = ["Location", "Treatment", "Age", "Gender/Race", "Outcome", "Treatment_Value", "Outcome_Value"]
    location = "NA"
    treatment = "Vegetables"
    outcome = "Depression"

    # Categorical variable dict to hold values
    gender_race_dict = {'OVERALL': 1, 'NAA': 2, 'MALE': 3, 'FEMALE': 4, 'WHT': 5, 'BLK': 6, 'HIS': 7, 'ASN': 8}
    age_dict = {"65PLUS": 1, "5064": 2, "AGE_OVERALL": 3}
    
    # Sample values based on background distributions of variables
    curr_sample = []
    treatment_val = None
    age = None
    gender_race = None
    # Calculate value for each variable of interest
    for var in variables:
        if var in ["Location"]:
            curr_sample.append(location)
        elif var in ["Treatment"]:
            curr_sample.append(treatment)
        elif var in ["Treatment_Value"]:
            lower, upper = 0, 100
            mu, sigma = 50, 34
            X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            # Sample value from truncated normal distribution
            treatment_val = X.rvs(1)[0]
            curr_sample.append(str(4*age_dict[age] - 1*gender_race_dict[gender_race] + treatment_val))
        elif var in ["Outcome_Value"]:
            # ASSUMPTION: Outcome is 1.1 times the value of the treatment
            curr_sample.append(str(-2*age_dict[age] + 3*gender_race_dict[gender_race] + treatment_val*1.1+np.random.normal(0,1)))
        elif var in ["Age"]:
            age_vals = ["65PLUS", "5064", "AGE_OVERALL"]
            age = random.sample(age_vals, 1)[0]
            curr_sample.append(age)
        elif var in ["Gender/Race"]:
            gender_vals = ['OVERALL', 'NAA', 'MALE', 'FEMALE', 'WHT', 'BLK', 'HIS', 'ASN']
            gender_race = random.sample(gender_vals, 1)[0]
            curr_sample.append(gender_race)
        elif var in ["Outcome"]:
            curr_sample.append(outcome)

    return curr_sample


def generate_dataset(num_samples):
    """
    Generate full synthetic dataset
    """

    # Exploring samples for Treatment="Veggies" and Outcome="Depression"
    variables = ["Location", "Treatment", "Age", "Gender/Race", "Outcome", "Treatment_Value", "Outcome_Value"]

    # Format dataset into csv
    dataset = ",".join(variables)
    for i in range(num_samples):
        dataset += "\n%s" % (",".join(generate_sample()))
    
    return dataset


if __name__ == "__main__":
    print("\nRunning 'generate_synthetic.py'...\n")

    # Generate datasets of sample size 1,000, 10,000, and 100,000
    dataset_sizes = [1000, 10000, 100000]
    for num_samples in dataset_sizes:

        # Generate dataset
        dataset = generate_dataset(num_samples=num_samples)

        # Write to output file
        f = open('datasets/df_synthetic_%s.csv' % num_samples,'w')
        f.write(dataset)
        f.close()