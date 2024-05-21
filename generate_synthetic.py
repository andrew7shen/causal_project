
# Import packages
import numpy as np
import scipy.stats as stats
import random
import pandas as pd


def generate_sample():
    """
    Generate single synthetic sample based on background distributions
    """

    # Exploring samples for Treatment="Veggies" and Outcome="Depression" and no location
    variables = ["Location", "Treatment", "Treatment_Value", "Age", "Gender/Race", "Outcome", "Outcome_Value"]
    location = "NA"
    treatment = "Vegetables"
    outcome = "Depression"
    
    # Sample values based on background distributions of variables
    curr_sample = []
    treatment_val = None
    for var in variables:
        if var in ["Location"]:
            curr_sample.append(location)
        elif var in ["Treatment"]:
            curr_sample.append(treatment)
        elif var in ["Treatment_Value"]:
            lower, upper = 0, 100
            mu, sigma = 50, 10
            X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            treatment_val = X.rvs(1)[0]
            curr_sample.append(str(treatment_val))
        elif var in ["Outcome_Value"]:
            # ASSUMPTION: Outcome is 1.1 times the value of the treatment
            curr_sample.append(str(treatment_val*1.1))
        elif var in ["Age"]:
            age_vals = ["65PLUS", "5064", "AGE_OVERALL"]
            curr_sample.append(random.sample(age_vals, 1)[0])
        elif var in ["Gender/Race"]:
            gender_vals = ['OVERALL', 'NAA', 'MALE', 'FEMALE', 'WHT', 'BLK', 'HIS', 'ASN']
            curr_sample.append(random.sample(gender_vals, 1)[0])
        elif var in ["Outcome"]:
            curr_sample.append(outcome)

    return curr_sample


def generate_dataset(num_samples):
    """
    Generate full synthetic dataset
    """

    # Exploring samples for Treatment="Veggies" and Outcome="Depression"
    variables = ["Location", "Treatment", "Treatment_Value", "Age", "Gender/Race", "Outcome", "Outcome_Value"]

    # Format dataset into csv
    dataset = ",".join(variables)
    for i in range(num_samples):
        dataset += "\n%s" % (",".join(generate_sample()))
    
    return dataset


if __name__ == "__main__":
    print("\nRunning 'generate_synthetic.py'...\n")

    # Generate single sample
    num_samples = 5000 # To match number of samples in actual dataset
    dataset = generate_dataset(num_samples=num_samples)

    # Write to output file
    f = open('datasets/df_synthetic.csv','w')
    f.write(dataset)
    f.close()