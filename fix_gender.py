import pandas as pd

df = pd.read_csv('dataset.csv')

# Standardize gender values
def map_gender(g):
    if g in ['M', 'Male']:
        return 'Male'
    elif g in ['F', 'Female']:
        return 'Female'
    elif g in ['Other', 'Prefer not to say']:
        return g
    else:
        return 'Other'  # Map all other identities to 'Other'

df['Gender'] = df['Gender'].apply(map_gender)

df.to_csv('dataset.csv', index=False)