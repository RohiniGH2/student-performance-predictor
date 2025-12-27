import pandas as pd
import numpy as np
import random

n = 500  # number of entries

# Value pools
genders = ['M', 'F']
socio_econ = ['Low', 'Medium', 'High']
extracurricular = ['Yes', 'No']

data = {
    'Gender': np.random.choice(genders, n),
    'Age': np.random.randint(15, 23, n),
    'Previous GPA': np.round(np.random.normal(3.0, 0.5, n).clip(0, 4), 2),
    'Past class failures': np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05]),
    'Study time per week': np.random.randint(2, 21, n),
    'Attendance rate': np.round(np.random.normal(90, 7, n).clip(60, 100), 1),
    'Parental education level': np.round(np.random.normal(3, 1, n).clip(0, 4), 1),
    'Socioeconomic status': np.random.choice(socio_econ, n, p=[0.3, 0.5, 0.2]),
    'Motivation level': np.random.randint(1, 6, n),
    'Self-discipline': np.random.randint(1, 6, n),
    'Health status': np.random.randint(1, 6, n),
    'Hours of sleep': np.random.randint(4, 10, n),
    'Parental involvement': np.random.randint(1, 6, n),
    'Peer support': np.random.randint(1, 6, n),
    'Participation in extracurricular activities': np.random.choice(extracurricular, n),
}

# Final Grade (basic model using weighted sum + noise)
grade = (
    data['Previous GPA'] * 20 +
    np.array(data['Study time per week']) * 1.5 +
    np.array(data['Motivation level']) * 2 +
    np.array(data['Self-discipline']) * 2 +
    np.array(data['Parental involvement']) * 1.5 +
    np.array(data['Peer support']) * 1.2 +
    np.array(data['Attendance rate']) * 0.3 +
    np.random.normal(0, 5, n)
)
data['Final grade (out of 100)'] = np.round(grade.clip(0, 100), 1)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)

print("✅ dataset.csv has been created with 500 entries.")
