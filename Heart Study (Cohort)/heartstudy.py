import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport as pr

df = pd.read_csv('D:\\Dropbox\\PythonProjects\\pythonprojects\\Heart Study (Cohort)\\fulldata.csv')
df

# Using profiler to quickly explore the dataset
profile = pr(df, title='Heart Dataset', html={'style': {'full_width': True}})
profile
