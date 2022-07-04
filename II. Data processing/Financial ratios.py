import pandas as pd
import os

### Load data
os.chdir("/Users/aayushmarishi/Desktop/Jamie/FS courses/Financial Management/Data processing/Documentation")
ratio = pd.read_csv('Financial ratios.csv')

### Drop unncessary columns
ratio['year'] = pd.DatetimeIndex(ratio['public_date']).year
ratio_2 = ratio.drop(['adate','qdate','public_date'], axis = 1)
print(ratio_2.isnull().sum())

###Extract chosen ratios
ratio_3 = ratio_2[['gvkey', 'debt_at', 'debt_ebitda', 'debt_assets', 'cash_debt', 'year']]
ratio_3 = ratio_3.dropna(axis=0)
ratio_3 = ratio_3.drop_duplicates(subset=['gvkey', 'year'], keep='last')

ratio_3.to_csv('Ratios cleaned.csv')