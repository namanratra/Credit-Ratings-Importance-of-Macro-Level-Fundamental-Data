import pandas as pd
import os

### Load data
os.chdir("/Users/aayushmarishi/Desktop/Jamie/FS courses/Financial Management/Data processing/Documentation")
fiscal = pd.read_csv('Fiscal year end original.csv')

### Drop unncessary columns
fiscal = fiscal.drop(['indfmt','consol','popsrc','datafmt','curcd','costat','datadate'], axis =1)
fiscal = fiscal.rename(columns={'fyear': 'fiscal_year', 'fyr': 'fiscal_month'})

### Drop duplicates 
fiscal2 = fiscal.drop_duplicates(subset=['gvkey', 'fiscal_year'], keep='last')
fiscal2.to_csv('Fiscal cleaned.csv')