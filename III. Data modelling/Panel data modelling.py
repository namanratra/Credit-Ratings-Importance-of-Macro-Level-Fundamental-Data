import pandas as pd
import os
from linearmodels import PanelOLS
import statsmodels.api as sm
from linearmodels import PooledOLS
import itertools

### Load data
os.chdir("/Data processing/Documentation")
data_ori = pd.read_csv('Final rating dataset.csv', index_col = ['gvkey', 'year'])

dataset = data_ori.drop(['Unnamed: 0', 'Company', 'state'], axis=1)

years = dataset.index.get_level_values('year').to_list()
dataset['year'] = pd.Categorical(years)

### Aggregate macro predictor columns
col = ['PerCapitaIncome', 'GDP_per_capita'] + list(dataset.columns)[6:-2]

### Create report for storing univariate analysis result
uni_report = pd.DataFrame(columns=['Model', 'Independent variable','R2', 'Coefficient_ind',
                                   'Coefficient_debt_at', 'Coefficient_debt_assets',
                                   'Coefficient_debt_ebitda', 'Coefficient_cash_debt',
                                   'p-value'])

ratios = ['debt_at', 'debt_assets', 'debt_ebitda', 'cash_debt']

for c in col:
    exog = sm.tools.tools.add_constant(dataset[c])
    endog = dataset['Rating_num']
    # Pooled OLS model
    mod = PooledOLS(endog, exog)
    pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)
    
    # fixed effects model
    exog_vars = ratios + [c]
    exog = sm.add_constant(dataset[exog_vars])
    model_fe = PanelOLS(endog, exog, entity_effects = True) 
    fe_res = model_fe.fit() 
    
    # store results
    uni_report.loc[len(uni_report)] = ['Pooled OLS', c, pooledOLS_res.rsquared, 
               pooledOLS_res.params[1],0, 0, 0, 0, pooledOLS_res.pvalues[-1]]
    uni_report.loc[len(uni_report)] = ['Fixed effect', c, fe_res.rsquared, 
               fe_res.params[-1], fe_res.params[-5], fe_res.params[-4],
               fe_res.params[-3], fe_res.params[-2], fe_res.pvalues[-1]]

## export report
#uni_report.to_csv('Univariate models.csv')


## Multivariate analysis with 2 non-financial variables
multi = ['tornado_count', 'tropical_storm_death', 'tropical_storm_count', 
         'wildfire_dummy', 'wildfire_count', 'tropical_storm_dummy', 
         'tornado_injury', 'flood_count', 'flood_death', 'avg_tornado_scale',
         'PerCapitaIncome', 'GDP_per_capita', 'heat_death', 'winter_storm_count']

mul_2_report = pd.DataFrame(columns=['Model', 'Independent variable1', 
                                     'Independent variable2', 'Adjusted R2', 'Coefficient1',  
                                     'Coefficient2'])

for subset in itertools.combinations(multi, 2):
    indep = list(subset)
    exog = sm.tools.tools.add_constant(dataset[indep])
    endog = dataset['Rating_num']
    
    # fixed effects model
    exog_vars = ratios + indep
    exog = sm.add_constant(dataset[exog_vars])
    model_fe = PanelOLS(endog, exog, entity_effects = True) 
    fe_res = model_fe.fit() 
    fe_adj_rsquared = 1 - (1-fe_res.rsquared) * 10562 / 10560
    
    # record results
    mul_2_report.loc[len(mul_2_report)] = ['Fixed effect', indep[0], indep[1], 
                     fe_adj_rsquared, fe_res.params[-2], fe_res.params[-1]]
    
mul_2_report.to_csv('Multivariate2 models.csv')

## Multivariate analysis with 3 non-financial variables
mul_3_report = pd.DataFrame(columns=['Model', 'Independent variable1', 
                                     'Independent variable2', 'Independent variable3',
                                     'Adjusted R2', 'Coefficient1', 'Coefficient2', 
                                     'Coefficient3'])

for subset in itertools.combinations(multi, 3):
    indep = list(subset)
    exog = sm.tools.tools.add_constant(dataset[indep])
    endog = dataset['Rating_num']
    
    # fixed effects model
    exog_vars = ratios + indep
    exog = sm.add_constant(dataset[exog_vars])
    model_fe = PanelOLS(endog, exog, entity_effects = True) 
    fe_res = model_fe.fit()
    fe_adj_rsquared = 1 - (1-fe_res.rsquared) * 10562 / 10559
    
    # record results
    mul_3_report.loc[len(mul_3_report)] = ['Fixed effect', indep[0], indep[1], 
                     indep[2], fe_adj_rsquared, fe_res.params[-3], 
                     fe_res.params[-2], fe_res.params[-1]]

mul_3_report.to_csv('Multivariate3 models.csv')


##Multivariate analysis with 4 non-financial variables
mul_4_report = pd.DataFrame(columns=['Model', 'Independent variable1', 
                                     'Independent variable2', 'Independent variable3',
                                     'Independent variable4', 'Adjusted R2', 'Coefficient1', 
                                     'Coefficient2', 'Coefficient3', 
                                     'Coefficient4'])

for subset in itertools.combinations(multi, 4):
    indep = list(subset)
    exog = sm.tools.tools.add_constant(dataset[indep])
    endog = dataset['Rating_num']
    
    # fixed effects model
    exog_vars = ratios + indep
    exog = sm.add_constant(dataset[exog_vars])
    model_fe = PanelOLS(endog, exog, entity_effects = True) 
    fe_res = model_fe.fit() 
    fe_adj_rsquared = 1 - (1-fe_res.rsquared) * 10562 / 10558
    
    # record results
    mul_4_report.loc[len(mul_4_report)] = ['Fixed effect', indep[0], indep[1], 
                     indep[2], indep[3], fe_adj_rsquared, fe_res.params[-4], 
                     fe_res.params[-3], fe_res.params[-2], fe_res.params[-1]]

mul_4_report.to_csv('Multivariate4 models.csv')


## Multivariate analysis with 5 non-financial variables
mul_5_report = pd.DataFrame(columns=['Model', 'Independent variable1', 
                                     'Independent variable2', 'Independent variable3',
                                     'Independent variable4', 'Independent variable5',
                                     'Adjusted R2', 'Coefficient1', 'Coefficient2',
                                     'Coefficient3', 'Coefficient4', 
                                     'Coefficient5'])

for subset in itertools.combinations(multi, 5):
    indep = list(subset)
    exog = sm.tools.tools.add_constant(dataset[indep])
    endog = dataset['Rating_num']
    # Pooled OLS model
    mod = PooledOLS(endog, exog)
    pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)
    
    # fixed effects model
    exog_vars = ratios + indep
    exog = sm.add_constant(dataset[exog_vars])
    model_fe = PanelOLS(endog, exog, entity_effects = True) 
    fe_res = model_fe.fit() 
    fe_adj_rsquared = 1 - (1-fe_res.rsquared) * 10562 / 10557
    
    # record results
    mul_5_report.loc[len(mul_5_report)] = ['Fixed effect', indep[0], indep[1], 
                     indep[2], indep[3], indep[4], fe_adj_rsquared, fe_res.params[-5], 
                     fe_res.params[-4], fe_res.params[-3], fe_res.params[-2], 
                     fe_res.params[-1]]

mul_5_report.to_csv('Multivariate5 models.csv')

# sort results from multivariate models using combinations of 5
sorted_multi5 = mul_5_report.sort_values(by = 'Adjusted R2', ascending = False)

# output the result of the best model that has the highest adjusted r squared values
selected_model = sorted_multi5.iloc[0,:]
selected_model_result = pd.DataFrame(columns=['Model', 'Independent variable1', 
                                     'Independent variable2', 'Independent variable3',
                                     'Independent variable4', 'Independent variable5',
                                     'Adjusted R2', 'Coefficient1', 'Coef1 pvalue',
                                     'Coefficient2', 'Coef2 pvalue',
                                     'Coefficient3', 'Coef3 pvalue', 
                                     'Coefficient4', 'Coef4 pvalue',
                                     'Coefficient5', 'Coef5 pvalue'])

indep = list(selected_model[1:6])
exog_vars = ratios + indep
exog = sm.add_constant(dataset[exog_vars])
model_fe = PanelOLS(endog, exog, entity_effects = True) 
fe_res = model_fe.fit() 
fe_adj_rsquared = 1 - (1-fe_res.rsquared) * 10562 / 10557

selected_model_result.loc[len(selected_model_result)] = ['Fixed effect', 
                           indep[0], indep[1], indep[2], indep[3], indep[4], 
                           fe_adj_rsquared, fe_res.params[-5], fe_res.pvalues[-5],
                           fe_res.params[-4], fe_res.pvalues[-4], fe_res.params[-3], 
                           fe_res.pvalues[-3], fe_res.params[-2], fe_res.pvalues[-2],
                           fe_res.params[-1], fe_res.pvalues[-1] ]
print(selected_model_result)
