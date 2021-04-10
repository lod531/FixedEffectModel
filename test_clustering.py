import pandas as pd
import FixedEffectModelPyHDFE.api as FEM
from FixedEffectModelPyHDFE.DemeanDataframe import get_np_columns
#import FixedEffectModel.api as FEM
import numpy as np
from patsy import dmatrices
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_stata('/home/abom/Desktop/FixedEffectModel/data/test.dta')
consist_input = ['wks_ue','tenure']
output_input = ['ttl_exp']
category_input = ['idcode']
cluster_input = ['idcode', 'birth_yr', 'fifty_clusts', 'sixty_clusts']
#cluster_input = ['idcode', 'sixty_clusts', 'fifty_clusts']

print("absorbed", category_input)
print("clustered", cluster_input)
result1 = FEM.ols_high_d_category(df,consist_input,output_input,category_input,cluster_input,formula=None,c_method='cgm2',robust=False,epsilon = 1e-8,max_iter = 1e6)
 
result1.summary()



y, X = dmatrices('ttl_exp ~ wks_ue + tenure', data=df, return_type='dataframe')

model = sm.OLS(y, X)

res = model.fit(cov_type='cluster', cov_kwds={'df_correction':True, 'groups':get_np_columns(df, columns=['birth_yr'])})
res = model.fit()

#print(res.summary())
#print(res.get_robustcov_results(df_correction = False, cov_type='cluster', groups=get_np_columns(df, columns=['fifty_clusts']) ).summary())



#import pandas as pd
#import econtools.metrics as mt
#
## Load a data file with columns 'ln_wage', 'educ', and 'state'
#
#y = 'ttl_exp'
#X = ['wks_ue', 'tenure']
#fe_var = ''
#cluster_var = 'birth_yr'
#
#results = mt.reg(
#    df,                     # DataFrame
#    y,                      # Dependent var (string)
#    X,                      # Independent var(s) (string or list of strings)
#    addcons=True,
#    cluster=cluster_var     # Cluster var (string)
#)
#print(results)
#print(results.Ftest(["wks_ue", "tenure"]))
