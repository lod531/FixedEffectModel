import FixedEffectModel.api as FEM
import pandas as pd
import numpy as np

df = pd.read_stata('/home/abom/Desktop/regPyHDFE/data/test.dta')
consist_input = ['wks_ue','tenure']
output_input = ['ttl_exp']
category_input = ['year', 'idcode']
cluster_input = ['year']
print("absorbed", category_input)
print("clustered", cluster_input)
result1 = FEM.ols_high_d_category(df,consist_input,output_input,category_input,cluster_input,formula=None,robust=False,c_method = 'cgm',epsilon = 1e-8,max_iter = 1e6)
 
result1.summary()
