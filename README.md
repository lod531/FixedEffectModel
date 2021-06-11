FixedEffectModelPyHDFE: A Python Package for Linear Model with High Dimensional Fixed Effects.
=======================
**FixedEffectModel** is a Python Package designed and built by **Kuaishou DA ecology group**. It provides solutions for linear model with high dimensional fixed effects,including support for calculation in variance (robust variance and multi-way cluster variance), fixed effects, and standard error of fixed effects. It also supports model with instrument variables (will upgrade in late Nov.2020).

As You may have noticed, this is not **FixedEffectModel**, but rather FixedEffectModel**PyHDFE**. The goal of this library is to reproduce the brilliant **regHDFE** Stata package on Python. To this end, the algorithm FEM used to calculate fixed effects has been replaced with **PyHDFE**, and a number of further changes have been made.

Presently, this package replicates regHDFE functionality for most use cases. For examples, please see tests/test\_clustering.py.

If You find a regression whose output is different in FEMPyHDFE than what regHDFE produces please open an issue on this repo!
 
# Installation

Install this package directly from PyPI
```bash
$ pip install FixedEffectModelPyHDFE
```
# Limitations

The original FEM package includes functionality other than absorbing and clustering variables - for example it includes instrumental variable functionality. The focus of this package for the moment is solely on the absorption and clustering functions, so no guarantees on any other functionality.

# Documentation

Documentation is provided by Kuaishou DA group [here](https://github.com/ksecology/FixedEffectModel). Below is a copy of their README for convenience (and slight modification to reflect that PyHDFE is also being used in this package)
# Main Functions

|Function name| Description|Usage
|-------------|------------|----|
|ols_high_d_category|get main result|ols_high_d_category(data_df, consist_input=None, out_input=None, category_input=None, cluster_input=[],fake_x_input=[], iv_col_input=[], formula=None, robust=False, c_method='cgm', psdef=True, epsilon=1e-8, max_iter=1e6, process=5)|
|ols_high_d_category_multi_results|get results of multiple models based on same dataset|ols_high_d_category_multi_results(data_df, models, table_header)|
|getfe|get fixed effects|getfe(result, epsilon=1e-8)|
|alpha_std|get standard error of fixed effects|alpha_std(result, formula, sample_num=100)|


# Example

For a plethora of examples, please also see tests/test_clustering.py

```python
import FixedEffectModelPyHDFE.api as FEM
import pandas as pd

df = pd.read_csv('path/to/yourdata.csv')

#define model
#you can define the model through defining formula like 'dependent variable ~ continuous variable|fixed_effect|clusters|(endogenous variables ~ instrument variables)'
formula_without_iv = 'y~x+x2|id+firm|id+firm'
formula_without_cluster = 'y~x+x2|id+firm|0|(Q|W~x3+x4+x5)'
formula = 'y~x+x2|id+firm|id+firm|(Q|W~x3+x4+x5)'
result1 = FEM.ols_high_d_category(df, formula = formula,robust=False,c_method = 'cgm',epsilon = 1e-8,psdef= True,max_iter = 1e6)

#or you can define the model through defining each part
# a.k.a. predictors
consist_input = ['x','x2']
# a.k.a. target
output_input = ['y']
# a.k.a. variables to be absorbed
category_input = ['id','firm']
cluster_input = ['id','firm']
endo_input = ['Q','W']
iv_input = ['x3','x4','x5']
c_method='cgm'
result1 = FEM.ols_high_d_category(df,consist_input,out_input,category_input,cluster_input,endo_input,iv_input,formula=None,robust=False,c_method = c_method,epsilon = 1e-8,max_iter = 1e6)

#show result
result1.summary()
```


# Requirements
- Python 3.6+
- Pandas and its dependencies (Numpy, etc.)
- Scipy and its dependencies
- statsmodels and its dependencies
- networkx
- PyHDFE

# Citation
If you use FixedEffectModel in your research, please cite the following:

Kuaishou DA Ecology. **FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.**<https://github.com/ksecology/FixedEffectModel>,2020.Version 0.x

BibTex:
```
@misc{FixedEffectModel,
  author={Kuaishou DA Ecology},
  title={{FixedEffectModel: {A Python Package for Linear Model with High Dimensional Fixed Effects}},
  howpublished={https://github.com/ksecology/FixedEffectModel},
  note={Version 0.x},
  year={2020}
}
```

Jeff Gortmaker and Anya Tarascina. **PyHDFE: High Dimensional Fixed Effect Absorption.**<https://github.com/jeffgortmaker/pyhdfe>,2019.Version 0.x

BibTex:
```
@misc{PyHDFE,
  author={Jeff Gortmaker with Anya Tarascina},
  title={{PyHDFE: {High Dimensional Fixed Effect Absorption},
  howpublished={https://github.com/jeffgortmaker/pyhdfe},
  note={Version 0.x},
  year={2019}
}
```

# Feedback
This package welcomes feedback. If you have any additional questions or comments, please contact <da_ecology@kuaishou.com>.


# Reference
[1] Simen Gaure(2019).  lfe: Linear Group Fixed Effects. R package. version:v2.8-5.1 URL:https://www.rdocumentation.org/packages/lfe/versions/2.8-5.1

[2] A Colin Cameron and Douglas L Miller. A practitioner’s guide to cluster-robust inference. Journal of human resources, 50(2):317–372, 2015.

[3] Simen Gaure. Ols with multiple high dimensional category variables. Computational Statistics & Data Analysis, 66:8–18, 2013.

[4] Douglas L Miller, A Colin Cameron, and Jonah Gelbach. Robust inference with multi-way clustering. Technical report, Working Paper, 2009.

[5] Jeffrey M Wooldridge. Econometric analysis of cross section and panel data. MIT press, 2010.
