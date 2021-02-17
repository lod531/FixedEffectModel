FixedEffectModelPyHDFE: A Python Package for Linear Model with High Dimensional Fixed Effects.
=======================
**FixedEffectModel** is a Python Package designed and built by **Kuaishou DA ecology group**. It provides solutions for linear model with high dimensional fixed effects,including support for calculation in variance (robust variance and multi-way cluster variance), fixed effects, and standard error of fixed effects. It also supports model with instrument variables (will upgrade in late Nov.2020).

As You may have noticed, this is not **FixedEffectModel**, but rather FixedEffectModel**PyHDFE**. In this version, the fixed effects backend was switched to use the PyHDFE library, offering significant speed increases with no downsides. Some modifications have also been made in order to replicate reghdfe Stata package behaviour.
# Installation

Install this package directly from PyPI
```bash
$ pip install FixedEffectModelPyHDFE
```

# reghdfe reproduction

## Reproduced cases:

* Absorbing any number of variables with no clustering.

* Absorbing a single variable and clustering with that same variable.

* Absorbing a single variable and clustering on any number of variables
(including clustering and absorbing same variable).


## Cases sort of reproduced:

* Absorbing multiple variables and clustering on a single variable
(mild differences e.g. 484.87 v.s. 490.0133 Fval, this is probably
due to DoF adjustments when there is a mix of nested and not nested 
fixed effects within clusters).

* Absorbing multiple and clustering on multiple values, with two variable shared in
clustering and absorption.
    * This seems to deviate a little when absorbed degrees of freedom are not 
nested within a cluster. That's probably not the core issue though - 
if it were You'd expect clustering and absorbing same list of variables to just
work.
    * It's hard to pinpoint the cause for this, but the larger the F-value the 
larger the deviation e.g. 78.36 v.s. 78.4016 in smallest case 659.24 v.s. 670.6178 and in the case of larger values it's 3459.8180 v.s. 3115.68 when it's way up there.

    * It is not the case that the more variables are clustered and absorbed the worse
the divergence becomes e.g. 2 shared variables 3026.68 v.s. 3228.3221
5 shared variables 78.35 v.s. 78.4791 fvals.

    * Reghdfe will sometimes output thisL "Warning: VCV matrix was non-positive semi-definite; adjustment from Cameron, Gelbach & Miller applied." This adjustment may be missing.


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

#get fixed effects
getfe(result1 , epsilon=1e-8)

#define the expression of standard error of difference between two fixed effect estimations you want to know
expression = 'id_1-id_2'
#get standard error
alpha_std(result1, formula = expression , sample_num=100)

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
