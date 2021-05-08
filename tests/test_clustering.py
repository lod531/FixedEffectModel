# this tells python to act as if though We are one folder up
import sys
sys.path.insert(0,'..')

import pandas as pd
import FixedEffectModelPyHDFE.api as FEM
from FixedEffectModelPyHDFE.DemeanDataframe import get_np_columns
#import FixedEffectModel.api as FEM
import numpy as np
from patsy import dmatrices
import statsmodels.formula.api as smf
import statsmodels.api as sm

import unittest

from math import isclose

NLS_WORK = "./../data/test.dta"
AUTO = "./../data/auto_drop_na.dta"
TOLERANCE = 0.01

class FixedEffectsModelTests(unittest.TestCase):
    def setup(self, data_directory, target, regressors, absorb, cluster):
        df = pd.read_stata(data_directory).dropna()
        self.result = FEM.ols_high_d_category(df, 
                                    regressors, 
                                    target, 
                                    absorb,
                                    cluster,
                                    formula=None,
                                    robust=False,
                                    epsilon = 1e-8,
                                    max_iter = 1e6)

#########################################################################
#########################################################################
# nls work dataset
    def test_pure_regression_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['0'],
                cluster=[])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 4759.71, atol=TOLERANCE))
        # comparing standard errors
        assert(np.all(np.isclose(self.result.bse, np.asarray([.0415457, .0042073, .0078136]), atol=TOLERANCE)))
        # comparing tvalues
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([99.46, 3.60, 96.85]), atol=TOLERANCE)))

    def test_clustering_single_variable_no_absorb2_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['0'],
                cluster=['race'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 127593.72, atol=TOLERANCE))
        # comparing standard errors
        assert(np.all(np.isclose(self.result.bse, np.asarray([.148934, .0065111, .0113615]), atol=TOLERANCE)))
        # comparing tvalues
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([27.75, 2.32, 66.61]), atol=TOLERANCE)))


    def test_clustering_single_variable_no_absorb_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['0'],
                cluster=['fifty_clusts'])
        assert(np.isclose(self.result.fvalue, 10230.63, atol=TOLERANCE))

        assert(np.all(np.isclose(self.result.bse, np.asarray([.048274, .0044294, .0052923]), atol=TOLERANCE)))

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([85.60, 3.42, 143.00]), atol=TOLERANCE)))



    def test_clustering_two_variables_no_absorb_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['0'],
                cluster=['fifty_clusts', 'sixty_clusts'])
        assert(np.isclose(self.result.fvalue, 12347.24, atol=TOLERANCE))

        assert(np.all(np.isclose(self.result.bse, np.asarray([.0518019, .0048228, .00492]), atol=TOLERANCE)))

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([79.77, 3.14, 153.82]), atol=TOLERANCE)))

    def test_clustering_many_variables_no_absorb_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['0'],
                cluster=['fifty_clusts', 'sixty_clusts', 'birth_yr', 'idcode'])
        assert(np.isclose(self.result.fvalue, 4664.62, atol=TOLERANCE))

        assert(np.all(np.isclose(self.result.bse, np.asarray([.0551555, .0080815, .007881]), atol=TOLERANCE)))

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([74.92, 1.87, 96.03]), atol=TOLERANCE)))


#########################################################################
#########################################################################
# Boston auto dataset 

    def test_pure_regression_boston_auto_dataset(self):
        self.setup(AUTO, 
                target=['price'],
                regressors=['weight', 'length', 'turn'],
                absorb=['0'],
                cluster=[])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 14.78, atol=TOLERANCE))
        # comparing standard errors

        assert(np.all(np.isclose(self.result.bse, np.asarray([4667.441, 1.143408, 40.13139, 128.8455]), atol=TOLERANCE)))
        # comparing tvalues

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([3.19, 4.67, -1.75, -2.28]), atol=TOLERANCE)))
        

    def test_clustering_one_variable_no_absorb_auto_dataset(self):
        self.setup(AUTO, 
                target=['price'],
                regressors=['weight', 'length', 'turn'],
                absorb=['0'],
                cluster=['rep78'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 17.17, atol=TOLERANCE))
        # comparing standard errors

        assert(np.all(np.isclose(self.result.bse, np.asarray([6132.17, .8258151, 24.15393, 191.4521]), atol=TOLERANCE)))
        # comparing tvalues

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([2.42, 6.46, -2.91, -1.53]), atol=TOLERANCE)))
        


    def test_clustering_two_variables_no_absorb_auto_dataset(self):
        self.setup(AUTO, 
                target=['price'],
                regressors=['weight', 'length', 'turn'],
                absorb=['0'],
                cluster=['rep78', 'headroom'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 27.03, atol=TOLERANCE))
        # comparing standard errors

        assert(np.all(np.isclose(self.result.bse, np.asarray([6037.897, 1.210828, 44.88812, 183.8683]), atol=TOLERANCE)))
        # comparing tvalues

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([2.46, 4.41, -1.57, -1.60]), atol=TOLERANCE)))
        

    def test_clustering_two_variables_no_absorb_auto_dataset(self):
        self.setup(AUTO, 
                target=['price'],
                regressors=['weight', 'length', 'turn'],
                absorb=['0'],
                cluster=['rep78', 'headroom'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 27.03, atol=TOLERANCE))
        # comparing standard errors

        assert(np.all(np.isclose(self.result.bse, np.asarray([6037.897, 1.210828, 44.88812, 183.8683]), atol=TOLERANCE)))
        # comparing tvalues

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([2.46, 4.41, -1.57, -1.60]), atol=TOLERANCE)))



    def test_clustering_three_variables_no_absorb_auto_dataset(self):
        self.setup(AUTO, 
                target=['price'],
                regressors=['weight', 'length'],
                absorb=['0'],
                cluster=['rep78', 'headroom', 'turn'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 6.49, atol=TOLERANCE))
        # comparing standard errors

        assert(np.all(np.isclose(self.result.bse, np.asarray([5766.596, 1.884386, 59.54815]), atol=TOLERANCE)))
        # comparing tvalues

        assert(np.all(np.isclose(self.result.tvalues, np.asarray([1.78, 2.46, -1.62]), atol=TOLERANCE)))
 

       





if __name__ == '__main__':
    unittest.main()
#df = pd.read_stata('/home/abom/Desktop/FixedEffectModel/data/test.dta')
#consist_input = ['wks_ue','tenure']
#output_input = ['ttl_exp']
#category_input = ['idcode']
#cluster_input = ['idcode', 'birth_yr', 'fifty_clusts', 'sixty_clusts']
##cluster_input = ['idcode', 'sixty_clusts', 'fifty_clusts']
#
#print("absorbed", category_input)
#print("clustered", cluster_input)
#result1 = FEM.ols_high_d_category(df,consist_input,output_input,category_input,cluster_input,formula=None,c_method='cgm2',robust=False,epsilon = 1e-8,max_iter = 1e6)
#import pdb; pdb.set_trace() 
#result1.summary()
