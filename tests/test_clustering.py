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

NLS_WORK = "./../data/test_dropped_na.dta"
CEREAL = "./../data/cereal.dta"
AUTO = "./../data/auto_drop_na.dta"
TOLERANCE = 0.01

class FixedEffectsModelTests(unittest.TestCase):
    def setup(self, data_directory, target, regressors, absorb, cluster):
        df = pd.read_stata(data_directory)
        df.reset_index(drop=True, inplace=True)
        print(len(df.index))
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

    def test_no_absorb_cluster_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['0'],
                cluster=['idcode', 'birth_yr', 'fifty_clusts', 'sixty_clusts'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 4664.62, atol=TOLERANCE))
        # comparing standard errors
        assert(np.all(np.isclose(self.result.bse, np.asarray([.0551555, .0080815, .007881]), atol=TOLERANCE)))
        # comparing tvalues
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([74.92, 1.87, 96.03]), atol=TOLERANCE)))


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



    def test_just_absorb_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['fifty_clusts', 'sixty_clusts', 'birth_yr', 'idcode'],
                cluster=[])
        assert(np.isclose(self.result.fvalue, 3891.51, atol=TOLERANCE))
        assert(np.all(np.isclose(self.result.bse, np.asarray([.0047052, .0096448]), atol=TOLERANCE)))
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([6.48, 88.22]), atol=TOLERANCE)))



    def test_cluster_1_absorb_1_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['fifty_clusts'],
                cluster=['sixty_clusts'])
        assert(np.isclose(self.result.fvalue, 9884.24, atol=TOLERANCE))
        assert(np.all(np.isclose(self.result.bse, np.asarray([.004654, .0055812]), atol=TOLERANCE)))
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([3.18, 135.54]), atol=TOLERANCE)))

    def test_cluster_1_absorb_1_2_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['fifty_clusts'],
                cluster=['fifty_clusts'])
        assert(np.isclose(self.result.fvalue, 10100.50, atol=TOLERANCE))
        assert(np.all(np.isclose(self.result.bse, np.asarray([.0044538, .005324]), atol=TOLERANCE)))
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([3.33, 142.09]), atol=TOLERANCE)))

    def test_cluster_many_absorb_1_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['fifty_clusts'],
                cluster=['fifty_clusts', 'sixty_clusts', 'idcode', 'year'])
        assert(np.isclose(self.result.fvalue, 86.89, atol=TOLERANCE))
        assert(np.all(np.isclose(self.result.bse, np.asarray([.0189465, .0574001]), atol=TOLERANCE)))
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([0.78, 13.18]), atol=TOLERANCE)))


    def test_cluster_3_absorb_3_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['fifty_clusts', 'sixty_clusts', 'ind_code'],
                cluster=['idcode', 'year', 'grade'])
        assert(np.isclose(self.result.fvalue, 113.61, atol=TOLERANCE))
        assert(np.all(np.isclose(self.result.bse, np.asarray([.0168144, .0501467]), atol=TOLERANCE)))
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([0.93, 15.03]), atol=TOLERANCE)))


    def test_cluster_3_absorb_3_2_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['fifty_clusts', 'sixty_clusts', 'ind_code'],
                cluster=['fifty_clusts', 'sixty_clusts', 'ind_code'])
        assert(np.isclose(self.result.fvalue, 2525.34, atol=TOLERANCE))
        assert(np.all(np.isclose(self.result.bse, np.asarray([.004604, .0106474]), atol=TOLERANCE)))
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([3.41, 70.78]), atol=TOLERANCE)))



    def test_cluster_4_absorb_4_nls_work_dataset(self):
        self.setup(NLS_WORK, 
                target=['ttl_exp'],
                regressors=['wks_ue', 'tenure'],
                absorb=['fifty_clusts', 'sixty_clusts', 'ind_code', 'idcode'],
                cluster=['fifty_clusts', 'sixty_clusts', 'ind_code', 'idcode'])
        assert(np.isclose(self.result.fvalue, 3191.76, atol=TOLERANCE))
        assert(np.all(np.isclose(self.result.bse, np.asarray([.00498, .010914]), atol=TOLERANCE)))
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([6.17, 77.85]), atol=TOLERANCE)))


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



    def test_clustering_3_absorb_3_variables_auto_dataset(self):
        self.setup(AUTO, 
                target=['price'],
                regressors=['weight', 'length'],
                absorb=['rep78', 'headroom', 'turn'],
                cluster=['rep78', 'headroom', 'turn'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 21.46, atol=TOLERANCE))
        # comparing standard errors
        assert(np.all(np.isclose(self.result.bse, np.asarray([1.583412, 36.85108295]), atol=TOLERANCE)))
        # comparing tvalues
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([3.78, -1.01]), atol=TOLERANCE)))
 

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


#########################################################################
#########################################################################
# CEREAL DATASET

    def test_clustering_1_absorb_1_cluster_cereal_dataset(self):
        self.setup(CEREAL, 
                target=['rating'],
                regressors=['sugars', 'fat'],
                absorb=['shelf'],
                cluster=['shelf'])
        # comparing fvalue
        assert(np.isclose(self.result.fvalue, 148.48, atol=TOLERANCE))
        # comparing standard errors
        assert(np.all(np.isclose(self.result.bse, np.asarray([.1386962, 1.402163]), atol=TOLERANCE)))
        # comparing tvalues
        assert(np.all(np.isclose(self.result.tvalues, np.asarray([-15.49, -2.65]), atol=TOLERANCE)))
















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
