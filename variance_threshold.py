import numpy as np
import os
import sklearn

class VarianceThreshold:
    # using variance as the criterion
    def __init__(self,n_feature_target):
        self.target = n_feature_target # 目标将为多少维
    
    def feed(self,data,labels):
        '''
        data : input feature array shape (n_samples,n_features) (37322,2048)
        '''
        variance = np.var(data,axis=0) # find the variance in one feature, not one sample
        top_k_idx = np.argsort(variance)[::-1][0:self.target]
        # print(top_k_idx)
        return top_k_idx
