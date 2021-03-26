import os
import sys
import numpy as np

class FisherScore:
    def __init__(self,n_feature_target):
        self.target = n_feature_target
    
    def feed(self,data,label):
        '''
        Features with high quality should assign similar values to instances in the same class 
        and different values to instances from different classes.

        We have 50 classes

        data : input feature array shape (n_samples,n_features) (37322,2048)
        label: input label array shape (n_sample,)  (37322,)
        '''
        mean = np.mean(data,axis=0) # (2048,)
        mean_ij = np.zeros((50,2048)) # (50,2048)
        var_ij = np.zeros((50,2048)) # (50,2048)
        n = np.zeros(50) # (50,)
        # first we serparate different labels
        # feature_class[i] maps to feature have label i
        for i in range(50):
            label_idx = (label==i+1)
            selected_samples = data[label_idx,:]
            n[i] = selected_samples.shape[0]
            mean_ij[i,:] = np.mean(selected_samples,axis=0)
            var_ij[i,:] = np.var(selected_samples,axis=0)
        
        # Scores = np.zeros(2048)
        # for j in range(2048):
        #     Scores[j] = np.sum(n*np.square((mean_ij[:,j]-mean[j])))/np.square(np.sum(n*var_ij[:,j]))
        Scores = np.sum(n*np.square(mean_ij-mean).T,axis=1)/np.square(np.sum(n*var_ij.T,axis=1))

        top_k_idx = np.argsort(Scores)[::-1][0:self.target]
        # print(top_k_idx)
        return top_k_idx
                
