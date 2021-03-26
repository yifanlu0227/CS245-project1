import numpy as np
import os
import sys
from sklearn import svm

class GeneticAlgorithm:
    def __init__(self,fitness_target):
        '''
        fitness_target: the target of fitness function. We simply apply SVM val accuary as fitness function.
        '''
        self.fitness_target = fitness_target
        
    def feed(self,data,label):
        '''
        data : input feature array          (n_samples,n_features)  (37322,2048)
        label: label of sample              (n_sample,)             (37322,)
        '''
        self.data = data
        self.label = label

    def init_population(self,pop_size=30,chromo_length = 2048):
        self.pop_size = 30
        chromosomes = np.random.randint(0,2,(pop_size,chromo_length))
        return chromosomes

    def fitness_population(self,chromosomes):
        '''
        return: fitness (pop_size,1)
        '''

        return fitness # 


    def select_population(self,chromosomes,fitness):
        selected = chromosomes[fitness.argsort()[::-1][:self.pop_size//2]
