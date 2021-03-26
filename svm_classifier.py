import os
import numpy as np
from dataloader.feature_loader import FeatureLoader
from fisher_score import FisherScore
from variance_threshold import VarianceThreshold
from sklearn import svm
import time


loader = FeatureLoader()

dim_to_accuracies = []
dims = [20,50,100,200,500,800,1000,1500,2000]

for n_dim in dims:

    (trainset,trainset_label,testset,testset_label) = loader.load_data()
    # reducer = FisherScore(n_dim)
    reducer = VarianceThreshold(n_dim)
    selected_feature_idx = reducer.feed(trainset,trainset_label)
    trainset = trainset[:,selected_feature_idx]
    testset = testset[:,selected_feature_idx]

    indices = np.arange(trainset.shape[0])
    np.random.seed(227)
    np.random.shuffle(indices)
    trainset = trainset[indices]
    trainset_label = trainset_label[indices]

    c_choices = [1e-3]
    c_to_accuracies = {}
    num_folds = 5
    X_train_folds = np.array_split(trainset, num_folds)
    y_train_folds = np.array_split(trainset_label, num_folds)

    for c in c_choices:
        A = []
        for i in range(num_folds): 
            X_val_k = X_train_folds[i] #validation set
            y_val_k = y_train_folds[i] #validation set       
            X_train_k = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
            y_train_k = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
            start_time = time.time()
            classifier = svm.LinearSVC(C=c,max_iter=1000) # kernel implemented in liblinear
            classifier.fit(X_train_k, y_train_k)
            y_val_pred = classifier.predict(X_val_k)        
            num_correct = np.sum(y_val_pred == y_val_k)
            num_val = X_val_k.shape[0]
            accuracy = float(num_correct) / num_val
            end_time = time.time()
            print("dim is",n_dim,"c_choice is:",c," at fold:",i," accuary is:",accuracy," Time cost = ",(end_time-start_time))
            A.append(accuracy)
        print("After k-folds cross-validation, the accuary on valset is ",np.mean(A)," when c is ",c)
        c_to_accuracies[c] = np.mean(A)

        y_test_pred = classifier.predict(testset)        
        num_correct = np.sum(y_test_pred == testset_label)
        num_val = testset.shape[0]
        accuracy = float(num_correct) / num_val 
        dim_to_accuracies.append(accuracy)
        print("dim is",n_dim,"on testset the accuary is {}".format(accuracy))   

for i in range(len(dims)):
    print("dim:",dims[i]," acc:",dim_to_accuracies[i])