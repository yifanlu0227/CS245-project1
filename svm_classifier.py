import os
import numpy as np
from dataloader.feature_loader import FeatureLoader
from sklearn import svm


loader = FeatureLoader()
(trainset,trainset_label,testset,testset_label) = loader.load_data()
indices = np.arange(trainset.shape[0])
np.random.shuffle(indices)
trainset = trainset[indices]
trainset_label = trainset_label[indices]

c_choices = {1}
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

        classifier = svm.LinearSVC(C=c)
        classifier.fit(X_train_k, y_train_k)
        y_val_pred = classifier.predict(X_val_k)        
        num_correct = np.sum(y_val_pred == y_val_k)
        num_val = X_val_k.shape[0]
        accuracy = float(num_correct) / num_val
        print("c_choice is:",c," at fold:",i," accuary is:",accuracy)
        A.append(accuracy)
    c_to_accuracies[c] = np.mean(A)