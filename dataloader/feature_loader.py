import os
import sys
import sklearn
import numpy as np
import time

class FeatureLoader:
    def __init__(self):
        self.features_dir = "./ResNet101/AwA2-features.txt"
        self.filename_dir = "./ResNet101/AwA2-filenames.txt"
        self.labels_dir = "./ResNet101/AwA2-labels.txt"

        self.features_npy_dir = "./ResNet101/AwA2-features.npy"
        self.labels_npy_dir = "./ResNet101/AwA2-labels.npy"

    def load_data_original(self):
        start_time = time.time()
        features = np.load(self.features_npy_dir)
        end_time = time.time()
        print("load feature array of shape:",features.shape," Successfully. Using time ",(end_time-start_time)," seconds.")

        start_time = time.time()
        labels = np.load(self.labels_npy_dir)
        labels = labels.astype(int)
        end_time = time.time()
        print("load label array of shape:",labels.shape," Successfully. Using time ",(end_time-start_time)," seconds.")

        return features,labels

    def load_data(self):

        start_time = time.time()
        features = np.load(self.features_npy_dir)
        end_time = time.time()
        print("load feature array of shape:",features.shape," Successfully. Using time ",(end_time-start_time)," seconds.")

        start_time = time.time()
        labels = np.load(self.labels_npy_dir)
        labels = labels.astype(int)
        end_time = time.time()
        print("load label array of shape:",labels.shape," Successfully. Using time ",(end_time-start_time)," seconds.")

        # Split the images in each category into 60% for training and 40% for testing.
        labels_list = labels.reshape(-1).tolist()

        # [start,end), not including end
        labels_start_index = {}
        labels_end_index = {}
        for i in range(1,51):
            labels_start_index[i] = labels_list.index(i)
            labels_end_index[i] = labels_list.index(i) + labels_list.count(i)


        # split training set and test set, take care of initial state
        print("spliting training set and testset...")
        start = labels_start_index[1]
        end = labels_end_index[1]
        split_point = start + int((end-start)*0.6)
        trainset = features[start:split_point,:]
        trainset_label = labels[start:split_point]
        testset = features[split_point:end]
        testset_label = labels[split_point:end]
        
        for i in range(2,51):
            start = labels_start_index[i]
            end = labels_end_index[i]
            split_point = start + int((end-start)*0.6)
            trainset = np.concatenate((trainset,features[start:split_point,:]),axis=0)
            trainset_label = np.concatenate((trainset_label,labels[start:split_point]),axis=0)
            testset = np.concatenate((testset,features[split_point:end]),axis=0)
            testset_label = np.concatenate((testset_label,labels[split_point:end]),axis=0)
        print("spliting finished.")

        return (trainset,trainset_label,testset,testset_label)

        