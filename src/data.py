import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import boto3 as boto
import random
import _pickle as cPickle
from torch.utils.data.distributed import DistributedSampler
import os
import torchvision.transforms as T 
from torchvision.transforms import v2
from torchvision.io import read_image

#SINGLE GPU INSTANCE CAPABILITIES ONLY. WILL HAVE TO UPDATE FOR MULTIGPU
class PRPDataSet(Dataset):
    
    def __init__(self, patient_ids: list,patient_labels: dict, num_classification_output: int,path: str, size: tuple):
        
        #dictionary of where patient ids are keys and rating is value
        self.patient_labels = patient_labels
        
        #list of patient ids
        self.patient_ids = patient_ids
        
        #path
        self.path = path
        
        #calculating mean and standard deviation along channels for entire dataset
        mean,std = self.mean_std()
        
        self.num_classification_output = num_classification_output
        
        #transformation used on each image 
        #Note that mean/std are vectors of len(4) for the alpha,RBG channels
        self.transforms = v2.Compose([
            v2.Resize(size),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean = mean,std = std)
        ])
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self,idx:int):
        
        #indexing via list
        patient = self.patient_ids[idx]
        
        #pulling image from storage bucket and un-pickling it
        #THIS MAY HAVE TO BE CHANGED DEPENDING ON HOW EXACTLY THE IMAGES ARE STORED
        #image = cPickle.load(open(f'{self.path}/{patient}/image'),"rb")
        image = read_image(f'{patient}.png')
        
        #Resize image
        resized_image = self.transforms(image)
        
        #Get label
        label_tensor = [0 for _ in range(self.num_classification_output)]
        label = torch.tensor(self.patient_labels[patient])
        label_tensor[label] = 1
        label_tensor = torch.tensor(label_tensor)
        
        return resized_image, label_tensor
    
    #Should maybe try to calculate mean_std deviation once prior to running script and store data
    #So that we do not have to rerun this step everytime the script is rerun 
    #
    #Note we are calculating the mean of pixel average for each the image
    #And the mean of the pixel std. dev for each the images
    def mean_std(self):
        #sum accumulator tensor for 4 channels x number of patients
        sum_values = torch.zeros(4,len(self.patient_ids))
        #std. dev accumulator tensor for 4 channels x number of patients
        std_values = torch.zeros(4,len(self.patient_ids))
        #Looping over photos
        #Have to do this, cannot vectorize since the images are all of different shapes
        for i in range(len(self.patient_ids)):
            #Getting patient
            patient = self.patient_ids[i]
            #opening photo from path
            #Make sure to pickle these images as Tensor objects
            #image = cPickle.load(open(f'{patient}.png',"rb"))
            image = read_image(f'{patient}.png').float()
            #Getting pixel average of the photo
            pixel_avg = torch.mean(image, dim = (1,2))
            #Storing patient i in ith row of sum value
            sum_values[:,i] = pixel_avg
            #Getting pixel std. dev of the photo
            pixel_std = torch.std(image, dim = (1,2))
            #Storing patient i std. dev ith row of std. dev matrix
            std_values[:,i] = pixel_std
        
        #Calculating average mean and std. dev. values
        a = torch.mean(sum_values, axis = 1)
        b = torch.mean(std_values, axis = 1)
        
        return a,b
    
def sequential_train_test_split(split: tuple, labels:dict):
    
    #Getting number of patients 
    num_patients = len(labels.keys())
    #Getting how patients to include in training
    training_num = int(split[0] * num_patients)
    #creating training set
    training_patients = labels.keys()[:training_num]
    #Creating Test set 
    testing_patients = labels.keys()[training_num:]

    return training_patients, testing_patients

def random_train_test_split(split: tuple, labels:dict):
    
    #Converting label list to label set to use subtraction
    patient_set = set(labels.keys())
    
    #Traning set
    training_patients = set()

    #calculating number of patients
    num_patients = len(labels.keys())
    
    #calculating number of patients in training set
    training_num = int(split[0] * num_patients)
    
    
    i = 0
    #looping number of patients 
    while i != training_num:
        #Choosing a patient from the list of left over patients not yet chosen
        curr_patient = random.choice(list(patient_set-training_patients))
        #adding patient to list of training patient
        training_patients.add(curr_patient)
        #Iterating
        i += 1

    #Getting test set
    test_patients = patient_set - training_patients

    #Reconverting back to list 
    return list(training_patients), list(test_patients)

def get_train_test_dataset(dict_path: str, data_path: str, num_class_output: int,size: tuple ,sequential: bool, split:tuple):
    
    #Getting dictionary that has patient keys and scores as values
    #Path may need to be changed
    patient_labels = cPickle.load(open(dict_path, 'rb'))
    
    #If want the data to be split sequentially (i.e in order)
    if sequential:
        train, test = sequential_train_test_split(split,patient_labels)
    else:
    #If want data split randomly (More Likely used)
        train, test = random_train_test_split(split,patient_labels)
    
    #Creating PRPDataSet objects for both train and test sets
    train_set = PRPDataSet(train,patient_labels,num_class_output,data_path,size)
    test_set = PRPDataSet(train,patient_labels,num_class_output,data_path,size)
    
    return train_set,test_set

def PRPDataLoader(dict_path: str, data_path: str, num_class_output: int,size: tuple,sequential: bool, split: tuple, batch: int):
    
    #Getting PRPDataSet objects for both rain and test sets 
    train_set, test_set = get_train_test_dataset(dict_path, data_path, num_class_output, size ,sequential, split)
    
    #Creating DataLoader with Train and Set Data sets
    train_generator = DataLoader(train_set, batch_size = batch, shuffle = True)
    test_generator = DataLoader(test_set, batch_size = batch, shuffle = True)
    
    return train_generator,test_generator