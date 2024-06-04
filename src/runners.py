#Not set up for multi-gpu running, need to add things to get that ready 
#Also would like to add confusion matrix output
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
import _pickle as cPickle
import random
import itertools
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import boto3 as boto
import torch.multiprocessing as mp



#class Trainer:
    
    #def __init___(self, 
                  #model : torch.nn.Module,
                  #optimizer: torch.optim.Optimizer,
                  #loss_fn: torch.nn,
                  #save_interval: int, 
                  #metric_interval: int,
                  #train_data: DataLoader,
                  #validation_data: DataLoader = None, 
                  #test_data: DataLoader = None,
                  #save_path: str = None): 
        
        #Setting all variables equal to a local counterpart 
        #self.model = model
        #self.optimizer = optimizer
        #self.loss_fn = loss_fn
        #self.save_interval = save_interval
        #self.metric_interval = metric_interval
        #self.train_data = train_data
        #self.validation_data = validation_data
        #self.test_data = test_data
        #self.save_path = save_path
        
        #going to be used in evaluating function to decrease latency of model 
        #self.curr_predictions = []
        #self.curr_labels = []
        
class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 save_interval: int,
                 metric_interval: int,
                 train_data: DataLoader,
                 validation_data: DataLoader = None,
                 test_data: DataLoader = None,
                 save_path: str = None):
        
        # Setting all variables equal to a local counterpart
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_interval = save_interval
        self.metric_interval = metric_interval
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.save_path = save_path

        # Going to be used in evaluating function to decrease latency of model
        self.curr_predictions = []
        self.curr_labels = []
        
    def _run_batch(self,batch: torch.Tensor, batch_labels: torch.Tensor):
        #Setting current gradient to zero for new batch
        self.optimizer.zero_grad()
        #Running model on current batch
        #print("running _run_batch")
        pred_output = self.model(batch)
        #print("Done running batch")
        #Appending predicted values to list for evaluation 
        self.curr_predictions.append(pred_output)
        #Appending label values to list for evaluation
        self.curr_labels.append(batch_labels)
        
        #Computing loss 
        #print(f"PRED OUTPUT: {pred_output}")
        #print(f"BATCH LABEL: {batch_labels}")
        loss = self.loss_fn(pred_output.float(),batch_labels.float())
        
        #Computing gradient for each parameter in model
        loss.backward()
        #grads = [p.grad for p in self.model.parameters()]
        #print(grads[:50])
        #Gradient descent 
        self.optimizer.step()
        
    def _run_epoch(self, epoch:int):
        #Including epoch num in init bc likely will want to print out at somepoint to see how quickly model runs
        #Setting model to train
        
        self.model.train()
        #Re-initiating prediction/label accumulator for evaluation of specific epoch
        
        self.curr_predictions = []
        self.curr_labels = []
        
        #Looping over each training batch
        #print("training model")
        for batch_tensor, batch_labels in self.train_data:
            
            #print(f"batch_tensor: {batch_tensor.shape}")
            #print(f"batch_labels: {batch_labels.shape}")
            #print(batch_tensor)
            
            #Running gradient descent on each batch
            self._run_batch(batch_tensor,batch_labels)
        #print("done training model")
        #print("exiting _run_epoch")
    
    #TODO: FINISH how to save to server
    def _save_checkpoint(self, epoch: int):
        #Getting the model weights at particular checkpoint
        #print("in save_checkpoint method")
        checkpoint_model = self.model.state_dict()
        #print("after getting the state-dict")
        #Pickling model into checkpoint_{epoch} file
        #Note that pickle.dump saves model in local directory
        #Need to delete after dump and upload
        torch.save(checkpoint_model,f'checkpoint_{epoch}.pt')
        #cPickle.dump(checkpoint_model, open(f'checkpoint_{epoch}.pt', 'wb'))
        #print("after pickle dump")
        #NEED TO FINISH SAVING TO FOLDER OF DICTIONARIES
        
    
    def train(self, num_epochs:int):
        
        #Looping over number of epochs
        for epoch in range(1,num_epochs+1):
            
            #Running an epoch
            print(f"running {epoch} epoch")
            self._run_epoch(epoch)
            
            #print("outside self.save_interval")
            #Code to save the model every save_interval   
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                #print("In save_interval")
                self._save_checkpoint(epoch)
            #Saving the last model
            elif epoch == num_epochs:
                self._save_checkpoint(epoch)

            #Evaluating model every metric_interval
            #print("outside self.metric_interval")
            if self.metric_interval > 0 and epoch % self.metric_interval == 0:
                #Decreases time bc saved inferences for training data in list
                #Evaluating Training set
                self._evaluate(None)
                #Will have to do inference for test set
                if self.test_data != None:
                    #Evaluating test set
                    self._evaluate(self.test_data)
                    #Resetting the model to train 
                    self.model.train()
                
    def _evaluate(self, dataloader: DataLoader = None):
        #Converting to torch.no_grad to prevent gradient calculation
        with torch.no_grad():
            #Set model to evaluation
            self.model.eval()
            #If dataloader none, it means we are looking at the current training set accuracy 
            if dataloader == None:
                #Using already predicted values that we accumulated during the training 
                #This obviously is a lower bound on the accuracy of our model on the training set
                #However, transformer latency is extremely large so this will decrease training time overall
                #Also we don't care about the actual training accuracy, we only care about the overall trends of 
                #training accuracy
                predict_output = torch.vstack(self.curr_predictions)
                labels = torch.vstack(self.curr_labels)
                print("/tTRAINING SET VALUES")
            else:
                #Creating accumulators for test set
                test_predict = []
                test_labels = []
                #Looping over each tensor and label in the dataloader
                for batch_tensor, batch_label in dataloader:
                    #Predicting using model on test set
                    #print("IN THE ELSE STATEMENT TO TEST TEST SET")
                    prediction = self.model(batch_tensor)
                    #accumulating model predictions and labels of test set
                    test_predict.append(prediction)
                    test_labels.append(batch_label)
                    #print(f"TEST PREDICT len:{len(test_predict)}")
                #Vstacking outputs and labels so that tensors read (patient x 1)
                #Note loss function is MSE, so output from model will be a singular value that relates to our 
                #actual scale
                #This differs from CrossEntropyLoss, where model output would be vector of length (num_classes)
                #and each entry would be a probability of particular class
                predict_output = torch.vstack(test_predict)
                labels = torch.vstack(test_labels)
                print("/tTEST SET VALUES")
            
            #Squeezing output to get rid of nested tensors
            predict_output = torch.squeeze(predict_output)
            labels = torch.squeeze(labels)
            
            predict_output = torch.argmax(predict_output,axis = 1)
            labels = torch.argmax(labels,axis = 1)
            #Calculating loss of the model for train/test set
            loss = self.loss_fn(predict_output.float(), labels.float())
            #Calculating Mean Absolute Error based on train/test set
            #print(f"PREDICT_OUTPUT: {predict_output}")
            #print(f"LABELS: {labels}")
            MAE = (predict_output.float() - labels.float()).abs().mean().item()
            
            #Rounding predicted output so that it matches the exact categories given by Norwood scale
            #predict_output = torch.round(predict_output)
            
            #Calculating how many predictions were correct
            num_correct = (predict_output == labels).sum().item()
            
            #Calculating accuracy of model
            acc = num_correct / len(labels)
            
            print(f"\t\t NUMBER CORRECT: {num_correct}")
            print(f"\t\t ACCURACY: {acc}")
            print(f"\t\t MEAN ABSOLUTE ERROR: {MAE}")
            print(f"\t\t LOSS: {loss}")
            print(f"\t\t PREDICTED: {predict_output}")
            print(f"\t\t LABELS: {labels}")
            print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++")                     