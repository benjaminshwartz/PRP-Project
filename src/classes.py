import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import pydicom as dicom
import math
import time

class PositionalEncoding(nn.Module):
        
    def __init__(self, data_shape, dropout = .1):
        super(PositionalEncoding,self).__init__()
        
        #Get data shape
        #self.in_channels, self.row_len, self.col_len = data_shape
        self.row_len, self.col_len = data_shape
        
        self.learned_embedding = torch.zeros(data_shape)
        self.learned_embedding = nn.Parameter(self.learned_embedding[None,:,:])
                                              
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,data):
        
        data = data + self.learned_embedding
        data = self.dropout(data)
        
        return data

class Convlayer(nn.Module):
    
    def __init__(self,data_shape:tuple,num_patches:int,output_dim:int = None):
        super(Convlayer,self).__init__()
        self.num_patches = num_patches
        
        self.batch,self.in_channels, self.row_len, self.col_len = data_shape
        
        assert self.row_len % num_patches == 0 
        assert self.col_len % num_patches == 0
        
        
        self.patch_row = self.row_len // num_patches
        self.patch_col = self.col_len // num_patches
        
        self.embed_dim = int(self.in_channels * self.patch_row * self.patch_col)
        
        self.kernel_len = (int(self.patch_row), int(self.patch_col))
        
        patch_area = int(self.row_len * self.col_len)
        
        self.conv2d_1 = nn.Conv2d(in_channels = self.in_channels, 
                                  out_channels = self.embed_dim, 
                                  kernel_size =self. kernel_len, 
                                  stride = self.kernel_len)
        
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2) 
        if output_dim == None:
            self.dnn = nn.Linear(self.embed_dim,self.embed_dim)
        else:
            self.output_dim = output_dim
            self.dnn = nn.Linear(self.embed_dim,output_dim)
    def forward(self,data):
        
        
        #x = self.conv2d_1(data)
        #print(x.shape)
        #x = self.relu(x)
        #x = self.flatten(x)
        #print(x.shape)
        #x = torch.transpose(x,1,2)
        #print(f'DATA SHAPE: {data.shape}')
        batch = data.shape[0]
        patches = data.unfold(2,self.row_len,self.row_len).unfold(3,self.col_len,self.col_len)
        patches = torch.reshape(patches,(batch,self.num_patches**2,self.embed_dim))
        
        #print(x.shape)
        x = self.dnn(patches)
        
        return x
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self,data_shape,num_heads):
        
        super(MultiHeadAttention,self).__init__()
        #self.batch, self.patch, self.embed = data_shape
        self.patch, self.embed = data_shape
        self.attn = nn.MultiheadAttention(self.embed,num_heads,batch_first = True)
    def forward(self,data):
        #data = torch.reshape(data,(data.shape[1],data.shape[2],data.shape[3]))
        outputs , _ = self.attn(query=data, key=data, value=data, need_weights = False)
        return outputs

class MLP(nn.Module):
    def __init__(self,data_shape,output_size,dropout = .1):
        super(MLP,self).__init__()
        #self.batch, self.patch, self.embed = data_shape
        self.patch, self.embed = data_shape
        hidden_output = self.embed * 2
        self.lnn1 = nn.Linear(self.embed, hidden_output)
        self.dropout1 = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(hidden_output, output_size)
        self.dropout2 = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    def forward(self,data):
        
        x = self.lnn1(data)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fnn2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        
        return x
        
class TransformerEncoder(nn.Module):
    
    def __init__(self,data_shape,num_heads,dropout=.1):
        super(TransformerEncoder,self).__init__()
        self.data_shape = data_shape
        self.patch, self.embed = data_shape
        #self.batch, self.patch, self.embed = data_shape
        self.ln1 = nn.LayerNorm([self.patch,self.embed])
        self.ln2 = nn.LayerNorm([self.patch,self.embed])
        self.MHA = MultiHeadAttention(data_shape,num_heads)
        self.mlp = MLP(data_shape, output_size = self.embed, dropout=dropout)
        
    def forward(self,data):
        
        x = self.ln1(data)
        att_out = self.MHA(x)
        att_out = att_out + data
        after_ln2 = self.ln2(att_out)
        after_ln2 = self.mlp(after_ln2)
        after_ln2 = after_ln2 + att_out
        
        return after_ln2
        
class VisionTransformer(nn.Module):
    def __init__(self,data_shape,num_heads,num_layers = 6,dropout = .1):
        super(VisionTransformer,self).__init__()
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f'{i}', TransformerEncoder(data_shape=data_shape,num_heads = num_heads,dropout = dropout))
    
    def forward(self,data):
        x = data
        for blk in self.blks:
            x = blk(x)
        return x
    
class ClassificationHead(nn.Module):
    def __init__(self,
                 input_layer,
                 hidden_layer_1,
                 hidden_layer_2,
                 hidden_layer_3,
                 hidden_layer_4,
                 hidden_layer_5,
                 num_output,
                 dropout=.1):
        super(ClassificationHead,self).__init__()
        self.ln1 = nn.LayerNorm(input_layer)
        self.fnn1 = nn.Linear(input_layer,hidden_layer_1)
        self.dropout_1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_layer_1)
        self.fnn2 = nn.Linear(hidden_layer_1,hidden_layer_2)
        self.dropout_2 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(hidden_layer_2)
        self.fnn3 = nn.Linear(hidden_layer_2,hidden_layer_3)
        self.dropout_3 = nn.Dropout(dropout)
        self.ln4 = nn.LayerNorm(hidden_layer_3)
        self.fnn4 = nn.Linear(hidden_layer_3,hidden_layer_4)
        self.dropout_4 = nn.Dropout(dropout)
        self.ln5 = nn.LayerNorm(hidden_layer_4)
        self.fnn5 = nn.Linear(hidden_layer_4,hidden_layer_5)
        self.dropout_5 = nn.Dropout(dropout)
        self.fnn6 = nn.Linear(hidden_layer_5,num_output)
        
    def forward(self,data):
        x = self.ln1(data)
        x = self.fnn1(x)
        x = self.dropout_1(x)
        x = self.ln2(x)
        x = self.fnn2(x)
        x = self.dropout_2(x)
        x = self.ln3(x)
        x = self.fnn3(x)
        x = self.dropout_3(x)
        x = self.ln4(x)
        x = self.fnn4(x)
        x = self.dropout_4(x)
        x = self.ln5(x)
        x = self.fnn5(x)
        x = self.dropout_5(x)
        x = self.fnn6(x)
        
        return x
    
class PRPModel(nn.Module):
    def __init__(self,
                 data_shape,
                 num_patch:int,
                 num_heads,
                 num_output,
                 num_layers,
                 conv_output_dim,
                 hidden_layer_1,
                 hidden_layer_2,
                 hidden_layer_3,
                 hidden_layer_4,
                 hidden_layer_5,
                dropout = .1):
        super(PRPModel,self).__init__()
        self.batch_size = data_shape[0]
        self.data_shape = data_shape[1:]
        
        self.conv_layer = Convlayer(data_shape,num_patch,conv_output_dim)
        self.in_channels, self.row_len, self.col_len = self.data_shape
        assert self.row_len % num_patch == 0 
        assert self.col_len % num_patch == 0
        
        patch_row = self.row_len // num_patch
        patch_col = self.col_len // num_patch
        
        embed_dim = patch_row * patch_col * self.in_channels
        assert embed_dim % num_heads == 0, f"embed_dimension is not divisible by num_heads \nembed_dim: {embed_dim},heads:{num_heads}"
        
        #self.data_shape = (self.batch_size,num_patch**2,patch_row * patch_col * self.in_channels)
        self.data_shape = (num_patch**2,patch_row * patch_col * self.in_channels)
        
        self.pos_encode = PositionalEncoding(self.data_shape,dropout)
        
        
        self.visual_transformer = VisionTransformer(self.data_shape,num_heads,num_layers,dropout)
        
        #self.input_layer = self.data_shape[1] * self.data_shape[2]
        #self.input_layer = self.data_shape[0] * self.data_shape[1]
        self.input_layer = self.data_shape[0] * self.data_shape[1]
        
        #self.ClassificationHead = ClassificationHead(self.input_layer,
                                                     #hidden_layer_1,
                                                     #hidden_layer_2,
                                                     #hidden_layer_3,
                                                     #num_output,
                                                     #dropout=.1)
                            
        self.ClassificationHead = ClassificationHead(self.data_shape[1],
                                                     hidden_layer_1,
                                                     hidden_layer_2,
                                                     hidden_layer_3,
                                                     hidden_layer_4,
                                                     hidden_layer_5,
                                                     num_output,
                                                     dropout=.1)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self,data):
        batch_size = data.shape[0]
        
        x = self.conv_layer(data)
        x = self.pos_encode(x)
        x = self.visual_transformer(x)
        #print(x.shape)
        #x = torch.reshape(x,(batch_size,self.input_layer))
        x = torch.squeeze(x[:,-1,:])
        x = self.ClassificationHead(x)
        #print(f"BEFORE SOFTMAX: {x}")
        x = self.softmax(x)
        #x = torch.argmax(x,axis = 1)
        return x