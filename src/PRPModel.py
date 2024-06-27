import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from datetime import datetime, timedelta

from data import PRPDataLoader
from classes import PRPModel
from runners import Trainer

#Note that the Embed_dim needs to be divisible by the num_heads of the attention mechanism
def main(dict_path: str,
        data_path:str,
        batch_size: int ,
        sequential : bool , 
        split: tuple , 
        image_size: tuple,
        num_patch: int,
        num_heads: int,
        vision_num_layers :int,
        conv_output_dim:int,
        hidden_layer_1:int,
        hidden_layer_2:int,
        hidden_layer_3:int,
        hidden_layer_4:int,
        hidden_layer_5:int,
        num_classification_output : int,
        dropout: float,
        save_interval:int,
        metric_interval:int,
        save_path:str,
        num_epochs:int
               ):
    train_data , test_data = PRPDataLoader(dict_path,data_path,num_classification_output,image_size,sequential,split,batch_size)
    in_channels, row_len, col_len = 4, image_size[0], image_size[1]
    
    assert row_len % num_patch == 0, 'row_len not divisible by num_patches' 
    assert col_len % num_patch == 0, "col_len not divisible by num_patches"
        
    patch_row = row_len // num_patch
    patch_col = col_len // num_patch
    embed_dim = patch_row * patch_col * in_channels
    assert embed_dim % num_heads == 0, "embed_dimension is not divisible by num_heads"

    model = PRPModel((batch_size,4,image_size[0],image_size[1]),
                     num_patch,
                     num_heads,
                     num_classification_output,
                     vision_num_layers,
                     conv_output_dim,
                     hidden_layer_1,
                     hidden_layer_2,
                     hidden_layer_3,
                     hidden_layer_4,
                     hidden_layer_5,
                     dropout)
    
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001)
    
    mse_loss = torch.nn.MSELoss()

    trainer = Trainer(model,adam_optimizer,mse_loss,save_interval,metric_interval,train_data,test_data= test_data,save_path = save_path)

    trainer.train(num_epochs=num_epochs)

if __name__ == "__main__":
    dict_path = ""
    data_path = ""
    batch_size = 1
    sequential = False
    split = (.8,.2)
    image_size = (300,300)
    num_patch = 50
    num_heads = 10
    vision_num_layers = 6
    conv_output_dim = None #Keeps dimension the same 
   #These can be changed, will likely need to be changed
   #And updated 
    hidden_layer_1 = 720
    hidden_layer_2 = 360
    hidden_layer_3 = 90
    hidden_layer_4 = 100,
    hidden_layer_5 = 25

    num_classification_output = 5
    dropout = .1
    save_interval = 1
    metric_interval = 1
    save_path = ""
    num_epochs = 100

    main(dict_path,
         data_path,
         batch_size,
         sequential,
         split,
         image_size,
         num_patch,
         num_heads,
         vision_num_layers,
         conv_output_dim,
         hidden_layer_1,
         hidden_layer_2,
         hidden_layer_3,
         hidden_layer_4,
         hidden_layer_5,
         num_classification_output,
         dropout,
         save_interval,
         metric_interval,
         save_path,
         num_epochs)



