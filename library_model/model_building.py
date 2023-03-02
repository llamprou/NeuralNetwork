import torch
from torch import nn
import numpy as np
import pickle
import os
import pandas as pd
import copy
import random
import torch.optim as optim
from library_model import model_training as mt
from library_model import layers as lay
import copy


#-----------------------------------------------------------------------------------
#generate standard models 
def get_StandardFeedforward(dims_list, lr, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    hyper_param = [(dims_list[k], dims_list[k+1], True, True) for k in range(len(dims_list)-1)]
    model = lay.FeedForward(hyper_param).to(device)
    opt = optim.SGD(model.parameters(), lr)
    return model, opt

def get_SkipFeedforward(dims_list, lr, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    hyper_param = [(dims_list[k], dims_list[k+1], True, True) for k in range(len(dims_list)-1)]
    model = lay.SkipFeedForward(hyper_param).to(device)
    opt = optim.SGD(model.parameters(), lr)
    return model, opt





#-----------------------------------------------------------------------------------
#Transformer model 
def get_registered_Transformer_model(in_vocab_size, out_vocab_size, dim_in, dim_key, heads, dim_internal, copies, lr, start, end, optimizer = "sgd", device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    attention_block, FF_block = lay.get_coder_blocks(dim_in, dim_key, heads, dim_internal)
    transformer = lay.TransformerLayer(attention_block, FF_block, copies).to(device)
    position_enc = lay.Positional_enc(dim_in, max_dim= 5000).to(device)
    linear = lay.linear_layer((dim_in, out_vocab_size, False, False)).to(device)
    enc_embedding=nn.Embedding(in_vocab_size, dim_in).to(device)
    dec_embedding=nn.Embedding(out_vocab_size, dim_in).to(device)

    model = lay.Transformer_model(position_enc, enc_embedding, dec_embedding, transformer, linear, start, end).to(device)
    #initializing according to the transformer paper
    for p in model.parameters(): 
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr)
    else:
        print("Optimizer choice not recognized")
    
    return model, opt 



#-----------------------------------------------------------------------------------
# example of a learning rate schedule used in Annotated Transformer
def learning_rate_function(model_size, factor, warmup_steps):
    return lambda epoch : factor* (model_size)**(-0.5) * min((epoch+1)**(-0.5), (epoch+1) * (warmup_steps)**(-1.5)) 

def learning_rate_step(factor, drop, time):
    return lambda epoch : factor/(drop**(epoch//time))

#-----------------------------------------------------------------------------------
# class for model operations 
# USE .pt FOR SAVING MODELS!! The other methods seem to generate small errors in the saved tensors that ruin saved network performance.
class NN_operating_tools(mt.Model_training, nn.Module):
    def __init__(self, model, opt, learning_rate_schedule=None, saved_model=None): #learning_rate_schedule must be a lambda function
        nn.Module.__init__(self)
        self.model = model
        self.opt =opt
        if learning_rate_schedule is not None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, learning_rate_schedule)
        else:
            self.scheduler =None
        self.loss = nn.CrossEntropyLoss()
        mt.Model_training.__init__(self, model, opt, self.loss)

        if saved_model is not None:
            self.load_model(saved_model)

    
    def load_model(self, saved_model):
        dir =os.path.join("saved_models", saved_model)
        file_opener={".csv": self.load_model_csv, ".pt": self.load_model_pt, ".npy" : self.load_model_npy, ".pkl" : self.load_model_pkl}
        idx = saved_model.find(".")
        file_opener[saved_model[idx:]](dir)


    def load_model_pt(self, directory):
        t_load = torch.load(directory)
        self.model.load_state_dict(t_load)

    def load_model_csv(self, dir):
        load_model = pd.read_csv(dir)
        for idx, param in enumerate(self.model.parameters()):
            with torch.no_grad():
                param.copy_(torch.tensor(load_model.iloc[idx, 1]))


    def load_model_npy(self, dir):
        load_model = np.load(dir, allow_pickle=True)
        for k, param in enumerate(self.model.parameters()):
            with torch.no_grad():
                param.copy_(torch.from_numpy(load_model[k]))


    def load_model_pkl(self, dir):
        with open(dir, "rb") as f:
            load_model = pickle.load(f)
        for k, param in enumerate(self.model.parameters()):
                with torch.no_grad():
                    param.copy_(torch.tensor(load_model[k]))

    
    def save_model(self, name):
        dir =os.path.join("saved_models", name)
        file_saver={".csv": self.save_csv, ".pt": self.save_pt, ".npy" : self.save_npy, ".pkl" : self.save_pkl}
        idx = name.find(".")
        file_saver[name[idx:]](dir)
        

    def save_pt(self, directory):
        torch.save(copy.deepcopy(self.model.state_dict()), directory)


    def save_npy(self, directory):
        trained_model = [copy.deepcopy(p.detach()).numpy() for p in self.model.parameters()]
        np.save(directory, trained_model)

    def save_csv(self,directory):
        pass

    def save_pkl(self,directory):
        pass





    #-----------------------------------------------------------------------------------
#Transformer model NOT registered with nn.Module --parameters are collected manually using custom "parameters" class ---I was just playing around
#def get_Transformer_model(in_vocab_size, out_vocab_size, dim_in, dim_key, heads, dim_internal, copies, lr):
#    attention_block, FF_block = lay.get_coder_blocks(dim_in, dim_key, heads, dim_internal)
#    transformer = lay.TransformerLayer(attention_block, FF_block, copies)
#    position_enc = lay.Positional_enc(dim_in, max_dim= 5000)
#    linear = lay.linear_layer((dim_in, out_vocab_size, False, False))
#    enc_embedding=nn.Embedding(in_vocab_size, dim_in)
#    dec_embedding=nn.Embedding(out_vocab_size, dim_in)

#    model = lay.Transformer_model(position_enc, enc_embedding, dec_embedding, transformer, linear)
    #initializing according to the transformer paper
    #for p in model.parameters(): 
    #    if p.dim() > 1:
    #        nn.init.xavier_uniform_(p)

#    opt = optim.SGD(model.parameters(), lr)
#    return model, opt 
