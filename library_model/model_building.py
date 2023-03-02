import torch
from torch import nn
import numpy as np
import pickle
import os
import pandas as pd
import copy
import random
import torch.optim as optim
from library_model import layers as lay
import copy




#-----------------------------------------------------------------------------------
#GET TRANSFORMER BUILDING BLOCKS
#-----------------------------------------------------------------------------------
def get_circuits(state): #state is a class containing the NN and data hyperparameters
    p=state.parameters
    att_hyperparams = (p.d_model, p.d_key, p.nheads)
    ff_hyperparams = [(p.d_model, p.d_hid, True, True), (p.d_hid, p.d_model, True, False)]
    return lay.attention(att_hyperparams, p.d_model, p.attention_dropout, p.resnorm_dropout).to(state.device), lay.FeedForward(ff_hyperparams, p.d_model, p.feedforward_dropout, p.resnorm_dropout).to(state.device)
         

def get_transformer_parts(state): #state is a class containing the NN and data hyperparameters
    p= state.parameters
    att, ff = get_circuits(state)
    
    class TransformerParts:
        attention = att
        feedforward = ff
        encoder = lay.Encoder(p.ntokens, p.d_model, p.nlayers, attention, feedforward).to(state.device)
        decoder = lay.Decoder(p.ntokens, p.d_model, p.nlayers, attention, feedforward).to(state.device)
        linear = lay.linear_layer((p.d_model, p.ntokens_out, False, False)).to(state.device)
    return TransformerParts()



#-----------------------------------------------------------------------------------
#GET OPTIMIZER AND SCHEDULER
#-----------------------------------------------------------------------------------
def get_optimizer(state, model):
    if state.training.optimizer == "adam":
        opt = optim.Adam(model.parameters(), state.training.lr, betas=(0.9, 0.98), eps=1e-9)
    elif state.training.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), state.training.lr)
    else:
        print("Optimizer choice not recognized")
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, state.training.schedule)
    return opt, scheduler


#-----------------------------------------------------------------------------------
# example of a learning rate schedule used in Annotated Transformer
def learning_rate_function(model_size, factor, warmup_steps):
    return lambda epoch : factor* (model_size)**(-0.5) * min((epoch+1)**(-0.5), (epoch+1) * (warmup_steps)**(-1.5)) 

def learning_rate_step(factor, drop, time):
    return lambda epoch : factor/(drop**(epoch//time))



#-----------------------------------------------------------------------------------
#NETWORK FUNCTIONS
#-----------------------------------------------------------------------------------
class Network_functions(nn.Module):
    def __init__(self, model, load_saved=None): #learning_rate_schedule must be a lambda function
        nn.Module.__init__(self)
        self.model = model
        if load_saved is not None:
            self.load_model(load_saved)

    
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





