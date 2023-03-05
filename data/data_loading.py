import torch
import numpy as np
import random



#------------------------------------------------------------------------------------
#CLASS FOR STORING INPUT TEXT ITERATOR
#------------------------------------------------------------------------------------
class Data:
    def __init__(self):
      self.tokenizer = None
      self.train_dataloader= None
      self.test_dataloader = None
      self.sample_input = None

    def encode(self, input):
      if self.tokenizer is not None:
        return self.tokenizer.text_encoding(input)
      else:
        print("No tokenizer selection")

    def decode(self, input):
      if self.tokenizer is not None:
        return self.tokenizer.text_decoding(input)
      else:
        print("No tokenizer selection")



#------------------------------------------------------------------------------------
#HELPER FUNCTIONS FOR PROCESSING INPUT TEXT ITERATOR
#------------------------------------------------------------------------------------
def process_data(train_iter, coders_cls, state, network = "transformer"):
    tr = state.training
    coders = coders_cls(train_iter)  #generates vocab, contains tokenizer, text encoders and decoders
    text_data = " ".join(list(item for item in train_iter))  #merges text items of Wikitext2 generator to form single text
    train_data = coders.text_encoding(text_data).view(-1).to(state.device)  #encodes text into flat tensor and sends it to device
    train_dl, test_dl = get_dataloaders(train_data[:int(train_data.size(0)*tr.data_fraction)], tr.seq_length, tr.batch_size, tr.data_split, network) 
    return coders, train_dl, test_dl

def get_dataloaders(data, seq_length, batch_size, data_split, network):
    data = data[:(len(data)//seq_length)*seq_length].view(-1,seq_length) #organizes flat data tensor to fixed-length sequences
    train_dl = get_encoder_dataloader(data[:int(data.size(0)*data_split),:], data[:int(data.size(0)*data_split),:], batch_size, shuffle_batch=True) if network == "encoder" else get_tranformer_dataloader(data[:int(data.size(0)*data_split),:], data[:int(data.size(0)*data_split),:], batch_size, shuffle_batch=True)
    test_dl = get_encoder_dataloader(data[int(data.size(0)*data_split):,:], data[int(data.size(0)*data_split):,:], batch_size, shuffle_batch=True) if network == "encoder" else get_tranformer_dataloader(data[int(data.size(0)*data_split):,:], data[int(data.size(0)*data_split):,:], batch_size, shuffle_batch=True)
    return train_dl, test_dl



#------------------------------------------------------------------------------------
#HELPER FUNCTIONS FOR GETTING TRANSFORMER DATALOADERS
#------------------------------------------------------------------------------------

#transformer dataloader
def get_tranformer_dataloader(*args, **kwargs):
  generator = Dataloader_iter(*args, **kwargs)
  class new_generator():
    def __init__(self, generator):
      self.generator = generator
      self.len = len(generator)
    
    def __iter__(self):
      self.iter_idx = 0
      iterator = iter(self.generator)
      while self.iter_idx < self.len:
        x,y = next(iterator)
        yield ([x, y[:, :-1]], y[:, 1:])
        self.iter_idx +=1
    def __len__(self):
      return self.len

  return new_generator(generator)


  #encoder dataloader
def get_encoder_dataloader(*args, **kwargs):
  generator = Dataloader_iter(*args, **kwargs)
  class new_generator():
    def __init__(self, generator):
      self.generator = generator
      self.len = len(generator)
    
    def __iter__(self):
      self.iter_idx = 0
      iterator = iter(self.generator)
      while self.iter_idx < self.len:
        _,y = next(iterator)
        yield ([y[:, :-1]], y[:, 1:])
        self.iter_idx +=1
    def __len__(self):
      return self.len

  return new_generator(generator)



#------------------------------------------------------------------------------------
#GENERIC DATALOADER CLASS
#------------------------------------------------------------------------------------

class Dataloader_iter(object):
  def __init__(self, input, output, batch_size, shuffle_batch=False, inp_transformation=None, out_transformation=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if inp_transformation is not None:
      self.input = inp_transformation(input).to(device)
    else:
      self.input = input.to(device)

    if out_transformation is not None:
      self.output = out_transformation(output).to(device)
    else:
      self.output = output.to(device)
    
    self.shuffle = shuffle_batch
    self.batch_size = batch_size
    self.length= len(self.input)

  def __iter__(self):
    self.iter_idx = 0
    self.idx = [k for k in range(self.length//self.batch_size)]
    if self.shuffle:
      random.shuffle(self.idx)

    while (self.iter_idx < int(self.length/self.batch_size)):
      sample =  self.idx[self.iter_idx-1]
      yield self.input[sample * self.batch_size : (sample+1) * self.batch_size], self.output[sample * self.batch_size : (sample+1) * self.batch_size]
      self.iter_idx += 1

  def __len__(self):
    return int(self.length/self.batch_size)
      
 


