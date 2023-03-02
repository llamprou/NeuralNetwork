import torch
import numpy as np
import random




#Building an iterator over minibatches of inputed text after tensor encoding
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



#class for reading text file inputs and saving text outputs

