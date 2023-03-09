import torch
from torch import nn
from data import text as txt
import time
import os
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

class Model_training(nn.Module):
    def __init__(self, model, opt, loss, scheduler):
        super().__init__()
        self.model = model
        self.opt =opt
        self.loss = loss
        self.scheduler = scheduler
        self.min_loss = None

    

    def forward(self, epochs, data):
        self.data = data
        self.test_loss = []
        self.train_loss = []
        best_loss = 10**5
        with TemporaryDirectory() as tempdir:
            best_model_path = os.path.join(tempdir, "best_model.pt")
            for epoch in range(epochs):
                t1= time.time()
                self.run_transformer_epoch()
                t2= time.time() 
                epoch_test_loss = self.evaluate()
                print(f"Epoch: {epoch} | time: {t2-t1:5.2f} | test_loss = {epoch_test_loss:5.2f}\n")
                if epoch_test_loss< best_loss:
                    best_loss = epoch_test_loss
                    torch.save(self.model.state_dict(), best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.plot_loss()
        self.min_loss = min(self.test_loss)
        self.data = None


    def run_transformer_epoch(self):
        train_iterator = iter(self.data.train_dataloader)
        total_loss =0
        k = 0
        for input_batch, output_batch in train_iterator:
            self.model.train()
            t_prev = time.time()
            out_prob = self.model(*input_batch)
            loss = self.loss(out_prob.view(-1,out_prob.size(-1)), output_batch.reshape(-1))
            total_loss += loss.item()
            
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) #prevents blow-ups of backpropagated derivatives
            self.opt.step()
            
            k+=1
            lr = self.scheduler.get_last_lr()[0]
            if k%200 ==0:
                self.train_loss.append(total_loss/200)
                print(f"Batch {k} | lr = {lr:.3f} | time = {(time.time()-t_prev)* 5:5.2f} | train_loss {total_loss/200:5.2f}")
                total_loss =0
            if k%400 ==0:
                _ = self.evaluate()
            self.scheduler.step()
        #self.scheduler.step()
        


    def evaluate(self):
        self.model.eval()
        test_iterator = iter(self.data.test_dataloader)
        total_loss =0
        k=1
        for input_batch, output_batch in test_iterator:
            out_prob = self.model(*input_batch)
            loss = self.loss(out_prob.view(-1, out_prob.size(-1)), output_batch.reshape(-1))
            total_loss += loss.item()
            k+=1
        self.test_loss.append(total_loss/k)
        return total_loss/k


    #fix
    def evaluate_autoregression(self, sample_input, length=40):
        self.model.eval()
        text_instance_encoded = self.model.autoregression(sample_input, length)
        if self.tokenizer is not None:
            output = self.data.tokenizer.text_decoding(text_instance_encoded)
        else:
            output = text_instance_encoded
        return output
    

    def plot_loss(self):
        plt.figure(figsize=(9,3))
        
        plt.subplot(1,2,1)
        plt.plot([n for n in range(len(self.train_loss))], self.train_loss, "ro")
        plt.ylabel("training loss")
        plt.xlabel("batch/200")
        
        plt.subplot(1,2,2)
        plt.plot([n for n in range(len(self.test_loss))], self.test_loss, "ro")
        plt.ylabel("test loss")
        plt.xlabel("batch/400")
        
        plt.show()



class CNN_training(nn.Module):
  def __init__(self, data, state, model, opt, scheduler):
    super().__init__()
    self.d = data
    self.m = model
    self.s = state
    self.criterion = nn.CrossEntropyLoss()
    self.opt = opt
    self.scheduler = scheduler
    self.train_loss =[]
    self.test_loss =[]

  def run_epoch(self):
    self.m.train()
    train_iterator = iter(self.d.train_dataloader)
    total_loss =0
    k = 0
    for input_batch, output_batch in train_iterator:
      t1 = time.time()
      out_prob = self.m(input_batch.to(self.s.device))
      out_true = torch.argmax(output_batch, dim=-1).to(self.s.device)
      loss = self.criterion(out_prob.view(-1, out_prob.size(-1)), out_true)
      total_loss += loss.item()
      
      self.opt.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.m.parameters(), 0.5) #prevents blow-ups of backpropagated derivatives
      self.opt.step()
      
      k+=1
      lr = self.scheduler.get_last_lr()[0]
      if k%200 ==0:
        self.train_loss.append(total_loss/200)
        print(f"Batch {k} | lr = {lr:.2f} | time = {time.time()-t1:5.3} | train loss {total_loss/200:.2f}")
        total_loss =0
      if k%400 ==0:
        _ = self.evaluate()
      self.scheduler.step()
    t2= time.time() 
    return t2-t1, total_loss   

  def evaluate(self):
    self.m.eval()
    test_iter = iter(self.d.test_dataloader)
    accuracy =0
    total_loss = 0
    number = 1
    for Xt, yt in test_iter:
      if (number < 20):
        out = self.m(Xt.to(self.s.device))
        out_true = torch.argmax(yt, dim=-1).to(self.s.device)
        loss = self.criterion(out.view(-1, out.size(-1)), out_true)
        total_loss += loss.item()
        out_pred = torch.argmax(out, dim=-1)
        number +=1
        accuracy += (sum(out_pred.view(-1)==out_true)/len(out_pred.view(-1))).item()
    self.test_loss.append(total_loss/number)
    return accuracy/number
  
  def plot_loss(self):
    plt.figure(figsize=(9,3))
    
    plt.subplot(1,2,1)
    plt.plot(range(len(self.train_loss)), self.train_loss, "ro")
    plt.ylabel("training loss")
    plt.xlabel("batch/200")
    
    plt.subplot(1,2,2)
    plt.plot(range(len(self.test_loss)), self.test_loss, "ro")
    plt.ylabel("test loss")
    plt.xlabel("batch/400")
    
    plt.show()


  def forward(self, epochs):
    for epoch in range(epochs):
      dt, _ = self.run_epoch()
      a = self.evaluate()
      print(f"Epoch: {epoch} | time: {dt:.2f} -- test loss = {self.test_loss[-1]:.2f} -- accuracy = {a:.2f}\n")
    self.plot_loss()      
