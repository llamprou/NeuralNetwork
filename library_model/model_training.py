import torch
from torch import nn
from data import text as txt
import time
import os
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

class Records_class:
  def __init__(self, **param_logs):
    self.parameters={p : [] for p in param_logs.keys()}
    self.logs=param_logs

class Training_records:
  def __init__(self, train_log, test_log, records):
    self.train_loss = []
    self.test_loss = []
    self.accuracy = []
    self.test_log = test_log
    self.train_log = train_log
    self.records = records
    self.lr =[]

  def record(self, step, lr, dt, loss):
    self.lr.append(lr)
    if step%self.train_log ==0:
      self.train_loss.append(loss/self.train_log)
      print(f"Batch {step} | lr = {lr:.3f} | time = {(dt)* 5:5.2f} | train_loss {loss/self.train_log:5.2f}")
      loss =0
    if step%self.test_log ==0:
      a = self.evaluate()
      self.accuracy.append(a)

    if self.records is not None:
      for param in self.records.parameters.keys():
        if step%self.records.logs[param] ==0:
          self.records.parameters[param] = self.records.parameters[param] +[self.model.state_dict()[param].detach().clone()]
    return loss


  def plot_loss(self):
    plt.figure(figsize=(9,3))
    
    plt.subplot(1,2,1)
    plt.plot(range(len(self.train_loss)), self.train_loss, "ro")
    plt.ylabel("training loss")
    plt.xlabel("batch/"+str(self.train_log))
    
    plt.subplot(1,2,2)
    plt.plot(range(len(self.test_loss)), self.test_loss, "ro")
    plt.ylabel("test loss")
    plt.xlabel("batch/"+str(self.test_log))
    
    plt.show()


class Model_training(nn.Module, Training_records):
    def __init__(self, model, opt, loss, scheduler, train_log = 200, test_log = 400, records=None):
        nn.Module.__init__(self)
        Training_records.__init__(self, train_log, test_log, records)
        self.model = model
        self.opt =opt
        self.loss = loss
        self.scheduler = scheduler
        self.min_loss = None
    

    def forward(self, epochs, data):
        self.data = data
        best_loss = 10**5
        with TemporaryDirectory() as tempdir:
            best_model_path = os.path.join(tempdir, "best_model.pt")
            for epoch in range(epochs):
                t1= time.time()
                self.run_transformer_epoch()
                t2= time.time() 
                print(f"Epoch: {epoch} | time: {t2-t1:5.2f} | test_loss = {self.test_loss[-1]:5.2f}\n")
                if self.test_loss[-1]< best_loss:
                    best_loss = self.test_loss[-1]
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
            total_loss = self.record(k, self.scheduler.get_last_lr()[0], time.time()-t_prev, total_loss)
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
    



class CNN_training(nn.Module, Training_records):
  def __init__(self, data, state, model, opt, scheduler, train_log=15, test_log=25, records =None):
    nn.Module.__init__(self)
    Training_records.__init__(self, train_log, test_log, records)
    self.d = data
    self.model = model
    self.s = state
    self.criterion = nn.CrossEntropyLoss()
    self.opt = opt
    self.scheduler = scheduler



  def run_epoch(self):
    self.model.train()
    train_iterator = iter(self.d.train_dataloader)
    total_loss =0
    k = 0
    tin=time.time()
    for input_batch, output_batch in train_iterator:
      t1 = time.time()
      out_prob = self.model(input_batch.to(self.s.device))
      out_true = torch.argmax(output_batch, dim=-1).to(self.s.device)
      loss = self.criterion(out_prob.view(-1, out_prob.size(-1)), out_true)
      total_loss += loss.item()
      
      self.opt.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) #prevents blow-ups of backpropagated derivatives
      self.opt.step()
      
      k+=1
      lr = self.scheduler.get_last_lr()[0]
      total_loss = self.record(k, lr, time.time()-t1, total_loss)
      self.scheduler.step()
    tfin= time.time() 
    return tfin-tin, total_loss   

  def evaluate(self):
    self.model.eval()
    test_iter = iter(self.d.test_dataloader)
    accuracy =0
    total_loss = 0
    number = 1
    for Xt, yt in test_iter:
      if (number < 20):
        out = self.model(Xt.to(self.s.device))
        out_true = torch.argmax(yt, dim=-1).to(self.s.device)
        loss = self.criterion(out.view(-1, out.size(-1)), out_true)
        total_loss += loss.item()
        out_pred = torch.argmax(out, dim=-1)
        number +=1
        accuracy += (sum(out_pred.view(-1)==out_true)/len(out_pred.view(-1))).item()
    self.test_loss.append(total_loss/number)
    return accuracy/number
  

  def forward(self, epochs):
    for epoch in range(epochs):
      dt, _ = self.run_epoch()
      print(f"Epoch: {epoch} | time: {dt:.2f} -- test loss = {self.test_loss[-1]:.2f} -- accuracy = {self.accuracy[-1]:.2f}\n")
    self.plot_loss()      


