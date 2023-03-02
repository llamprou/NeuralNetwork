import torch
from torch import nn
from data import text as txt
import time


class Model_training(nn.Module):
    def __init__(self, model, opt, loss, scheduler):
        super().__init__()
        self.model = model
        self.opt =opt
        self.loss = loss
        self.scheduler = scheduler

    

    def forward(self, epochs, data):
        self.data = data
        for epoch in range(epochs):
            dt, epoch_loss = self.run_transformer_epoch()
            print(f"Epoch: {epoch} -- time: {dt} -- loss = {epoch_loss}\n")

            #output = self.evaluate_transformer_output(tokenizer, sample_input)
            #print(output)
        self.data = None


    def run_transformer_epoch(self):
        self.model.train()
        train_iterator = iter(self.data.train_dataloader)
        total_loss =0
        t1= time.time()
        k = 0
        for input_batch, output_batch in train_iterator:
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
            if k%100 ==0:
              print(f"Batch {k} -- lr = {lr} -- time = {time.time()-t_prev} -- loss {loss}")
        self.scheduler.step()
        t2= time.time() 
        return t2-t1, total_loss   


    def evaluate_transformer_output(self, sample_input):
        self.model.eval()
        text_instance_encoded = self.model.autoregression(sample_input, 62)
        if self.tokenizer is not None:
            output = self.data.tokenizer.text_decoding(text_instance_encoded)
        else:
            output = text_instance_encoded
        return output
    



            
            #compute loss on test data
            #if test_dataloader is not None:
            #    test_iterator = iter(test_dataloader)
            #    accuracy=0
            #    with torch.no_grad():
            #        self.model.eval()
            #        for exp, pred in test_iterator:
            #            accuracy += self.evaluate((exp, pred)) #Has not been defined yet
            #    print(f"Epoch {epoch} achieved accuracy {accuracy/len(test_dataloader)}")