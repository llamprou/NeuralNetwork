import torch
from torch import nn
from data import text as txt
import time



class Model_training(nn.Module):
    def __init__(self, model, opt, loss):
        super().__init__()
        self.model = model
        self.opt =opt
        self.loss = loss
    

    def run_transformer_epoch(self, train_dataloader, tokenizer, device):
        self.model.train()
        train_iterator = iter(train_dataloader)
        total_loss =0
        t1= time.time()
        for input_batch, output_batch in train_iterator:
            _, out_prob = self.model(input_batch)

            #Convert output words from integers to one-hot vectors ---I could do this separately and define a dictionary to increase speed
            expectation = torch.tensor([[tokenizer.token_onehot[int(seq_elem)] for seq_elem in batch_elem] for batch_elem in output_batch], dtype=torch.float).to(device)

            loss = self.loss(out_prob, expectation).to(device)
            total_loss += loss
            
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        t2= time.time() 
        return t2-t1, total_loss   



    def evaluate_transformer_output(self, tokenizer, sample_input):
        self.model.eval()
        text_instance_encoded = self.model.autoregression(sample_input, 62)
        if tokenizer is not None:
            output = tokenizer.text_decoding(text_instance_encoded)
        else:
            output = text_instance_encoded
        return output
    

    def fit_transformer(self, epochs, train_dataloader, test_dataloader=None, tokenizer = None, sample_input=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        for epoch in range(epochs):
            dt, epoch_loss = self.run_transformer_epoch(train_dataloader, tokenizer, device)
            print(f"Epoch: {epoch} -- time: {dt} -- loss = {epoch_loss}\n")

            output = self.evaluate_transformer_output(tokenizer, sample_input)
            print(output)


            
            #compute loss on test data
            #if test_dataloader is not None:
            #    test_iterator = iter(test_dataloader)
            #    accuracy=0
            #    with torch.no_grad():
            #        self.model.eval()
            #        for exp, pred in test_iterator:
            #            accuracy += self.evaluate((exp, pred)) #Has not been defined yet
            #    print(f"Epoch {epoch} achieved accuracy {accuracy/len(test_dataloader)}")