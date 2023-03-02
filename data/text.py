import torch
import numpy as np
import data.data_loading as dat

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#Word level tokenizer and text encoder/decoder
class My_word_tokenizer():
  def __init__(self, text):
    self.punctuations = list(".,?:[]()!;/,-_}{")
    if (type(text)!=str):
      text = " ".join([elem for elem in text])
    transformed_text = self.punctuation(text, "split")
    self.tokens = list(set(transformed_text.split(" "))) 
    self.token_encoding = {token : torch.tensor([k]).unsqueeze(0) for k, token in enumerate(self.tokens)}
    self.token_onehot ={k : [1 if l ==k else 0 for l in range(len(self.tokens))] for k in range(len(self.tokens))}
      
    
  #method for placing gaps before punctuation signs so that they will tokenized like words during encoding
  #and also for removing same gaps before text decoding.
  def punctuation(self, text, mode):
    for p in self.punctuations:
      if mode == "split":
        text=text.replace(p, " "+p) 
      elif mode == "join":
        text = text.replace(" "+p , p)
      else:
        print("Mode selection error")
    return text 


  def text_encoding(self, text):
    text = (self.punctuation(text, "split")).split(" ")
    return torch.cat([self.token_encoding[word] for word in text], dim=1)


  #greedy decoding of the text
  def text_decoding(self, output):
    text = " ".join(map(str, [list(self.token_encoding.keys())[word.detach()] for word in output.squeeze()]))
    text = self.punctuation(text, "join")
    return text



#Pytorch library tokenizer and text en/decoder
class library_text_coders:
    def __init__(self, train_iter):
        self.train_iter = train_iter
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

    def list_text_encoding(self, text_iter):
        encoded = [torch.tensor(self.vocab(self.tokenizer(item))) for item in text_iter]
        return encoded
    
    def text_encoding(self, text_data):
        return torch.tensor(self.vocab(self.tokenizer(text_data)))

    def text_decoding(self, input):
        return " ".join(map(lambda t: self.vocab.lookup_token(t.item()), input))
    
    def text_merging(self, encoded_text_list):
        return torch.cat(tuple(filter(lambda t: t.numel()>0, encoded_text_list)))




# input/output texts are expected to be lists of text pieces, like sentences/paragraphs/books to be translated, etc. 
# All list elements are assumed to be word sequences of the same length. If not, a fake token needs to be introduced to ensure that.
def get_tranformer_data(input_text, output_text, batch_size, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
  tokenizer_in = My_word_tokenizer("".join(input_text))
  tokenizer_out = My_word_tokenizer("".join(output_text))
  encoded_input = torch.cat([tokenizer_in.text_encoding(in_piece) for in_piece in input_text], dim = 0)
  encoded_output = torch.cat([tokenizer_out.text_encoding(out_piece) for out_piece in output_text], dim = 0)
  dataloader = dat.get_tranformer_dataloader(encoded_input, encoded_output, batch_size, True)
  start = tokenizer_out.token_encoding["<start>"].to(device)
  end = tokenizer_out.token_encoding["<end>"].to(device)
  return tokenizer_in, tokenizer_out, dataloader, start, end



def transformer_text_generator(model, tokenizer_in, tokenizer_out, length, encoder_input):
  encoder_input = tokenizer_in.text_encoding(encoder_input)
  text_instance_encoded = model.autoregression(encoder_input, length)
  text_instance = tokenizer_out.text_decoding(text_instance_encoded)
  print(text_instance)



#-----------------------------------------------------------------------------------
#Character tokenizer and text encoder/decoder for LSTM
class My_char_tokenizer(object):
  def __init__(self, text):
    self.tokens = list(set(text))
    self.tokenizer = { letter : torch.tensor(k) for k, letter in enumerate(self.tokens)}


  def text_encoding(self, text):
    data=[]
    for char in text:
      elem = torch.zeros(len(self.tokens)).scatter_(0, self.tokenizer[char], 1 )
      data.append(elem)  
    input_tensor = torch.stack(tuple(data), dim =0)
    return input_tensor


  #Probabilistic decoding of the text
  def text_decoding(self, input):
    text_output= ""
    input_det = input.detach()    
    for k in range(len(input_det)):
      inp_list = input_det[k].tolist()
      probability = [elem/sum(inp_list) for elem in inp_list] #This renormalization of probability is needed due to a truncation error in going from torch.tensors to lists. It is such a small difference that is irrelevant.
      sampled_char = np.random.choice(list(self.tokenizer.keys()), p= probability)
      text_output += sampled_char 
    return text_output



#Text generator
def generate_text_lstm(model, tokenizer, start, length, forget_gate=True):
  start_vec = tokenizer.tokenizer[start]
  text_instance_encoded = model.generate_prob(start_vec, length, forget_gate)
  text_instance = tokenizer.text_decoding(text_instance_encoded)
  print(text_instance)



def get_lstm_data(text, bs):
  tokenizer = My_char_tokenizer(text)
  train_dataloader = dat.Dataloader_iter(text[:-1], text[1:], batch_size=bs, shuffle_batch=True, inp_transformation= tokenizer.text_encoding, out_transformation= tokenizer.text_encoding)
  return tokenizer, train_dataloader




