import torch
import numpy as np
from torch import nn
import copy
import math



#define linear layer class
class linear_layer(nn.Module):
    def __init__(self, hyper_param): #(inp_dim, hid_dim, bias_is_true, relu_is_true)
        super().__init__()
        (self.inp_dim, self.hid_dim, self.bias_is_true, self.relu_is_true) = hyper_param
        self.weight = nn.Parameter(torch.randn(self.inp_dim, self.hid_dim)/torch.sqrt(torch.tensor(self.inp_dim)))
        if self.bias_is_true:
            self.bias = nn.Parameter(torch.randn(self.hid_dim))
        self.relu =nn.ReLU()
        
    def forward(self, input):
        output= torch.tensordot(input, self.weight,  dims = ([-1],[0]) ) 
        if self.bias_is_true:
            output+= self.bias
        if self.relu_is_true:
            output = self.relu(output)
        return output





#-----------------------------------------------------------------------------------
#define FeedForward class
class FeedForward(nn.Module):
    def __init__(self, hyper_param): #([(dim_in, dim_out, bias_is_true, relu_is_true) ,(), ... ])
        nn.Module.__init__(self) 
        self.hyper_param = hyper_param
        self.layers = nn.ModuleList([linear_layer(param) for param in self.hyper_param])

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


#---------------------------------------------------------------------------------
#multiheaded attention class

class attention(nn.Module):
    def __init__(self, hyper_param): #(dim_in, dim_key, dim_heads)
        nn.Module.__init__(self) 
        (self.dim_in, self.dim_key, self.heads) = hyper_param
        self.attention = nn.ModuleList([linear_layer((self.dim_in, self.dim_key*self.heads, False, False)) for _ in range(3)])
        self.final = linear_layer((self.heads * self.dim_key, self.dim_in, False, False))
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, inputs, mask=None):
        q,k,v = tuple(layer(inp).view(inp.size(0), inp.size(1), self.heads, self.dim_key).transpose(1,2) for inp, layer in zip(inputs, self.attention))
        score = torch.matmul(q,k.transpose(-1,-2)).masked_fill(mask==0, -1e9) if mask is not None else torch.matmul(q,k.transpose(-1,-2))
        p_atten = self.dropout(self.softmax(score/torch.sqrt(torch.tensor(self.dim_key))))
        preactivation = torch.matmul(p_atten,v).transpose(1,2).reshape(v.size(0), -1, self.heads*self.dim_key)
        return self.final(preactivation)

#---------------------------------------------------------------------------------
#define layer norm class
#The Annotated transformer adds 2 extra learnable parameters in this layer --I don't.
class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon= 10**(-7)


    def forward(self, input):
        mean = torch.mean(input, dim= -1, keepdim= True)
        std = torch.std(input, dim=-1, keepdim= True) + self.epsilon
        output = (input - mean)*(1/std)
        return output






#-----------------------------------------------------------------------------------
#Skip connection and layer normalization decorator + dropout 
def SkipAndNormalize_decorator(cls):
    class ResNorm_wrapper(nn.Module):
        def __init__(self, hyper_param):
            super().__init__()
            self.layers = nn.ModuleList([cls(hyper_param), LayerNorm()])
            self.dropout = nn.Dropout(0.1)

        
        def forward(self, residual_stream, *input):
            h1 = self.layers[0](*input) + residual_stream
            return self.dropout(self.layers[1](h1))
    return ResNorm_wrapper    



@SkipAndNormalize_decorator
class SkipAttention(attention):
    def __init__(self, hyper_param): #(dim_in, dim_key, dim_heads)
        super().__init__(hyper_param)
        



@SkipAndNormalize_decorator
class SkipFeedForward(FeedForward):
    def __init__(self, hyper_param): #[(dim_in, dim_out, bias_is_true, relu_is_true),(),()]
        super().__init__(hyper_param)
        


#-----------------------------------------------------------------------------------
#Helper function for getting attention and FF blocks for building En/De-coders, Transformers
def get_coder_blocks(dim_in, dim_key, heads, dim_internal, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    att_hyperparams = (dim_in, dim_key, heads)
    ff_hyperparams = [(dim_in, dim_internal, True, True), (dim_internal, dim_in, True, False)]
    return SkipAttention(att_hyperparams).to(device), SkipFeedForward(ff_hyperparams).to(device)



#-----------------------------------------------------------------------------------
#define encoder class
class Encoder(nn.Module):
    def __init__(self, skip_attention, skip_feedforward):
        super().__init__()
        self.layers = nn.ModuleList([skip_attention, skip_feedforward])

    
    def forward(self, input, mask=None): 
        inputs = [input for _ in range(3)]
        h1= self.layers[0](input, inputs, mask)
        output = self.layers[1](h1, h1)
        return output



# define decoder class
class Decoder(nn.Module):
    def __init__(self, skip_attention, skip_feedforward):
        super().__init__()
        self.layers = nn.ModuleList([skip_attention, copy.deepcopy(skip_attention), skip_feedforward ])
    

    def forward(self, input, enc_mask=None, dec_mask=None): #input = list [encoder_output, decoder_input]
        encoder_output = input[0]
        decoder_input = input[1]
        h1 = self.layers[0](decoder_input, [decoder_input for _ in range(3)], dec_mask)
        h2 = self.layers[1](h1, [h1, encoder_output, encoder_output], enc_mask)
        output = self.layers[2](h2, h2)
        return [encoder_output, output]




#-----------------------------------------------------------------------------------
#define encoder/decoder stack decorator
def get_stack(cls):
    class Stack_wrapper(nn.Module):
        def __init__(self, skip_attention, skip_feedforward, copies):
            super().__init__()
            blocks = [[copy.deepcopy(skip_attention), copy.deepcopy(skip_feedforward)] for _ in range(copies)]
            self.layers = nn.ModuleList([cls(*block) for block in blocks])

        def forward(self, input, *masks): #there can be 1 or 2 masks depending on whether we are stacking encoders or decoders
            output = input
            for layer in self.layers:
                output = layer(output, *masks)
            return output
    return Stack_wrapper
            

@get_stack
class EncoderStack(Encoder):
    def __init__(self, skip_attention, skip_feedforward):
        super().__init__(skip_attention, skip_feedforward)


@get_stack
class DecoderStack(Decoder):
    def __init__(self, skip_attention, skip_feedforward):
        super().__init__(skip_attention, skip_feedforward)





#-----------------------------------------------------------------------------------
#transformer layer class
class TransformerLayer(nn.Module):
    def __init__(self, skip_attention, skip_feedforward, copies, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        nn.Module.__init__(self) 
        self.layers = nn.ModuleList([EncoderStack(skip_attention, skip_feedforward, copies).to(device), DecoderStack(copy.deepcopy(skip_attention), copy.deepcopy(skip_feedforward), copies).to(device)])


    def forward(self, inputs, enc_mask=None, dec_mask=None): #inputs is a list [encoder input, decoder input]
        h1 = self.encode(inputs[0], enc_mask)
        return self.decode([h1, inputs[1]], enc_mask, dec_mask)


    def encode(self, input, enc_mask=None):
        return self.layers[0](input, enc_mask)

    def decode(self, inputs, enc_mask=None, dec_mask=None):
        output = self.layers[1](inputs, enc_mask, dec_mask)
        return output[1]



#-----------------------------------------------------------------------------------
#Positional encoding class
class Positional_enc(nn.Module):
    def __init__(self, dim_in, max_dim=500):
        nn.Module.__init__(self)
        self.dim_in, self.max_dim = dim_in, max_dim
        #construct positional encoding for single batch element
        argument = torch.tensordot(torch.arange(max_dim, dtype=torch.float), torch.exp(-math.log(1000) *torch.arange(0, dim_in, 2, dtype=torch.float)/dim_in) , dims= 0)
        pos_enc= torch.empty(max_dim, dim_in)
        pos_enc[:, 0::2] = torch.sin(argument)
        pos_enc[:, 1::2] = torch.cos(argument)
        #introduce batch dimension (=0) 
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        input =input + self.pos_enc[:, :input.size(1), :].requires_grad_(False)
        return self.dropout(input)


#-----------------------------------------------------------------------------------
#Mask
def construct_mask(size, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    uppertri = torch.triu(torch.ones(1, size, size), diagonal=1)
    return (uppertri ==0).to(device)




#-----------------------------------------------------------------------------------
#include positional encoding, masking, and softmax
class Transformer_model(nn.Module):
    def __init__(self, position_enc, enc_embedding, dec_embedding, transformer, linear, start, end):
        nn.Module.__init__(self) 
        self.layers = nn.ModuleList([enc_embedding, dec_embedding, transformer, linear])
        self.position_enc = [position_enc, copy.deepcopy(position_enc)]
        self.softmax = nn.Softmax(dim=-1)
        self.start = start
        self.end = end

    def embed_encode(self, input, enc_mask=None):
        h1 = self.layers[0](input)
        encoder_input = self.position_enc[0](math.sqrt(h1.size(-1))*h1)
        enc_output = self.layers[2].encode(encoder_input, enc_mask)
        return enc_output 
    
    def embed_decode(self, enc_output, dec_input, enc_mask=None): #inputs is a list [enc_input, dec_input]
        h2 = self.layers[1](dec_input)
        h2= self.layers[2].decode([enc_output,  self.position_enc[1](math.sqrt(h2.size(-1))*h2 )], enc_mask, construct_mask(h2.size(1)) ) 
        return self.layers[3](h2)

    def forward(self, inputs, enc_mask=None): #inputs = [enc_input, dec_input] returns a tuple of outputs and probabilities
        enc_output = self.embed_encode(inputs[0])
        presoftmax_out = self.embed_decode(enc_output, inputs[1], enc_mask)
        probabilities = self.softmax(presoftmax_out) 
        output_seq = torch.argmax(probabilities, dim =-1)
        return output_seq, presoftmax_out


    def autoregression(self, input, length, enc_mask = None, mode= "greedy"): #input= encoder_input (1 x seq_length)
        enc_output = self.embed_encode(input) 
        inp = self.start #(batch_size x seq_length) 
        for _ in range(length):
            prob = self.softmax(self.embed_decode(enc_output, inp, enc_mask)) 
            
            # Greedy inference
            if (mode == "greedy"):
                out = torch.argmax(prob, dim =-1)
                next_word = (torch.tensor([out[:, -1]]).unsqueeze(0)).to(out, non_blocking=True)
                inp = torch.cat([inp, next_word], dim=1)
                if next_word == self.end:
                    break
                else: 
                    continue
            
            # Beam search using Bayesian inference for next two words A_1, A_2 
            # Our strategy is: Starting from the two most likely tokens for A_1, we predict the most likely next token A_2max(A_1) given each choice and then select the A_1 that maximizes P(A_1)*P(A_2max | A_1)
            # In this way, the next word A_1 is chosen so that it maximizes P(A_1 U A_2)
            elif (mode == "beam"):
                largest_two_probabilities, likely_words = torch.topk(prob[:,-1], 2, dim= -1)
                predicted_batch = torch.cat([inp, inp], dim =0)
                predicted_batch=torch.cat([predicted_batch, likely_words.transpose(0,1)], dim=1)
                new_enc_output = torch.cat([enc_output]*2, dim=0)
                h = self.softmax(self.embed_decode(new_enc_output, predicted_batch.type(torch.long), enc_mask))[:,-1] 
                next_probabilities, _ = torch.max(h.detach(), dim = -1)

                most_likely_word_idx =(torch.argmax(largest_two_probabilities.squeeze()* next_probabilities)).item()
                next_word = likely_words[:, most_likely_word_idx].unsqueeze(0)
                inp = torch.cat([inp, next_word], dim=1)
                if next_word == self.end:
                    break
                else: 
                    continue
        return inp
        

        