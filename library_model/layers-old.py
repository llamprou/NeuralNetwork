import torch
import numpy as np
from torch import nn
from itertools import chain
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
        output= torch.tensordot(input, self.weight,  dims = ([1],[0]) ) 
        if self.bias_is_true:
            output+= self.bias
        if self.relu_is_true:
            output = self.relu(output)
        return output




#-----------------------------------------------------------------------------------
#Auxiliary parent class for collecting parameters in a generator  
class parameter_generator(object):
    def __init__(self, layers):
        self.layers = layers
        

    def parameters(self):
        generator = self.layers[0].parameters()
        for layer in self.layers[1:]:
            generator = chain(generator, layer.parameters())

        for elem in generator:
            yield elem
            


#-----------------------------------------------------------------------------------
#define FeedForward class
class FeedForward(parameter_generator, nn.Module):
    def __init__(self, hyper_param): #([(dim_in, dim_out, bias_is_true, relu_is_true) ,(), ... ])
        nn.Module.__init__(self) 
        self.hyper_param = hyper_param
        self.layers = [linear_layer(param) for param in self.hyper_param]
        parameter_generator.__init__(self, self.layers) 

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output



#-----------------------------------------------------------------------------------
#define attention class
class attention(parameter_generator,nn.Module):
    def __init__(self, hyper_param): #(dim_in, dim_key, dim_heads)
        nn.Module.__init__(self) 
        (self.dim_in, self.dim_key, self.heads) = hyper_param
        attention_matrices = [linear_layer((self.dim_in, self.dim_key, False, False)) for _ in range(3*self.heads)]
        self.layers = [*attention_matrices, linear_layer((self.heads * self.dim_key, self.dim_in, False, False))]

        parameter_generator.__init__(self, self.layers)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.1)
    
    def activation_generator(self, inputs, mask=None):
        for k in range(self.heads):
            score = torch.tensordot(self.layers[3*k](inputs[0]), self.layers[3*k+1](inputs[1]), dims = ([1],[1]))/(torch.sqrt(torch.tensor(self.dim_key)))
            if mask is not None:
                score= score.masked_fill(mask==0, -1e9)
            score = self.dropout(self.softmax(score))
            value = torch.tensordot(score, self.layers[3*k+2](inputs[2]), dims = ([1],[0]))
            yield value


    def forward(self, inputs, mask=None): #inputs is list of [query, key, value]
        activations = self.activation_generator(inputs, mask)
        output = torch.cat([act for act in activations], dim=1)
        return self.layers[-1](output)
        

    def parameter_dictionary(self):
        auxiliary = {0 : "query", 1 : "key", 2 : "value"}
        dictionary = {"head" + str(int(k / 3)) + "." + auxiliary[int(k % 3)] + ".weight" : p for k, p in enumerate(self.parameters())}
        return dictionary






#---------------------------------------------------------------------------------
#define layer norm class
class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon= 10**(-7)


    def forward(self, input):
        mean = torch.mean(input, dim= 1, keepdim= True)
        std = torch.std(input, dim=1, keepdim= True) + self.epsilon
        output = (input - mean)*(1/std)
        return output






#-----------------------------------------------------------------------------------
#Skip connection and layer normalization decorator + dropout
def SkipAndNormalize_decorator(cls):
    class ResNorm_wrapper(parameter_generator):
        def __init__(self, hyper_param):
            self.layers = [cls(hyper_param), LayerNorm()]
            self.dropout = nn.Dropout(0.1)
            parameter_generator.__init__(self, self.layers)

        
        def __call__(self, residual_stream, *input):
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
#Get attention and FF blocks for building En/De-coders, Transformers
def get_coder_blocks(dim_in, dim_key, heads, dim_internal):
    att_hyperparams = (dim_in, dim_key, heads)
    ff_hyperparams = [(dim_in, dim_internal, True, True), (dim_internal, dim_in, True, False)]
    return SkipAttention(att_hyperparams), SkipFeedForward(ff_hyperparams)



#-----------------------------------------------------------------------------------
#define encoder class
class Encoder(parameter_generator):
    def __init__(self, skip_attention, skip_feedforward):
        self.layers = [skip_attention, skip_feedforward]
        parameter_generator.__init__(self, self.layers)
    
    def __call__(self, input, mask=None): 
        inputs = [input for _ in range(3)]
        h1= self.layers[0](input, inputs, mask)
        output = self.layers[1](h1, h1)
        return output



# define decoder class
class Decoder(parameter_generator):
    def __init__(self, skip_attention, skip_feedforward):
        self.layers = [skip_attention, copy.deepcopy(skip_attention), skip_feedforward ]
        parameter_generator.__init__(self, self.layers)
    

    def __call__(self, input, enc_mask=None, dec_mask=None): #input = list [encoder_output, decoder_input]
        encoder_output = input[0]
        decoder_input = input[1]
        h1 = self.layers[0](decoder_input, [decoder_input for _ in range(3)], dec_mask)
        h2 = self.layers[1](h1, [h1, encoder_output, encoder_output], enc_mask)
        output = self.layers[2](h2, h2)
        return [encoder_output, output]




#-----------------------------------------------------------------------------------
#define encoder/decoder stack decorator
def get_stack(cls):
    class Stack_wrapper(parameter_generator):
        def __init__(self, skip_attention, skip_feedforward, copies):
            blocks = [[copy.deepcopy(skip_attention), copy.deepcopy(skip_feedforward)] for _ in range(copies)]
            self.layers = [cls(*block) for block in blocks]
            parameter_generator.__init__(self, self.layers)

        def __call__(self, input, *masks): #there can be 1 or 2 masks depending on whether we are stacking encoders or decoders
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
class TransformerLayer(parameter_generator, nn.Module):
    def __init__(self, skip_attention, skip_feedforward, copies):
        nn.Module.__init__(self) 
        self.layers = [EncoderStack(skip_attention, skip_feedforward, copies), DecoderStack(copy.deepcopy(skip_attention), copy.deepcopy(skip_feedforward), copies)]
        parameter_generator.__init__(self, self.layers)


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
    def __init__(self, dim_in, max_dim=5000):
        nn.Module.__init__(self)
        self.dim_in, self.max_dim = dim_in, max_dim
        argument = torch.tensordot(torch.arange(max_dim, dtype=torch.float), torch.exp(-math.log(10000) *torch.arange(0, dim_in, 2, dtype=torch.float)/dim_in) , dims= 0)
        self.pos_enc= torch.empty(max_dim, dim_in)
        self.pos_enc[:, 0::2] = torch.sin(argument)
        self.pos_enc[:, 1::2] = torch.cos(argument)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        input =input + self.pos_enc[:input.size(0),:].requires_grad_(False)
        return self.dropout(input)


#-----------------------------------------------------------------------------------
#Mask
def construct_mask(size):
    uppertri = torch.triu(torch.ones(size, size), diagonal=1)
    return uppertri ==0




#-----------------------------------------------------------------------------------
#include positional encoding, masking, and softmax
class Transformer_model(parameter_generator, nn.Module):
    def __init__(self, position_enc, enc_embedding, dec_embedding, transformer, linear, mode="greedy"):
        nn.Module.__init__(self) 
        self.mode = mode
        self.layers = [enc_embedding, dec_embedding, transformer, linear]
        self.position_enc = [position_enc, copy.deepcopy(position_enc)]
        parameter_generator.__init__(self, self.layers)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, inputs, enc_mask=None): #inputs = [enc_input, dec_input] returns a tuple of outputs and probabilities
        h1 = self.layers[0](inputs[0])
        encoder_input = self.position_enc[0](math.sqrt(h1.size(-1))*h1)
        enc_output = self.layers[2].encode(encoder_input, enc_mask)
        self.enc_output = enc_output  #saving encoder output for crosschecking purposes later
    
        h2 = self.layers[1](inputs[1])
        h2= self.layers[2].decode([enc_output,  self.position_enc[1](math.sqrt(h2.size(-1))*h2 )], enc_mask, construct_mask(h2.size(0)))
        self.prob = self.softmax(self.layers[3](h2)) 
        output = torch.argmax(self.prob, dim =1)
        return output


    def autoregression(self, inputs, enc_mask = None): #inputs = [enc_input, dec_input]
        batch_size = inputs[0].size(0)
        h1 = self.layers[0](inputs[0])
        encoder_input = self.position_enc[0](math.sqrt(h1.size(-1))*h1)
        enc_output = self.layers[2].encode(encoder_input, enc_mask)
        self.enc_output = enc_output  #saving encoder output for crosschecking purposes later
    
        prob_list=[]
        inp = inputs[1]
        for _ in range(batch_size):
            h2 = self.layers[1](inp)
            h2= self.layers[2].decode([enc_output,  self.position_enc[1](math.sqrt(h2.size(-1))*h2 )], enc_mask, construct_mask(inp.size(0)))
            prob = self.softmax(self.layers[3](h2)) 
            prob_list.append(prob[-1].unsqueeze(0))

            out = torch.argmax(prob, dim =1)
            next_word = torch.tensor([out[-1]])
            inp = torch.cat([inp, next_word], dim=0)
        self.prob = torch.cat(prob_list, dim=0) #saving probabilities for crosschecking later
        return inp
        

        