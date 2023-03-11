import torch
import numpy as np
from torch import nn
import copy
import math

#-----------------------------------------------------------------------------------
#A CLASS TO CONTAIN THE NETWORK STATE
#-----------------------------------------------------------------------------------
class Network_state:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        self.parameters = self.parameters_class()
        self.training = self.training_class()

    class parameters_class:
        def __init__(self):
            self.ntokens = None
            self.ntokens_out = None
            self.d_model = None
            self.nheads = None
            self.d_key = None
            self.d_hid = None
            self.nlayers = None
            self.attention_dropout =0.1
            self.feedforward_dropout =0.
            self.resnorm_dropout =0.1
            self.network = "encoder"


    class training_class:
        def __init__(self):
            self.lr = 1.
            self.w_decay = 0.
            self.batch_size = None
            self.seq_length = None
            self.optimizer = "sgd"
            self.schedule = None
            self.data_split = 0.8
            self.data_fraction = 1.
            self.w_decay = 0.


def param_count(state):
    p= state.parameters
    if p.network == "encoder":
        return (p.d_model*p.d_key*4* p.nheads* p.nlayers)+ (p.d_model*p.d_hid*2*p.nlayers)
    elif p.network == "transformer":
        return 3*(p.d_model*p.d_key*4* p.nheads* p.nlayers)+ 2*(p.d_model*p.d_hid*2*p.nlayers)
    else:
        print("Network type not recognized")


#-----------------------------------------------------------------------------------
#BUILD DECORATOR FOR RESIDUAL CONNECTION AND LAYER NORM
#-----------------------------------------------------------------------------------

#Skip connection and layer normalization decorator + dropout 
def SkipAndNormalize_decorator(cls):
    class ResNorm_wrapper(nn.Module):
        def __init__(self, hyper_param, size, layer_dropout, resnorm_dropout):
            super().__init__()
            self.layers = nn.ModuleList([cls(hyper_param, layer_dropout), LayerNorm(size)])
            self.dropout = nn.Dropout(resnorm_dropout)

        
        def forward(self, residual_stream, *input):
            h = self.layers[0](*input) + residual_stream
            return self.dropout(self.layers[1](h))
    return ResNorm_wrapper    


#-----------------------------------------------------------------------------------
#BUILD DECORATORS FOR STACKS, EMBEDDING AND POSITIONAL ENCODING
#-----------------------------------------------------------------------------------

#Stack decorator
def get_stack(cls):
    class Stack_wrapper(nn.Module):
        def __init__(self, copies, skip_attention, skip_feedforward):
            super().__init__()
            blocks = [[copy.deepcopy(skip_attention), copy.deepcopy(skip_feedforward)] for _ in range(copies)]
            self.layers = nn.ModuleList([cls(*block) for block in blocks])
            self.external_input= None  #This is used for the decoder stack which takes the encoder stack output as input 

        def forward(self, input, *masks): #there can be 1 or 2 masks depending on whether we are stacking encoders or decoders
            output = input
            for layer in self.layers:
                layer.encoder_output = self.external_input
                output = layer(output, *masks)
                layer.encoder_output = None
            return output
    return Stack_wrapper


#define embedding and position encoding decorator
def EmbedPosEncode(cls):
    class EmbedPosEncodeWrapper(nn.Module):
        def __init__(self, ntokens, d_model, nlayers, *blocks):
            nn.Module.__init__(self)
            self.cls = cls(nlayers, *blocks)
            self.pos_enc = Positional_enc(d_model, max_dim= 5000)
            self.embed=nn.Embedding(ntokens, d_model)
            self.external_input = None
        
        def forward(self, input, *masks):
            h = self.embed(input)
            h = self.pos_enc(math.sqrt(h.size(-1))*h)
            self.cls.external_input = self.external_input
            output = self.cls(h, *masks)
            self.cls.external_input = None
            return output
    return EmbedPosEncodeWrapper


#-----------------------------------------------------------------------------------
#BUILD THE BASIC LAYERS
#-----------------------------------------------------------------------------------


#Positional encoding class
class Positional_enc(nn.Module):
    def __init__(self, dim_in, max_dim=5000):
        nn.Module.__init__(self)
        self.dim_in, self.max_dim = dim_in, max_dim
        #construct positional encoding for single batch element
        argument = torch.tensordot(torch.arange(max_dim, dtype=torch.float), torch.exp(-math.log(10000) *torch.arange(0, dim_in, 2, dtype=torch.float)/dim_in) , dims= 0)
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


#---------------------------------------------------------------------------------
#define layer norm class
#Both the Annotated transformer and pytorch use 2 extra learnable parameters in this layer so I include them
class LayerNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.epsilon= 10**(-7)
        self.norm1 = nn.Parameter(torch.ones(size))
        self.norm2 = nn.Parameter(torch.zeros(size))


    def forward(self, input):
        mean = torch.mean(input, dim= -1, keepdim= True)
        std = torch.std(input, dim=-1, keepdim= True) + self.epsilon
        output = self.norm1*(input - mean)*(1/std) + self.norm2
        return output



#Convolutional-Pooling class
class ConvPool(nn.Module):
    def __init__(self, dim0, dim1, filter_dim, in_features, out_features, pooling): #stride is set to 1 and no padding
        super().__init__()
        self.dim0, self.dim1, self.filter_dim, self.indim, self.outdim, self.pool= dim0, dim1, filter_dim, in_features, out_features, int(pooling)
        self.filter = linear_layer((in_features*(filter_dim**2), out_features, True, False))
        self.relu = nn.ReLU()
        self.n_conv = self.dim0-self.filter_dim+1
        self.m_conv = self.dim1-self.filter_dim+1
        isometry = torch.zeros(self.n_conv, self.m_conv, self.filter_dim, self.filter_dim, self.dim0, self.dim1)
        i,j,n,m = np.ogrid[:self.n_conv, :self.m_conv, :self.filter_dim, :self.filter_dim]
        isometry[i,j,n,m, i+n, j+m] = 1
        self.register_buffer("isometry", isometry)

    def convolution(self, input): #input: (batch_dim, feature_dim, image_dim0*image_dim1), dims = (image_dim0, image_dim1)
        x = input.reshape(-1, int(self.indim), int(self.dim0), int(self.dim1))
        x = torch.tensordot(x, self.isometry, dims= ([-1,-2],[-1,-2]))
        x= x.view(x.size(0), x.size(1), self.n_conv*self.m_conv, self.filter_dim**2).transpose(-2,-3).reshape(x.size(0), self.n_conv*self.m_conv, -1)
        return self.filter(x).transpose(-1,-2)

    def pooling(self, input):
        w = int(self.pool)
        x = (input.view(-1, self.outdim, self.n_conv, self.m_conv))[:,:, :int((self.n_conv//w)*w), :int((self.m_conv//w)*w)] #pad down to appopriate size
        x = x.view(-1, self.outdim, self.n_conv//w, w, self.m_conv//w, w).transpose(-2,-3).reshape(-1, self.outdim, self.n_conv//w, self.m_conv//w, w**2)
        x_pool, _ = torch.max(x, dim=-1)
        return x_pool


    def forward(self, input): #flattened input
        out = self.relu(self.convolution(input))
        if self.pool>1:
            out = self.pooling(out)
        return out.view(out.size(0), out.size(1),-1)




#-----------------------------------------------------------------------------------
#define FeedForward class
@SkipAndNormalize_decorator
class FeedForward(nn.Module):
    def __init__(self, hyper_param, dropout=0.): #([(dim_in, dim_out, bias_is_true, relu_is_true) ,(), ... ])
        nn.Module.__init__(self) 
        self.hyper_param = hyper_param
        self.layers = nn.ModuleList([linear_layer(param) for param in self.hyper_param])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return self.dropout(output)


#---------------------------------------------------------------------------------
#multiheaded attention class
@SkipAndNormalize_decorator
class attention(nn.Module):
    def __init__(self, hyper_param, dropout=0.1): #(dim_in, dim_key, dim_heads)
        nn.Module.__init__(self) 
        (self.dim_in, self.dim_key, self.heads) = hyper_param
        self.attention = nn.ModuleList([linear_layer((self.dim_in, self.dim_key*self.heads, False, False)) for _ in range(3)])
        self.final = linear_layer((self.heads * self.dim_key, self.dim_in, False, False))
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
    
    #Pytorch combines q,k,v into a sinlge large tensor and separates them for computation using tensor views
    def forward(self, inputs, mask=None):
        q,k,v = tuple(layer(inp).view(inp.size(0), inp.size(1), self.heads, self.dim_key).transpose(1,2) for inp, layer in zip(inputs, self.attention))
        score = torch.matmul(q,k.transpose(-1,-2)).masked_fill(mask==0, -1e9) if mask is not None else torch.matmul(q,k.transpose(-1,-2))
        p_atten = self.dropout(self.softmax(score/torch.sqrt(torch.tensor(self.dim_key))))
        preactivation = torch.matmul(p_atten,v).transpose(1,2).reshape(v.size(0), -1, self.heads*self.dim_key)
        return self.final(preactivation)




#-----------------------------------------------------------------------------------
#define encoder class
@EmbedPosEncode
@get_stack
class Encoder(nn.Module):
    def __init__(self, skip_attention, skip_feedforward):
        super().__init__()
        self.layers = nn.ModuleList([skip_attention, skip_feedforward])

    
    def forward(self, input, mask=None): 
        inputs = [input for _ in range(3)]
        h= self.layers[0](input, inputs, mask)
        output = self.layers[1](h, h)
        return output



# define decoder class
@EmbedPosEncode
@get_stack
class Decoder(nn.Module):
    def __init__(self, skip_attention, skip_feedforward):
        super().__init__()
        self.layers = nn.ModuleList([skip_attention, copy.deepcopy(skip_attention), skip_feedforward ])
        self.encoder_output = None

    def forward(self, input, enc_mask=None, dec_mask=None): #input = list [encoder_output, decoder_input]
        h1 = self.layers[0](input, [input for _ in range(3)], dec_mask)
        h2 = self.layers[1](h1, [h1, self.encoder_output, self.encoder_output], enc_mask)
        output = self.layers[2](h2, h2)
        return output


#-----------------------------------------------------------------------------------
#BUILD TRANSFORMER MODELS
#-----------------------------------------------------------------------------------

#Encoder model class
class EncoderModel(nn.Module):
    def __init__(self, encoder, linear):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.linear = linear
  
    def forward(self, input):
        h = self.encoder(input, construct_mask(input.size(1)))
        return self.linear(h)


#Encoder-Decoder Transformer class
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, linear):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.linear = linear
  
    def forward(self, encoder_input, decoder_input):
        h = self.encoder(encoder_input, None)
        self.decoder.external_input = h
        out = self.decoder(decoder_input, None, construct_mask(decoder_input.size(1)))
        self.decoder.external_input = None
        return self.linear(out)









#Convolutional-Pooling class
"""class ConvPool(nn.Module):
    def __init__(self, filter_dim, in_features, out_features, pooling): #stride is set tp 1 and no padding
        super().__init__()
        self.filter_dim, self.indim, self.outdim, self.pool= filter_dim, in_features, out_features, int(pooling)
        self.filter = linear_layer((in_features*(filter_dim**2), out_features, True, False))
        self.relu = nn.ReLU()

    def convolution(self, input, dim0, dim1): #input: (batch_dim, feature_dim, image_dim0*image_dim1), dims = (image_dim0, image_dim1)
        x = input.transpose(-1,-2).reshape(-1, int(dim0), int(dim1), int(self.indim))
        w = int(self.filter_dim)
        output = torch.empty(x.size(0), (dim0-w +1), (dim1-w+1), self.indim*(w**2)).to(x)
        
        #A single small for loop over the size of the filter_dim/stride
        for i in range(w):
            for j in range(w):
                y = x[:,i:,j:] #make a step equal to the stride=1
                y = y[:,:(y.size(1)//w)*w, :(y.size(2)//w)*w] #truncate to right size
                output[:, i::w, j::w,:]= y.view(-1, y.size(1)//w, w, y.size(2)//w, w, self.indim).transpose(-3,-4).reshape(-1, y.size(1)//w, y.size(2)//w, self.indim*(w**2)) 
        return self.filter(output).view(output.size(0), -1, self.outdim).transpose(-1,-2)

    def pooling(self, input, dim0, dim1):
        w = int(self.pool)
        x = (input.view(-1, self.outdim, dim0, dim1))[:,:, :int((dim0//w)*w), :int((dim1//w)*w)] #pad down to appopriate size
        x = x.view(-1, self.outdim, dim0//w, w, dim1//w, w).transpose(-2,-3).reshape(-1, self.outdim, dim0//w, dim1//w, w**2)
        x_pool, _ = torch.max(x, dim=-1)
        return x_pool


    def forward(self, input, dim0, dim1): #flattened input
        out = self.relu(self.convolution(input, dim0, dim1))
        new_dim0, new_dim1 = (dim0-self.filter_dim +1), (dim1-self.filter_dim +1)
        if self.pool>1:
            out = self.pooling(out, new_dim0, new_dim1)
        return out.view(out.size(0), out.size(1),-1), (new_dim0//self.pool, new_dim1//self.pool)"""
