import torch
import math


class Parameter(object):
    def __init__(self, input, grad_on=False): #dim_in/out are lists
      self.value = input
      self.grad_on = grad_on
      self.grad= None
      self.prev_grad_nodes=[None]
      self.grad_fn = [None]

    def __call__(self):
      return self.value

    def backward(self):
      for param, gradfn in zip(self.prev_grad_nodes, self.grad_fn):
        if param is not None and gradfn is not None:
          param.grad = gradfn(self.grad) if param.grad is None else param.grad + gradfn(self.grad)
      for param in self.prev_grad_nodes:
        if param is not None:
          param.backward()




class Linear(object):
    def __init__(self, d_inp, d_hid, bias_on = True):
        self.d_inp, self.d_hid, self.bias_on = d_inp, d_hid, bias_on
        self.weight = Parameter(torch.randn(d_inp, d_hid)/math.sqrt(d_inp), grad_on=True)
        if self.bias_on is True:
            self.bias = Parameter(torch.randn(d_hid), grad_on=True)
        
    def __call__(self, input):
      return self.forward(input)

    def forward(self, input):
        out_value = torch.tensordot(input.value, self.weight.value, dims=([-1],[0])) + self.bias.value
        output= Parameter(out_value, grad_on=True)
        nodes= filter(lambda x: x[1].grad_on, zip(Linear.backward(input.value, self.weight.value),(self.weight, self.bias, input)))
        [output.grad_fn, output.prev_grad_nodes] = list(zip(*nodes))
        return output
    
    @staticmethod
    def backward(input, weight):
        def w_grad(back_inp):
            return torch.tensordot(input, back_inp, dims=([0],[0]))
        def b_grad(back_inp):
            return torch.sum(back_inp, dim=0)
        def z_grad(back_inp):
            return torch.tensordot(back_inp, weight, dims=([-1],[-1]))
        return w_grad, b_grad, z_grad



class MSELoss(object):
  def __init__(self, mode = "mean"):
    self.mode = mode
  
  def __call__(self, input, target):
    return self.forward(input, target)

  def forward(self, input, target):
    input = input
    deriv = input.value - target
    output= Parameter(torch.sum(deriv.pow(2).reshape(-1))/target.size(0), grad_on =True)
    output.prev_grad_nodes=[input] if input.grad_on else []
    output.grad_fn = [MSELoss.backward(deriv)]

    return output

  @staticmethod
  def backward(deriv):
    def z_grad(back_inp):
      return 2*deriv
    return z_grad
    
