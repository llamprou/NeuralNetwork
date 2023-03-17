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
      graph = self.make_computational_graph(set(self.prev_grad_nodes), {self : self.prev_grad_nodes})
      sorted_graph = self.topological_sort(graph, []) #this is a list of the model parameters ordered according to which derivatives should be computed first
      for param in sorted_graph:
        for child, deriv in zip(param.prev_grad_nodes, param.grad_fn):
          if child is not None and deriv is not None:
            child.grad = deriv(param.grad) if child.grad is None else child.grad + deriv(self.grad)


    def make_computational_graph(self, nodes, graph):
      if len(nodes) ==0:
        return graph
      else:
        extend_graph ={n : n.prev_grad_nodes for n in nodes if n.prev_grad_nodes != [None]}
        graph = {**graph, **extend_graph}
        new_nodes = []
        for n in extend_graph.values():
          new_nodes+= n
        return self.make_computational_graph(set(new_nodes), graph)
      
    def topological_sort(self, graph, ordered):
        if len(graph)==0:
            return ordered
        else:
            values = []
            for v in graph.values():
                values.extend(v)
            for n in graph.keys():
                if n not in set(values):
                    ordered.append(n)
                    break
            del graph[n]
            return self.topological_sort(graph, ordered)



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
    output= Parameter(torch.sum(deriv.pow(2).reshape(-1)), grad_on =True)
    if self.mode == "mean":
      output.value = output.value/target.size(0)
    output.prev_grad_nodes=[input] if input.grad_on else []
    output.grad_fn = [MSELoss.backward(deriv)]

    return output

  @staticmethod
  def backward(deriv):
    def z_grad(back_inp):
      return 2*deriv
    return z_grad
    



"""class Parameter(object):
    def __init__(self, input, grad_on=False): #dim_in/out are lists
      self.value = input
      self.grad_on = grad_on
      self.grad= None
      self.prev_grad_nodes=[None]
      self.grad_fn = [None]

    def __call__(self):
      return self.value

    def backward(self): #this only works for tree-like computational graphs. To handle the general case I need to topologically sort the graph first.
      for param, gradfn in zip(self.prev_grad_nodes, self.grad_fn):
        if param is not None and gradfn is not None:
          param.grad = gradfn(self.grad) if param.grad is None else param.grad + gradfn(self.grad)
      for param in self.prev_grad_nodes:
        if param is not None:
          param.backward()"""