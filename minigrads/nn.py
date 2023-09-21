import random
from typing import Any
from engine import Value
import math

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    def step(self):
        for p in self.parameters():
            p.data -= 0.1*p.grad
    # Template {abstract method}
    def parameters(self):
        return []

# Generating a single neuron that perform y = w*x + b
class Neuron(Module):

    def __init__(self, nin, nl=True):
        self.nin = nin
        self.nl = nl
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x): 
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        return act.tanh() if self.nl else act
    
    def parameters(self): return self.w + [self.b]

    def __repr__(self):
        return f'Neuron({len(self.w)})'
    
# Generating the nout neuron that perform linear transformation over input data
# Number of neuron
# We can letter stack them manually
class Layer(Module):
    def __init__(self, nin, nout, nl=True):
        self.nin = nin
        self.nl = nl
        self.nout = nout
        self.neurons = [Neuron(nin, nl) for _ in range(nout)]
    def __call__(self, x): return [n(x) for n in self.neurons]
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]
    # another implementation [self.neurons[i].parameters() for i in range(len(self.neurons))]
    def __repr__(self):
        return f"Layer({', '.join(str(n) for n in self.neurons)})"
    
class Sigmoid:
    def __init__(self):
        pass
    def forward(self, x): return [math.exp(-xi)/(1 + math.exp(-xi)) for xi in x]
        



# Optimizers
class Optim:
    def __init__(self, ps, lr=0.1):
        self.ps = ps
        self.lr = lr
    def step(self): 
        for ps in self.ps: 
            for p in ps:
                p.data -= self.lr*p.grad
