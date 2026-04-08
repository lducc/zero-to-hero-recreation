import random
from value import Value
from typing import List

class Module:
    def zero_grad(self):
        for param in self._params():
            param.grad = 0.0

    def _params(self):
        return []

class Neuron(Module):
    def __init__(self, nin: int, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x: List):
        assert len(x) == len(self.w), "Length of input must be the same of the 1st layer size"
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.relu() if self.nonlin else z

    def _params(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin: int, nout: int, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x: List[Value] | Value):
        out =  [neuron(x) for neuron in self.neurons]
        return out if len(out) > 1 else out[0]

    def _params(self):
        return [p for neuron in self.neurons for p in neuron._params()]

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]):
        ls = [nin] + nouts
        self.layers = [Layer(ls[i], ls[i + 1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x: List[Value] | Value):
        for layer in self.layers:
            x = layer(x)
        return x

    def _params(self):
        return [p for layer in self.layers for p in layer._params()]
